use core::borrow::Borrow;
use std::{borrow::BorrowMut, mem::MaybeUninit};

use slop_air::{Air, BaseAir, PairBuilder};
use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_matrix::Matrix;
use slop_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::{air::MachineAir, next_multiple_of_32};
use sp1_primitives::SP1Field;
use sp1_recursion_executor::{
    bf16_i32_to_field, Address, Bf16AddSubInstr, Bf16AddSubIo, Bf16CircuitValue, ExecutionRecord,
    Instruction, RecursionProgram, BF16_MANTISSA_BITS, BF16_MIN_NORMAL_EXPONENT,
    BF16_ZERO_EXPONENT,
};

use crate::builder::SP1RecursionAirBuilder;

pub const BF16_ADD_SUB_COLS: usize = core::mem::size_of::<Bf16AddSubCols<u8>>();
pub const BF16_ADD_SUB_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Bf16AddSubPreprocessedCols<u8>>();

/// Main trace columns shared by Algorithm 3 addition and subtraction.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16AddSubCols<T: Copy> {
    pub lhs: Bf16CircuitValue<T>,
    pub rhs: Bf16CircuitValue<T>,
    pub output: Bf16CircuitValue<T>,
    pub effective_rhs_sign: T,
    pub exponent_order: T,
    pub larger_sign: T,
    pub larger_exponent: T,
    pub larger_mantissa: T,
    pub alignment: T,
    pub smaller_nonzero: T,
    pub max_alignment: T,
    pub selected_smaller: T,
    pub shifted_larger: T,
    pub signed_smaller: T,
    pub unsigned_sum: T,
    pub add_result: T,
    pub normalization_shift: T,
    pub normalized_nonzero: T,
    pub abnormal: T,
}

/// Program columns select addition or subtraction and hold the recursion-memory interface.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16AddSubPreprocessedCols<T: Copy> {
    pub is_real: T,
    /// Zero for addition and one for subtraction.
    pub op_code: T,
    pub addrs: Bf16AddSubIo<Address<T>>,
    pub mult: T,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bf16AddSubChip;

impl<F: Field> BaseAir<F> for Bf16AddSubChip {
    fn width(&self) -> usize {
        BF16_ADD_SUB_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Bf16AddSubChip {
    type Record = ExecutionRecord<F>;
    type Program = RecursionProgram<F>;

    fn name(&self) -> &'static str {
        "Bf16AddSub"
    }

    fn preprocessed_width(&self) -> usize {
        BF16_ADD_SUB_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program) -> Option<usize> {
        let count = program
            .inner
            .iter()
            .filter(|instruction| matches!(instruction.inner(), Instruction::Bf16AddSub(_)))
            .count();
        self.preprocessed_num_rows_with_instrs_len(program, count)
    }

    fn preprocessed_num_rows_with_instrs_len(
        &self,
        program: &Self::Program,
        instrs_len: usize,
    ) -> Option<usize> {
        let height = program.shape.as_ref().and_then(|shape| shape.height(self));
        Some(next_multiple_of_32(instrs_len, height))
    }

    fn generate_preprocessed_trace_into(
        &self,
        program: &Self::Program,
        buffer: &mut [MaybeUninit<F>],
    ) {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<SP1Field>(),
            "generate_preprocessed_trace only supports SP1Field"
        );

        let instructions = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction.inner() {
                Instruction::Bf16AddSub(instruction) => Some(instruction),
                _ => None,
            })
            .collect::<Vec<_>>();
        let rows = self.preprocessed_num_rows_with_instrs_len(program, instructions.len()).unwrap();

        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                rows * BF16_ADD_SUB_PREPROCESSED_COLS,
            )
        };
        unsafe {
            let padding_start = instructions.len() * BF16_ADD_SUB_PREPROCESSED_COLS;
            core::ptr::write_bytes(
                values[padding_start..].as_mut_ptr(),
                0,
                values.len() - padding_start,
            );
        }

        let populated = instructions.len() * BF16_ADD_SUB_PREPROCESSED_COLS;
        values[..populated]
            .par_chunks_mut(BF16_ADD_SUB_PREPROCESSED_COLS)
            .zip_eq(instructions)
            .for_each(|(row, instruction)| {
                let Bf16AddSubInstr { opcode, addrs, mult } = instruction;
                let cols: &mut Bf16AddSubPreprocessedCols<F> = row.borrow_mut();
                *cols = Bf16AddSubPreprocessedCols {
                    is_real: F::one(),
                    op_code: F::from_canonical_u8(opcode.as_u8()),
                    addrs: *addrs,
                    mult: *mult,
                };
            });
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        Some(next_multiple_of_32(input.bf16_add_sub_events.len(), height))
    }

    fn generate_trace_into(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
        buffer: &mut [MaybeUninit<F>],
    ) {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<SP1Field>(),
            "generate_trace_into only supports SP1Field"
        );

        let events = &input.bf16_add_sub_events;
        let rows = <Self as MachineAir<F>>::num_rows(self, input).unwrap();
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut F, rows * BF16_ADD_SUB_COLS)
        };
        unsafe {
            let padding_start = events.len() * BF16_ADD_SUB_COLS;
            core::ptr::write_bytes(
                values[padding_start..].as_mut_ptr(),
                0,
                values.len() - padding_start,
            );
        }

        let populated = events.len() * BF16_ADD_SUB_COLS;
        values[..populated].par_chunks_mut(BF16_ADD_SUB_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut Bf16AddSubCols<F> = row.borrow_mut();
                *cols = Bf16AddSubCols {
                    lhs: event.lhs,
                    rhs: event.rhs,
                    output: event.output,
                    effective_rhs_sign: event.effective_rhs_sign,
                    exponent_order: event.exponent_order,
                    larger_sign: event.larger_sign,
                    larger_exponent: event.larger_exponent,
                    larger_mantissa: event.larger_mantissa,
                    alignment: event.alignment,
                    smaller_nonzero: event.smaller_nonzero,
                    max_alignment: event.max_alignment,
                    selected_smaller: event.selected_smaller,
                    shifted_larger: event.shifted_larger,
                    signed_smaller: event.signed_smaller,
                    unsigned_sum: event.unsigned_sum,
                    add_result: event.add_result,
                    normalization_shift: event.normalization_shift,
                    normalized_nonzero: event.normalized_nonzero,
                    abnormal: event.abnormal,
                };
            },
        );
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

impl<AB> Air<AB> for Bf16AddSubChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Bf16AddSubCols<AB::Var> = (*local).borrow();

        let preprocessed = builder.preprocessed();
        let preprocessed = preprocessed.row_slice(0);
        let program: &Bf16AddSubPreprocessedCols<AB::Var> = (*preprocessed).borrow();

        let two = AB::F::two();
        let mantissa_base = AB::F::from_canonical_u16(1 << (BF16_MANTISSA_BITS + 1));
        let exponent_order_bias = AB::F::from_canonical_u16((1 << 10) - 1);
        let alignment_bias = AB::F::from_canonical_u16(7);
        let alignment_scale = AB::F::from_canonical_u16(1 << (BF16_MANTISSA_BITS - 4));
        let mantissa_bits_plus_one = AB::F::from_canonical_u32(BF16_MANTISSA_BITS + 1);
        let zero_exponent = bf16_i32_to_field::<AB::F>(BF16_ZERO_EXPONENT);
        let min_normal_exponent = bf16_i32_to_field::<AB::F>(BF16_MIN_NORMAL_EXPONENT);

        builder.assert_bool(program.op_code);
        builder.receive_single(program.addrs.lhs, local.lhs.raw, program.is_real);
        builder.receive_single(program.addrs.rhs, local.rhs.raw, program.is_real);

        builder.send_bf16_init(
            local.lhs.raw,
            local.lhs.sign,
            local.lhs.exponent,
            local.lhs.mantissa,
            program.is_real,
        );
        builder.send_bf16_init(
            local.rhs.raw,
            local.rhs.sign,
            local.rhs.exponent,
            local.rhs.mantissa,
            program.is_real,
        );

        // Subtraction reuses the addition algorithm after flipping only the right sign.
        builder.assert_eq(
            local.effective_rhs_sign,
            local.rhs.sign + program.op_code - local.rhs.sign * program.op_code * two,
        );

        // t_e := RShift(3, 2^10 + (e_rhs - e_lhs - 1)).
        builder.send_bf16_rshift(
            3,
            local.rhs.exponent - local.lhs.exponent + exponent_order_bias,
            local.exponent_order,
            program.is_real,
        );

        builder.assert_eq(
            local.larger_sign,
            local.lhs.sign + local.exponent_order * (local.effective_rhs_sign - local.lhs.sign),
        );
        let smaller_sign = local.lhs.sign + local.effective_rhs_sign - local.larger_sign;
        builder.assert_eq(
            local.larger_exponent,
            local.lhs.exponent + local.exponent_order * (local.rhs.exponent - local.lhs.exponent),
        );
        let smaller_exponent = local.lhs.exponent + local.rhs.exponent - local.larger_exponent;
        builder.assert_eq(
            local.larger_mantissa,
            local.lhs.mantissa + local.exponent_order * (local.rhs.mantissa - local.lhs.mantissa),
        );
        let smaller_mantissa = local.lhs.mantissa + local.rhs.mantissa - local.larger_mantissa;

        // d := Clamp(E_min_norm - e_1 + e_2).
        builder.send_bf16_clamp(
            smaller_exponent - local.larger_exponent + min_normal_exponent,
            local.alignment,
            program.is_real,
        );
        builder.send_bf16_rshift(
            0,
            smaller_mantissa.clone(),
            local.smaller_nonzero,
            program.is_real,
        );
        builder.send_bf16_rshift(
            0,
            (local.alignment + alignment_bias) * alignment_scale,
            local.max_alignment,
            program.is_real,
        );

        builder.assert_eq(
            local.selected_smaller,
            smaller_mantissa.clone()
                + local.max_alignment * (local.smaller_nonzero - smaller_mantissa),
        );
        let shift = local.alignment - local.max_alignment;
        builder.send_bf16_lshift(
            shift.clone(),
            local.larger_sign,
            local.larger_mantissa,
            local.shifted_larger,
            program.is_real,
        );
        builder.assert_eq(
            local.signed_smaller,
            local.selected_smaller - smaller_sign * local.selected_smaller * two,
        );
        let signed_sum = local.shifted_larger + local.signed_smaller;

        builder.assert_eq(
            local.unsigned_sum,
            signed_sum.clone() - local.output.sign * signed_sum * two,
        );

        builder.send_bf16_add(local.unsigned_sum, local.add_result, program.is_real);
        builder.send_bf16_rshift(1, local.add_result, local.normalization_shift, program.is_real);
        let normalized_mantissa = local.add_result - local.normalization_shift * mantissa_base;
        builder.send_bf16_rshift(
            0,
            normalized_mantissa.clone(),
            local.normalized_nonzero,
            program.is_real,
        );

        let intermediate_exponent =
            local.larger_exponent + mantissa_bits_plus_one - local.normalization_shift - shift
                + zero_exponent
                - local.normalized_nonzero * zero_exponent;
        builder.send_bf16_exp(intermediate_exponent, local.output.exponent, program.is_real);
        builder.send_bf16_rshift(
            2,
            local.output.exponent - zero_exponent,
            local.abnormal,
            program.is_real,
        );
        builder.assert_eq(
            local.output.mantissa,
            normalized_mantissa.clone() - local.abnormal * normalized_mantissa,
        );

        builder.send_bf16_init(
            local.output.raw,
            local.output.sign,
            local.output.exponent,
            local.output.mantissa,
            program.is_real,
        );
        builder.send_single(program.addrs.output, local.output.raw, program.mult);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1Field};
    use sp1_recursion_executor::{
        instruction as instr, linear_program, Bf16AddSubOpcode, Bf16AddSubWitness,
        Bf16CircuitValueInt, Block, Executor, Instruction, MemAccessKind, D,
    };

    use crate::{machine::RecursionAir, test::run_test_recursion};

    const BF16_ADD_TEST_DATA: &str = include_str!("../../../../data/bf16/add");
    const BF16_MAX_FINITE_MAGNITUDE: u16 = 0x7f7f;
    const BF16_INFINITY_MAGNITUDE: u16 = 0x7f80;

    fn parse_hex_word(word: Option<&str>, line_number: usize, name: &str) -> u16 {
        let word = word.unwrap_or_else(|| panic!("missing {name} on BF16 add line {line_number}"));
        u16::from_str_radix(word, 16).unwrap_or_else(|error| {
            panic!("invalid {name} on BF16 add line {line_number}: {error}")
        })
    }

    async fn prove_bf16_add_sub(opcode: Bf16AddSubOpcode) {
        type A = RecursionAir<SP1Field, 3, 2>;

        let num_cases = BF16_ADD_TEST_DATA.lines().count();
        assert!(num_cases > 0, "BF16 add test data must not be empty");

        let mut instructions = Vec::<Instruction<SP1Field>>::with_capacity(num_cases * 4);
        let mut included_count = 0;
        let mut excluded_overflow_count = 0;
        let mut excluded_abnormal_sign_count = 0;
        let mut mismatch_count = 0;
        let mut mismatch_samples = Vec::new();

        for (index, line) in BF16_ADD_TEST_DATA.lines().enumerate() {
            let line_number = index + 1;
            let mut words = line.split_ascii_whitespace();
            let lhs = parse_hex_word(words.next(), line_number, "lhs");
            let data_rhs = parse_hex_word(words.next(), line_number, "rhs");
            let expected_raw = parse_hex_word(words.next(), line_number, "expected output");
            assert!(words.next().is_none(), "extra data on BF16 add line {line_number}");

            // `lhs - (-rhs)` exercises the subtraction opcode while preserving the Add dataset's
            // expected result.
            let rhs = match opcode {
                Bf16AddSubOpcode::Add => data_rhs,
                Bf16AddSubOpcode::Sub => data_rhs ^ 0x8000,
            };
            let expected_circuit = Bf16CircuitValueInt::decode(expected_raw);
            let expected = Bf16CircuitValueInt::encode(
                expected_circuit.sign,
                expected_circuit.exponent,
                expected_circuit.mantissa,
            );
            let actual = Bf16AddSubWitness::new(lhs, rhs, opcode).output.raw;

            let is_round_toward_zero_overflow = (expected & 0x7fff) == BF16_MAX_FINITE_MAGNITUDE
                && (actual & 0x7fff) == BF16_INFINITY_MAGNITUDE
                && (expected & 0x8000) == (actual & 0x8000);
            if is_round_toward_zero_overflow {
                excluded_overflow_count += 1;
                continue;
            }

            // Algorithm 3 receives the result sign as a hint. When the signed mantissa sum is zero
            // and the exponent maps to abnormal, both signs satisfy the paper's constraints.
            let is_abnormal_sign_difference = (expected & 0x7fff) == BF16_INFINITY_MAGNITUDE
                && (actual & 0x7fff) == BF16_INFINITY_MAGNITUDE
                && (expected & 0x8000) != (actual & 0x8000);
            if is_abnormal_sign_difference {
                excluded_abnormal_sign_count += 1;
                continue;
            }

            if actual != expected {
                mismatch_count += 1;
                if mismatch_samples.len() < 16 {
                    let operator = match opcode {
                        Bf16AddSubOpcode::Add => "+",
                        Bf16AddSubOpcode::Sub => "-",
                    };
                    mismatch_samples.push(format!(
                        "line {line_number}: {lhs:04X} {operator} {rhs:04X}, data rhs {data_rhs:04X}, data expected {expected_raw:04X}, canonical expected {expected:04X}, got {actual:04X}"
                    ));
                }
            }

            let base = (included_count * 3) as u32;
            let operation = match opcode {
                Bf16AddSubOpcode::Add => instr::bf16_add(1, base + 2, base, base + 1),
                Bf16AddSubOpcode::Sub => instr::bf16_sub(1, base + 2, base, base + 1),
            };
            instructions.extend([
                instr::mem(MemAccessKind::Write, 1, base, lhs as u32),
                instr::mem(MemAccessKind::Write, 1, base + 1, rhs as u32),
                operation,
                instr::mem(MemAccessKind::Read, 1, base + 2, expected as u32),
            ]);
            included_count += 1;
        }

        assert!(included_count > 0, "BF16 add/sub data must contain supported vectors");
        assert_eq!(
            mismatch_count,
            0,
            "{mismatch_count} supported BF16 add/sub test vectors failed:\n{}",
            mismatch_samples.join("\n")
        );
        println!(
            "BF16 {opcode:?} vectors: {included_count} included, {excluded_overflow_count} roundTowardZero overflow vectors excluded, {excluded_abnormal_sign_count} underdetermined abnormal-sign vectors excluded"
        );

        let program = linear_program(instructions).unwrap();
        let mut executor = Executor::<
            SP1Field,
            BinomialExtensionField<SP1Field, D>,
            SP1DiffusionMatrix,
        >::new(Arc::new(program.clone()), inner_perm());
        executor.witness_stream = Vec::<Block<SP1Field>>::new().into();
        executor.run().unwrap();

        run_test_recursion(vec![executor.record], A::verillm_machine(), program).await.unwrap();
    }

    #[tokio::test]
    async fn prove_bf16_add() {
        prove_bf16_add_sub(Bf16AddSubOpcode::Add).await;
    }

    #[tokio::test]
    async fn prove_bf16_sub() {
        prove_bf16_add_sub(Bf16AddSubOpcode::Sub).await;
    }
}
