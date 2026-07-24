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
    Address, Bf16CircuitValue, Bf16DivInstr, Bf16DivIo, ExecutionRecord, Instruction,
    RecursionProgram, BF16_ABNORMAL_EXPONENT, BF16_MANTISSA_BITS,
};

use crate::builder::SP1RecursionAirBuilder;

pub const BF16_DIV_COLS: usize = core::mem::size_of::<Bf16DivCols<u8>>();
pub const BF16_DIV_PREPROCESSED_COLS: usize = core::mem::size_of::<Bf16DivPreprocessedCols<u8>>();

/// Main trace columns for Algorithm 2, lookup-based BF16 division.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16DivCols<T: Copy> {
    pub lhs: Bf16CircuitValue<T>,
    pub rhs: Bf16CircuitValue<T>,
    pub output: Bf16CircuitValue<T>,
    pub quotient: T,
    pub quotient_shift: T,
    pub denominator_is_zero: T,
    pub clamp: T,
}

/// Program columns holding memory addresses and the output multiplicity.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16DivPreprocessedCols<T: Copy> {
    pub is_real: T,
    pub addrs: Bf16DivIo<Address<T>>,
    pub mult: T,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bf16DivChip;

impl<F: Field> BaseAir<F> for Bf16DivChip {
    fn width(&self) -> usize {
        BF16_DIV_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Bf16DivChip {
    type Record = ExecutionRecord<F>;
    type Program = RecursionProgram<F>;

    fn name(&self) -> &'static str {
        "Bf16Div"
    }

    fn preprocessed_width(&self) -> usize {
        BF16_DIV_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program) -> Option<usize> {
        let count = program.event_counts.bf16_div_events;
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

        let instruction_count = program.event_counts.bf16_div_events;
        let rows = self.preprocessed_num_rows_with_instrs_len(program, instruction_count).unwrap();

        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                rows * BF16_DIV_PREPROCESSED_COLS,
            )
        };
        unsafe {
            core::ptr::write_bytes(values.as_mut_ptr(), 0, values.len());
        }

        let populate = |row: &mut [F], instruction: Bf16DivInstr<F>| {
            let Bf16DivInstr { addrs, mult } = instruction;
            let cols: &mut Bf16DivPreprocessedCols<F> = row.borrow_mut();
            *cols = Bf16DivPreprocessedCols { is_real: F::one(), addrs, mult };
        };
        for analyzed in program.inner.iter() {
            let (offset, instruction) = match analyzed.inner() {
                Instruction::Bf16Div(instruction) => (analyzed.offset(), *instruction),
                Instruction::Bf16MeanBatch(batch) => {
                    (analyzed.secondary_offset(), batch.div_instruction())
                }
                _ => continue,
            };
            let start = offset * BF16_DIV_PREPROCESSED_COLS;
            populate(&mut values[start..start + BF16_DIV_PREPROCESSED_COLS], instruction);
        }
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        Some(next_multiple_of_32(input.bf16_div_events.len(), height))
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

        let events = &input.bf16_div_events;
        let rows = <Self as MachineAir<F>>::num_rows(self, input).unwrap();
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut F, rows * BF16_DIV_COLS)
        };
        unsafe {
            let padding_start = events.len() * BF16_DIV_COLS;
            core::ptr::write_bytes(
                values[padding_start..].as_mut_ptr(),
                0,
                values.len() - padding_start,
            );
        }

        let populated = events.len() * BF16_DIV_COLS;
        values[..populated].par_chunks_mut(BF16_DIV_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut Bf16DivCols<F> = row.borrow_mut();
                *cols = Bf16DivCols {
                    lhs: event.lhs,
                    rhs: event.rhs,
                    output: event.output,
                    quotient: event.quotient,
                    quotient_shift: event.quotient_shift,
                    denominator_is_zero: event.denominator_is_zero,
                    clamp: event.clamp,
                };
            },
        );
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

impl<AB> Air<AB> for Bf16DivChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Bf16DivCols<AB::Var> = (*local).borrow();

        let preprocessed = builder.preprocessed();
        let preprocessed = preprocessed.row_slice(0);
        let program: &Bf16DivPreprocessedCols<AB::Var> = (*preprocessed).borrow();

        let mantissa_base = AB::F::from_canonical_u16(1 << (BF16_MANTISSA_BITS + 1));
        let twice_abnormal = AB::F::from_canonical_u16((2 * BF16_ABNORMAL_EXPONENT) as u16);
        let one: AB::Expr = AB::F::one().into();

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

        // q := Div(m_lhs || m_rhs).
        builder.send_bf16_div(
            local.lhs.mantissa,
            local.rhs.mantissa,
            local.quotient,
            program.is_real,
        );

        // q_0 := RShift(1, q); u := q - 2^(M+1) q_0.
        builder.send_bf16_rshift(1, local.quotient, local.quotient_shift, program.is_real);
        let normalized_mantissa = local.quotient - local.quotient_shift * mantissa_base;

        // b := 1 - RShift(0, m_rhs).
        builder.send_bf16_rshift(
            0,
            local.rhs.mantissa,
            one - local.denominator_is_zero,
            program.is_real,
        );

        // e := e_lhs - e_rhs - q_0 + 2 E_abnormal b.
        let intermediate_exponent = local.lhs.exponent - local.rhs.exponent - local.quotient_shift
            + local.denominator_is_zero * twice_abnormal;

        builder.send_bf16_exp(
            intermediate_exponent.clone(),
            local.output.exponent,
            program.is_real,
        );
        builder.send_bf16_clamp(intermediate_exponent, local.clamp, program.is_real);

        builder.assert_eq(
            local.output.sign,
            local.lhs.sign + local.rhs.sign - local.lhs.sign * local.rhs.sign * AB::F::two(),
        );

        builder.send_bf16_round(
            local.clamp,
            normalized_mantissa,
            local.output.mantissa,
            program.is_real,
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
        instruction as instr, linear_program, Bf16CircuitValueInt, Bf16DivWitness, Block, Executor,
        Instruction, MemAccessKind, D,
    };

    use crate::{machine::RecursionAir, test::run_test_recursion};

    const BF16_DIV_TEST_DATA: &str = include_str!("../../../../data/bf16/div");
    const BF16_MAX_FINITE_MAGNITUDE: u16 = 0x7f7f;
    const BF16_INFINITY_MAGNITUDE: u16 = 0x7f80;

    fn parse_hex_word(word: Option<&str>, line_number: usize, name: &str) -> u16 {
        let word = word.unwrap_or_else(|| panic!("missing {name} on BF16 div line {line_number}"));
        u16::from_str_radix(word, 16).unwrap_or_else(|error| {
            panic!("invalid {name} on BF16 div line {line_number}: {error}")
        })
    }

    #[tokio::test]
    async fn prove_bf16_div() {
        type A = RecursionAir<SP1Field, 3, 2>;

        let num_cases = BF16_DIV_TEST_DATA.lines().count();
        assert!(num_cases > 0, "BF16 div test data must not be empty");

        let mut instructions = Vec::<Instruction<SP1Field>>::with_capacity(num_cases * 4);
        let mut included_count = 0;
        let mut excluded_overflow_count = 0;
        let mut excluded_infinite_denominator_count = 0;
        let mut mismatch_count = 0;
        let mut mismatch_samples = Vec::new();
        for (index, line) in BF16_DIV_TEST_DATA.lines().enumerate() {
            let line_number = index + 1;
            let mut words = line.split_ascii_whitespace();
            let lhs = parse_hex_word(words.next(), line_number, "lhs");
            let rhs = parse_hex_word(words.next(), line_number, "rhs");
            let expected_raw = parse_hex_word(words.next(), line_number, "expected output");
            assert!(words.next().is_none(), "extra data on BF16 div line {line_number}");

            let expected_circuit = Bf16CircuitValueInt::decode(expected_raw);
            let expected = Bf16CircuitValueInt::encode(
                expected_circuit.sign,
                expected_circuit.exponent,
                expected_circuit.mantissa,
            );
            let actual = Bf16DivWitness::new(lhs, rhs).output.raw;

            let is_round_toward_zero_overflow = (expected & 0x7fff) == BF16_MAX_FINITE_MAGNITUDE
                && (actual & 0x7fff) == BF16_INFINITY_MAGNITUDE
                && (expected & 0x8000) == (actual & 0x8000);
            if is_round_toward_zero_overflow {
                excluded_overflow_count += 1;
                continue;
            }

            // The paper deliberately merges infinity and NaN during initialization. Consequently,
            // it cannot reproduce the IEEE distinction where finite / infinity is signed zero,
            // while finite / NaN is abnormal.
            let is_infinite_denominator_semantic_difference = (rhs & 0x7fff)
                == BF16_INFINITY_MAGNITUDE
                && (expected & 0x7fff) == 0
                && (actual & 0x7fff) == BF16_INFINITY_MAGNITUDE
                && (expected & 0x8000) == (actual & 0x8000);
            if is_infinite_denominator_semantic_difference {
                excluded_infinite_denominator_count += 1;
                continue;
            }

            if actual != expected {
                mismatch_count += 1;
                if mismatch_samples.len() < 16 {
                    mismatch_samples.push(format!(
                        "line {line_number}: {lhs:04X} / {rhs:04X}, data {expected_raw:04X}, canonical expected {expected:04X}, got {actual:04X}"
                    ));
                }
            }

            let base = (included_count * 3) as u32;
            instructions.extend([
                instr::mem(MemAccessKind::Write, 1, base, lhs as u32),
                instr::mem(MemAccessKind::Write, 1, base + 1, rhs as u32),
                instr::bf16_div(1, base + 2, base, base + 1),
                instr::mem(MemAccessKind::Read, 1, base + 2, expected as u32),
            ]);
            included_count += 1;
        }

        assert!(included_count > 0, "BF16 div test data must contain supported vectors");
        assert_eq!(
            mismatch_count,
            0,
            "{mismatch_count} supported BF16 div test vectors failed:\n{}",
            mismatch_samples.join("\n")
        );
        println!(
            "BF16 div vectors: {included_count} included, {excluded_overflow_count} roundTowardZero overflow vectors excluded, {excluded_infinite_denominator_count} IEEE infinite-denominator vectors excluded"
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
}
