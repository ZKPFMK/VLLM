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
    Address, Bf16UnaryInstr, Bf16UnaryIo, ExecutionRecord, Instruction, RecursionProgram,
};

use crate::builder::SP1RecursionAirBuilder;

pub const BF16_UNARY_COLS: usize = core::mem::size_of::<Bf16UnaryCols<u8>>();
pub const BF16_UNARY_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Bf16UnaryPreprocessedCols<u8>>();

/// Main trace columns shared by raw-to-raw unary BF16 lookups.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16UnaryCols<T: Copy> {
    pub input: T,
    pub output: T,
}

/// Program columns selecting the lookup opcode and memory interface.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Bf16UnaryPreprocessedCols<T: Copy> {
    pub is_real: T,
    pub opcode: T,
    pub addrs: Bf16UnaryIo<Address<T>>,
    pub mult: T,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bf16UnaryChip;

impl<F: Field> BaseAir<F> for Bf16UnaryChip {
    fn width(&self) -> usize {
        BF16_UNARY_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Bf16UnaryChip {
    type Record = ExecutionRecord<F>;
    type Program = RecursionProgram<F>;

    fn name(&self) -> &'static str {
        "Bf16Unary"
    }

    fn preprocessed_width(&self) -> usize {
        BF16_UNARY_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program) -> Option<usize> {
        let count = program
            .inner
            .iter()
            .filter(|instruction| matches!(instruction.inner(), Instruction::Bf16Unary(_)))
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
                Instruction::Bf16Unary(instruction) => Some(instruction),
                _ => None,
            })
            .collect::<Vec<_>>();
        let rows = self.preprocessed_num_rows_with_instrs_len(program, instructions.len()).unwrap();

        let values = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut F,
                rows * BF16_UNARY_PREPROCESSED_COLS,
            )
        };
        unsafe {
            let padding_start = instructions.len() * BF16_UNARY_PREPROCESSED_COLS;
            core::ptr::write_bytes(
                values[padding_start..].as_mut_ptr(),
                0,
                values.len() - padding_start,
            );
        }

        let populated = instructions.len() * BF16_UNARY_PREPROCESSED_COLS;
        values[..populated]
            .par_chunks_mut(BF16_UNARY_PREPROCESSED_COLS)
            .zip_eq(instructions)
            .for_each(|(row, instruction)| {
                let Bf16UnaryInstr { opcode, addrs, mult } = instruction;
                let cols: &mut Bf16UnaryPreprocessedCols<F> = row.borrow_mut();
                *cols = Bf16UnaryPreprocessedCols {
                    is_real: F::one(),
                    opcode: F::from_canonical_u8(opcode.lookup_opcode()),
                    addrs: *addrs,
                    mult: *mult,
                };
            });
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        Some(next_multiple_of_32(input.bf16_unary_events.len(), height))
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

        let events = &input.bf16_unary_events;
        let rows = <Self as MachineAir<F>>::num_rows(self, input).unwrap();
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut F, rows * BF16_UNARY_COLS)
        };
        unsafe {
            let padding_start = events.len() * BF16_UNARY_COLS;
            core::ptr::write_bytes(
                values[padding_start..].as_mut_ptr(),
                0,
                values.len() - padding_start,
            );
        }

        let populated = events.len() * BF16_UNARY_COLS;
        values[..populated].par_chunks_mut(BF16_UNARY_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut Bf16UnaryCols<F> = row.borrow_mut();
                *cols = Bf16UnaryCols { input: event.input, output: event.output };
            },
        );
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

impl<AB> Air<AB> for Bf16UnaryChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Bf16UnaryCols<AB::Var> = (*local).borrow();

        let preprocessed = builder.preprocessed();
        let preprocessed = preprocessed.row_slice(0);
        let program: &Bf16UnaryPreprocessedCols<AB::Var> = (*preprocessed).borrow();

        builder.receive_single(program.addrs.input, local.input, program.is_real);
        builder.send_bf16_lookup(
            program.opcode,
            local.input,
            local.output,
            AB::F::zero(),
            AB::F::zero(),
            program.is_real,
        );
        builder.send_single(program.addrs.output, local.output, program.mult);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1Field};
    use sp1_recursion_executor::{
        instruction as instr, linear_program, Bf16UnaryOpcode, Bf16UnaryWitness, Block, Executor,
        Instruction, MemAccessKind, D,
    };

    use crate::{machine::RecursionAir, test::run_test_recursion};

    #[tokio::test]
    async fn prove_bf16_square() {
        type A = RecursionAir<SP1Field, 3, 2>;

        let inputs =
            [0x0000, 0x8000, 0x0001, 0x007f, 0x3f80, 0x3fc0, 0xc000, 0x7f7f, 0x7f80, 0xffff];
        let mut instructions = Vec::<Instruction<SP1Field>>::with_capacity(inputs.len() * 3);
        for (index, input) in inputs.into_iter().enumerate() {
            let output = Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, input).output;
            let base = (index * 2) as u32;
            instructions.extend([
                instr::mem(MemAccessKind::Write, 1, base, input as u32),
                instr::bf16_square(1, base + 1, base),
                instr::mem(MemAccessKind::Read, 1, base + 1, output as u32),
            ]);
        }

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
    async fn prove_bf16_rsqrt() {
        type A = RecursionAir<SP1Field, 3, 2>;

        let inputs =
            [0x0000, 0x0001, 0x0080, 0x3e80, 0x3f80, 0x4000, 0x4080, 0x7f7f, 0x7f80, 0xbf80];
        let mut instructions = Vec::<Instruction<SP1Field>>::with_capacity(inputs.len() * 3);
        for (index, input) in inputs.into_iter().enumerate() {
            let output = Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, input).output;
            let base = (index * 2) as u32;
            instructions.extend([
                instr::mem(MemAccessKind::Write, 1, base, input as u32),
                instr::bf16_rsqrt(1, base + 1, base),
                instr::mem(MemAccessKind::Read, 1, base + 1, output as u32),
            ]);
        }

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
    async fn prove_bf16_exponential() {
        type A = RecursionAir<SP1Field, 3, 2>;

        let inputs = [
            0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x42b2, 0xc2d0, 0x7f80, 0xff80, 0x7fc1,
        ];
        let mut instructions = Vec::<Instruction<SP1Field>>::with_capacity(inputs.len() * 3);
        for (index, input) in inputs.into_iter().enumerate() {
            let output = Bf16UnaryWitness::new(Bf16UnaryOpcode::Exponential, input).output;
            let base = (index * 2) as u32;
            instructions.extend([
                instr::mem(MemAccessKind::Write, 1, base, input as u32),
                instr::bf16_exponential(1, base + 1, base),
                instr::mem(MemAccessKind::Read, 1, base + 1, output as u32),
            ]);
        }

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
    async fn prove_bf16_softmax_pipeline() {
        type A = RecursionAir<SP1Field, 3, 2>;

        // softmax([-1, 0]) = [0x3e89, 0x3f3b] with BF16 round-toward-zero after every step.
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0xbf80),
            instr::mem(MemAccessKind::Write, 1, 1, 0x0000),
            instr::bf16_exponential(2, 2, 0),
            instr::bf16_exponential(2, 3, 1),
            instr::bf16_add(2, 4, 2, 3),
            instr::bf16_div(1, 5, 2, 4),
            instr::bf16_div(1, 6, 3, 4),
            instr::mem(MemAccessKind::Read, 1, 5, 0x3e89),
            instr::mem(MemAccessKind::Read, 1, 6, 0x3f3b),
        ];

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
