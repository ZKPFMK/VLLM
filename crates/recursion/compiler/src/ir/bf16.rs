use sp1_primitives::SP1Field;

use super::{Builder, Config, DslIr, Felt};

impl<C: Config> Builder<C> {
    /// Multiply two raw 16-bit BF16 encodings using the `VeriLLM` recursion chip.
    ///
    /// Both operands and the returned value are stored as `Felt<SP1Field>`, but the BF16 lookup
    /// relation constrains each of them to a valid 16-bit encoding.
    pub fn bf16_mul(&mut self, lhs: Felt<SP1Field>, rhs: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Mul(output, lhs, rhs));
        output
    }

    /// Divide two raw 16-bit BF16 encodings using Algorithm 2 from `VeriLLM`.
    ///
    /// Both operands and the returned value are stored as `Felt<SP1Field>`, but the BF16 lookup
    /// relation constrains each of them to a valid 16-bit encoding.
    pub fn bf16_div(&mut self, lhs: Felt<SP1Field>, rhs: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Div(output, lhs, rhs));
        output
    }

    /// Add two raw 16-bit BF16 encodings using Algorithm 3 from `VeriLLM`.
    pub fn bf16_add(&mut self, lhs: Felt<SP1Field>, rhs: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Add(output, lhs, rhs));
        output
    }

    /// Subtract two raw 16-bit BF16 encodings using the unified Algorithm 3 chip.
    pub fn bf16_sub(&mut self, lhs: Felt<SP1Field>, rhs: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Sub(output, lhs, rhs));
        output
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_algebra::AbstractField;
    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1ExtensionField};
    use sp1_recursion_executor::Executor;

    use crate::circuit::{AsmBuilder, AsmCompiler};

    use super::*;

    #[test]
    fn compiles_and_executes_bf16_mul() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let lhs: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x3fc0));
        let rhs: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0xc000));
        let output = builder.bf16_mul(lhs, rhs);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0xc040));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 1);
    }

    #[test]
    fn compiles_and_executes_bf16_div() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let lhs: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x4040));
        let rhs: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0xc000));
        let output = builder.bf16_div(lhs, rhs);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0xbfc0));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_div_events.len(), 1);
    }

    #[test]
    fn compiles_and_executes_bf16_add_sub() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let three: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x4040));
        let two: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x4000));
        let sum = builder.bf16_add(three, two);
        let difference = builder.bf16_sub(three, two);
        builder.assert_felt_eq(sum, SP1Field::from_canonical_u16(0x40a0));
        builder.assert_felt_eq(difference, SP1Field::from_canonical_u16(0x3f80));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), 2);
    }
}
