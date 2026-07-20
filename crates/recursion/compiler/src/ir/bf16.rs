use slop_algebra::AbstractField;
use sp1_primitives::SP1Field;

use super::{Builder, Config, DslIr, Felt};

/// Encode a positive integer as BF16 using the same round-toward-zero policy as the arithmetic
/// chips. The common GPT-2 dimensions (including 768 and 3072) are represented exactly.
fn usize_to_bf16_raw(value: usize) -> u16 {
    assert!(value > 0, "BF16 integer conversion requires a positive value");

    let exponent = value.ilog2();
    let mantissa = if exponent <= 7 { value << (7 - exponent) } else { value >> (exponent - 7) };
    debug_assert!((128..256).contains(&mantissa));

    (((exponent + 127) as u16) << 7) | (mantissa as u16 - 128)
}

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

    /// Square one raw 16-bit BF16 encoding with a single table query.
    pub fn bf16_square(&mut self, input: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Square(output, input));
        output
    }

    /// Compute the reciprocal square root of one raw 16-bit BF16 encoding with a single lookup.
    pub fn bf16_rsqrt(&mut self, input: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Rsqrt(output, input));
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

    /// Compute the mean of a non-empty BF16 vector.
    ///
    /// Values are added from left to right, and the final sum is divided by the BF16 encoding of
    /// `values.len()`.
    pub fn bf16_mean(&mut self, values: &[Felt<SP1Field>]) -> Felt<SP1Field> {
        assert!(!values.is_empty(), "BF16 mean requires at least one value");

        let mut sum = values[0];
        for &value in &values[1..] {
            sum = self.bf16_add(sum, value);
        }

        let divisor_raw = usize_to_bf16_raw(values.len());
        let divisor: Felt<SP1Field> = self.constant(SP1Field::from_canonical_u16(divisor_raw));
        self.bf16_div(sum, divisor)
    }

    /// Compute the population variance of a non-empty BF16 vector around `mean`.
    ///
    /// Each squared difference is evaluated by one raw BF16 square lookup, then their mean is
    /// computed from left to right using [`Self::bf16_mean`].
    pub fn bf16_variance(
        &mut self,
        values: &[Felt<SP1Field>],
        mean: Felt<SP1Field>,
    ) -> Felt<SP1Field> {
        assert!(!values.is_empty(), "BF16 variance requires at least one value");

        let mut squared_differences = Vec::with_capacity(values.len());
        for &value in values {
            let difference = self.bf16_sub(value, mean);
            squared_differences.push(self.bf16_square(difference));
        }
        self.bf16_mean(&squared_differences)
    }

    /// Apply GPT-2-style layer normalization to a non-empty BF16 vector.
    ///
    /// The mean and population variance are accumulated from left to right. Centered values are
    /// retained and reused after the single reciprocal-square-root lookup. Every operation rounds
    /// toward zero to BF16 before its result is consumed by the next operation.
    pub fn bf16_layer_norm(
        &mut self,
        values: &[Felt<SP1Field>],
        weight: &[Felt<SP1Field>],
        bias: &[Felt<SP1Field>],
        epsilon: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!values.is_empty(), "BF16 layer norm requires at least one value");
        assert_eq!(values.len(), weight.len(), "BF16 layer norm weight length mismatch");
        assert_eq!(values.len(), bias.len(), "BF16 layer norm bias length mismatch");

        let mean = self.bf16_mean(values);
        let centered = values.iter().map(|&value| self.bf16_sub(value, mean)).collect::<Vec<_>>();
        let squared = centered.iter().map(|&value| self.bf16_square(value)).collect::<Vec<_>>();
        let variance = self.bf16_mean(&squared);
        let variance_with_epsilon = self.bf16_add(variance, epsilon);
        let inverse_standard_deviation = self.bf16_rsqrt(variance_with_epsilon);

        centered
            .into_iter()
            .zip(weight.iter().copied())
            .zip(bias.iter().copied())
            .map(|((value, weight), bias)| {
                let normalized = self.bf16_mul(value, inverse_standard_deviation);
                let scaled = self.bf16_mul(normalized, weight);
                self.bf16_add(scaled, bias)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1ExtensionField};
    use sp1_recursion_executor::{
        Bf16AddSubOpcode, Bf16AddSubWitness, Bf16DivWitness, Bf16MulWitness, Bf16UnaryOpcode,
        Bf16UnaryWitness, Executor,
    };

    use crate::circuit::{AsmBuilder, AsmCompiler};

    use super::*;

    const GPT2_ONCE_HIDDEN_STATE: &str =
        include_str!("../../../data/gpt2/once/hidden_state.bf16.hex");

    fn gpt2_once_hidden_state() -> Vec<u16> {
        GPT2_ONCE_HIDDEN_STATE
            .lines()
            .map(|value| u16::from_str_radix(value, 16).unwrap())
            .collect()
    }

    fn reference_bf16_add(lhs: u16, rhs: u16) -> u16 {
        Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Add).output.raw
    }

    fn reference_bf16_sub(lhs: u16, rhs: u16) -> u16 {
        Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Sub).output.raw
    }

    fn reference_bf16_mean(values: &[u16]) -> u16 {
        let sum = values[1..].iter().fold(values[0], |sum, &value| reference_bf16_add(sum, value));
        Bf16DivWitness::new(sum, usize_to_bf16_raw(values.len())).output.raw
    }

    fn reference_bf16_layer_norm(
        values: &[u16],
        weight: &[u16],
        bias: &[u16],
        epsilon: u16,
    ) -> Vec<u16> {
        let mean = reference_bf16_mean(values);
        let centered =
            values.iter().map(|&value| reference_bf16_sub(value, mean)).collect::<Vec<_>>();
        let squared = centered
            .iter()
            .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, value).output)
            .collect::<Vec<_>>();
        let variance = reference_bf16_mean(&squared);
        let variance_with_epsilon = reference_bf16_add(variance, epsilon);
        let inverse_standard_deviation =
            Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, variance_with_epsilon).output;

        centered
            .into_iter()
            .zip(weight.iter().copied())
            .zip(bias.iter().copied())
            .map(|((value, weight), bias)| {
                let normalized = Bf16MulWitness::new(value, inverse_standard_deviation).output.raw;
                let scaled = Bf16MulWitness::new(normalized, weight).output.raw;
                reference_bf16_add(scaled, bias)
            })
            .collect()
    }

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
    fn compiles_and_executes_bf16_square() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x3fc0));
        let output = builder.bf16_square(input);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0x4010));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_unary_events.len(), 1);
        assert!(executor.record.bf16_mul_events.is_empty());
    }

    #[test]
    fn compiles_and_executes_bf16_rsqrt() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0x4000));
        let output = builder.bf16_rsqrt(input);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0x3f35));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_unary_events.len(), 1);
        assert_eq!(executor.record.bf16_unary_events[0].opcode, Bf16UnaryOpcode::Rsqrt);
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

    #[test]
    fn compiles_and_executes_bf16_mean() {
        let raw_values = gpt2_once_hidden_state();
        assert_eq!(raw_values.len(), 768);
        assert_eq!(usize_to_bf16_raw(raw_values.len()), 0x4440);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<Felt<_>>>();
        let mean = builder.bf16_mean(&values);
        builder.assert_felt_eq(mean, SP1Field::from_canonical_u16(0xbb3d));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_values.len() - 1);
        assert_eq!(executor.record.bf16_div_events.len(), 1);
    }

    #[test]
    fn compiles_and_executes_bf16_variance() {
        let raw_values = gpt2_once_hidden_state();
        assert_eq!(raw_values.len(), 768);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<Felt<_>>>();
        let mean = builder.bf16_mean(&values);
        let variance = builder.bf16_variance(&values, mean);
        builder.assert_felt_eq(mean, SP1Field::from_canonical_u16(0xbb3d));
        builder.assert_felt_eq(variance, SP1Field::from_canonical_u16(0x3dee));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), 2302);
        assert_eq!(executor.record.bf16_unary_events.len(), 768);
        assert!(executor.record.bf16_mul_events.is_empty());
        assert_eq!(executor.record.bf16_div_events.len(), 2);
    }

    #[test]
    fn compiles_and_executes_gpt2_bf16_layer_norm() {
        const BF16_ONE: u16 = 0x3f80;
        const BF16_ZERO: u16 = 0x0000;
        const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;

        let raw_values = gpt2_once_hidden_state();
        let raw_weight = vec![BF16_ONE; raw_values.len()];
        let raw_bias = vec![BF16_ZERO; raw_values.len()];
        let expected =
            reference_bf16_layer_norm(&raw_values, &raw_weight, &raw_bias, GPT2_LAYER_NORM_EPSILON);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<Felt<_>>>();
        let weight = raw_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<Felt<_>>>();
        let bias = raw_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<Felt<_>>>();
        let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
        let output = builder.bf16_layer_norm(&values, &weight, &bias, epsilon);
        assert_eq!(output.len(), raw_values.len());
        for index in [0, output.len() / 2, output.len() - 1] {
            builder.assert_felt_eq(output[index], SP1Field::from_canonical_u16(expected[index]));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), 3071);
        assert_eq!(executor.record.bf16_unary_events.len(), 769);
        assert_eq!(executor.record.bf16_mul_events.len(), 1536);
        assert_eq!(executor.record.bf16_div_events.len(), 2);
    }

    #[test]
    #[should_panic(expected = "BF16 mean requires at least one value")]
    fn rejects_empty_bf16_mean() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        builder.bf16_mean(&[]);
    }
}
