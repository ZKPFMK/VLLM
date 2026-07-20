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

    /// Compute the mathematical exponential of one raw BF16 encoding with a single lookup.
    pub fn bf16_exponential(&mut self, input: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16Exponential(output, input));
        output
    }

    /// Apply GPT-2's tanh-approximated GELU with a single raw-to-raw BF16 lookup.
    pub fn bf16_gelu_new(&mut self, input: Felt<SP1Field>) -> Felt<SP1Field> {
        let output = self.uninit();
        self.push_op(DslIr::Bf16GeluNew(output, input));
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

    /// Compute a non-empty BF16 dot product, rounding every multiplication and addition toward
    /// zero before the next operation consumes it.
    ///
    /// Products are accumulated strictly from left to right.
    pub fn bf16_dot(&mut self, lhs: &[Felt<SP1Field>], rhs: &[Felt<SP1Field>]) -> Felt<SP1Field> {
        assert!(!lhs.is_empty(), "BF16 dot product requires at least one value");
        assert_eq!(lhs.len(), rhs.len(), "BF16 dot product length mismatch");

        let mut sum = self.bf16_mul(lhs[0], rhs[0]);
        for (&lhs, &rhs) in lhs[1..].iter().zip(&rhs[1..]) {
            let product = self.bf16_mul(lhs, rhs);
            sum = self.bf16_add(sum, product);
        }
        sum
    }

    /// Apply a BF16 linear transformation with a bias.
    ///
    /// `weight` is a row-major `[input_features, output_features]` matrix, matching the layout of
    /// GPT-2's `Conv1D` weights. Each output dot product is accumulated from left to right, and
    /// every multiplication and addition rounds toward zero before its result is reused.
    pub fn bf16_linear(
        &mut self,
        input: &[Felt<SP1Field>],
        weight: &[Felt<SP1Field>],
        bias: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        assert!(!input.is_empty(), "BF16 linear requires at least one input feature");
        assert!(!bias.is_empty(), "BF16 linear requires at least one output feature");

        let output_features = bias.len();
        let expected_weight_len = input
            .len()
            .checked_mul(output_features)
            .expect("BF16 linear weight dimensions overflow");
        assert_eq!(weight.len(), expected_weight_len, "BF16 linear weight shape mismatch");

        let mut output = Vec::with_capacity(output_features);
        for output_index in 0..output_features {
            let column = weight
                .chunks_exact(output_features)
                .map(|row| row[output_index])
                .collect::<Vec<_>>();
            let dot = self.bf16_dot(input, &column);
            output.push(self.bf16_add(dot, bias[output_index]));
        }
        output
    }

    /// Split one GPT-2 QKV projection into query, key, and value heads.
    ///
    /// The input layout is `[Q, K, V]`, with each section laid out as
    /// `[num_heads, head_dimension]`. This only rearranges existing circuit variables and therefore
    /// emits no instructions or lookup events.
    pub fn bf16_split_qkv_heads(
        &self,
        qkv: &[Felt<SP1Field>],
        num_heads: usize,
    ) -> (Vec<Vec<Felt<SP1Field>>>, Vec<Vec<Felt<SP1Field>>>, Vec<Vec<Felt<SP1Field>>>) {
        assert!(!qkv.is_empty(), "BF16 QKV split requires at least one value");
        assert!(num_heads > 0, "BF16 QKV split requires at least one head");
        assert_eq!(qkv.len() % 3, 0, "BF16 QKV length must be divisible by three");

        let hidden_size = qkv.len() / 3;
        assert_eq!(
            hidden_size % num_heads,
            0,
            "BF16 QKV hidden size must be divisible by head count"
        );
        let head_dimension = hidden_size / num_heads;
        let (query, key_value) = qkv.split_at(hidden_size);
        let (key, value) = key_value.split_at(hidden_size);
        let split_heads = |values: &[Felt<SP1Field>]| {
            values.chunks_exact(head_dimension).map(<[Felt<SP1Field>]>::to_vec).collect::<Vec<_>>()
        };

        (split_heads(query), split_heads(key), split_heads(value))
    }

    /// Compute one scaled BF16 attention score: `(query dot key) * scale`.
    ///
    /// For GPT-2 with a head dimension of 64, `scale` is the raw BF16 value `0x3e00` (`1/8`).
    pub fn bf16_scaled_attention_score(
        &mut self,
        query: &[Felt<SP1Field>],
        key: &[Felt<SP1Field>],
        scale: Felt<SP1Field>,
    ) -> Felt<SP1Field> {
        let dot = self.bf16_dot(query, key);
        self.bf16_mul(dot, scale)
    }

    /// Compute the scaled attention scores for one query head against a flattened key cache.
    ///
    /// `keys` has row-major layout `[key_count, head_dimension]`. Autoregressive inference can pass
    /// only past and current keys, making every returned score causally visible without a mask.
    pub fn bf16_attention_scores(
        &mut self,
        query: &[Felt<SP1Field>],
        keys: &[Felt<SP1Field>],
        scale: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!query.is_empty(), "BF16 attention scores require a non-empty query");
        assert!(!keys.is_empty(), "BF16 attention scores require at least one key");
        assert_eq!(keys.len() % query.len(), 0, "BF16 attention key cache shape mismatch");

        keys.chunks_exact(query.len())
            .map(|key| self.bf16_scaled_attention_score(query, key, scale))
            .collect()
    }

    /// Compute a basic BF16 softmax, rounding every intermediate operation toward zero.
    ///
    /// Exponential values and their left-to-right sum remain in BF16. This version deliberately
    /// does not subtract the maximum score; a numerically stable softmax will additionally require
    /// a constrained BF16 comparison/maximum operation.
    pub fn bf16_softmax(&mut self, values: &[Felt<SP1Field>]) -> Vec<Felt<SP1Field>> {
        assert!(!values.is_empty(), "BF16 softmax requires at least one value");

        let exponential_values =
            values.iter().map(|&value| self.bf16_exponential(value)).collect::<Vec<_>>();
        let mut sum = exponential_values[0];
        for &value in &exponential_values[1..] {
            sum = self.bf16_add(sum, value);
        }

        exponential_values.into_iter().map(|value| self.bf16_div(value, sum)).collect()
    }

    /// Compute one attention head's BF16 weighted sum over a flattened value cache.
    ///
    /// `values` has row-major layout `[value_count, head_dimension]`. For each output feature, the
    /// products are accumulated in value-cache order by [`Self::bf16_dot`].
    pub fn bf16_attention_weighted_sum(
        &mut self,
        probabilities: &[Felt<SP1Field>],
        values: &[Felt<SP1Field>],
        head_dimension: usize,
    ) -> Vec<Felt<SP1Field>> {
        assert!(
            !probabilities.is_empty(),
            "BF16 attention weighted sum requires at least one probability"
        );
        assert!(
            head_dimension > 0,
            "BF16 attention weighted sum requires a nonzero head dimension"
        );
        let expected_value_len = probabilities
            .len()
            .checked_mul(head_dimension)
            .expect("BF16 attention value cache dimensions overflow");
        assert_eq!(values.len(), expected_value_len, "BF16 attention value cache shape mismatch");

        let mut output = Vec::with_capacity(head_dimension);
        for feature in 0..head_dimension {
            let value_column =
                values.chunks_exact(head_dimension).map(|value| value[feature]).collect::<Vec<_>>();
            output.push(self.bf16_dot(probabilities, &value_column));
        }
        output
    }

    /// Evaluate one BF16 autoregressive attention head through score, softmax, and value mixing.
    ///
    /// `keys` and `values` both use row-major `[cache_count, query.len()]` layout. Passing only past
    /// and current cache entries enforces causality without inserting an explicit mask.
    pub fn bf16_attention_head(
        &mut self,
        query: &[Felt<SP1Field>],
        keys: &[Felt<SP1Field>],
        values: &[Felt<SP1Field>],
        scale: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!query.is_empty(), "BF16 attention head requires a non-empty query");
        assert!(!keys.is_empty(), "BF16 attention head requires at least one key");
        assert_eq!(keys.len() % query.len(), 0, "BF16 attention key cache shape mismatch");
        assert_eq!(values.len(), keys.len(), "BF16 attention value cache shape mismatch");

        let scores = self.bf16_attention_scores(query, keys, scale);
        let probabilities = self.bf16_softmax(&scores);
        self.bf16_attention_weighted_sum(&probabilities, values, query.len())
    }

    /// Concatenate equally sized attention heads in head-major order.
    ///
    /// This only rearranges existing circuit variables and emits no instructions or lookup events.
    pub fn bf16_merge_attention_heads(&self, heads: &[Vec<Felt<SP1Field>>]) -> Vec<Felt<SP1Field>> {
        assert!(!heads.is_empty(), "BF16 attention merge requires at least one head");
        let head_dimension = heads[0].len();
        assert!(head_dimension > 0, "BF16 attention merge requires a nonzero head dimension");
        assert!(
            heads.iter().all(|head| head.len() == head_dimension),
            "BF16 attention head dimension mismatch"
        );

        heads.iter().flatten().copied().collect()
    }

    /// Add two non-empty BF16 vectors element by element.
    pub fn bf16_vector_add(
        &mut self,
        lhs: &[Felt<SP1Field>],
        rhs: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        assert!(!lhs.is_empty(), "BF16 vector addition requires at least one value");
        assert_eq!(lhs.len(), rhs.len(), "BF16 vector addition length mismatch");
        lhs.iter().zip(rhs).map(|(&lhs, &rhs)| self.bf16_add(lhs, rhs)).collect()
    }

    /// Merge attention heads, apply GPT-2's `c_proj`, and add the residual connection.
    ///
    /// The projection weight uses GPT-2's row-major `[merged_size, output_size]` `Conv1D` layout.
    /// Dropout is omitted because it is disabled during inference.
    pub fn bf16_attention_output(
        &mut self,
        heads: &[Vec<Felt<SP1Field>>],
        projection_weight: &[Felt<SP1Field>],
        projection_bias: &[Felt<SP1Field>],
        residual: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        let merged = self.bf16_merge_attention_heads(heads);
        assert_eq!(
            residual.len(),
            projection_bias.len(),
            "BF16 attention residual length mismatch"
        );
        let projected = self.bf16_linear(&merged, projection_weight, projection_bias);
        self.bf16_vector_add(residual, &projected)
    }

    /// Apply GPT-2's `gelu_new` activation element by element.
    pub fn bf16_gelu_new_vector(&mut self, values: &[Felt<SP1Field>]) -> Vec<Felt<SP1Field>> {
        assert!(!values.is_empty(), "BF16 GELU requires at least one value");
        values.iter().map(|&value| self.bf16_gelu_new(value)).collect()
    }

    /// Evaluate GPT-2's pre-normalized MLP sub-block and residual connection.
    ///
    /// This applies `ln_2`, `c_fc`, `gelu_new`, `c_proj`, then adds the original `residual`.
    /// Expansion weights use `[hidden_size, inner_size]`; projection weights use
    /// `[inner_size, hidden_size]`, matching GPT-2's `Conv1D` layout. Dropout is disabled during
    /// inference and is therefore omitted.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gpt2_mlp(
        &mut self,
        residual: &[Felt<SP1Field>],
        layer_norm_weight: &[Felt<SP1Field>],
        layer_norm_bias: &[Felt<SP1Field>],
        epsilon: Felt<SP1Field>,
        expansion_weight: &[Felt<SP1Field>],
        expansion_bias: &[Felt<SP1Field>],
        projection_weight: &[Felt<SP1Field>],
        projection_bias: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        assert!(!residual.is_empty(), "BF16 GPT-2 MLP requires a non-empty residual");
        let hidden_size = residual.len();
        assert_eq!(
            layer_norm_weight.len(),
            hidden_size,
            "BF16 GPT-2 MLP layer norm weight length mismatch"
        );
        assert_eq!(
            layer_norm_bias.len(),
            hidden_size,
            "BF16 GPT-2 MLP layer norm bias length mismatch"
        );

        let inner_size = expansion_bias.len();
        assert!(inner_size > 0, "BF16 GPT-2 MLP requires a non-empty expansion bias");
        let expected_expansion_weight_len = hidden_size
            .checked_mul(inner_size)
            .expect("BF16 GPT-2 MLP expansion dimensions overflow");
        assert_eq!(
            expansion_weight.len(),
            expected_expansion_weight_len,
            "BF16 GPT-2 MLP expansion weight shape mismatch"
        );
        assert_eq!(
            projection_bias.len(),
            hidden_size,
            "BF16 GPT-2 MLP projection bias length mismatch"
        );
        let expected_projection_weight_len = inner_size
            .checked_mul(hidden_size)
            .expect("BF16 GPT-2 MLP projection dimensions overflow");
        assert_eq!(
            projection_weight.len(),
            expected_projection_weight_len,
            "BF16 GPT-2 MLP projection weight shape mismatch"
        );

        let normalized =
            self.bf16_layer_norm(residual, layer_norm_weight, layer_norm_bias, epsilon);
        let expanded = self.bf16_linear(&normalized, expansion_weight, expansion_bias);
        let activated = self.bf16_gelu_new_vector(&expanded);
        let projected = self.bf16_linear(&activated, projection_weight, projection_bias);
        self.bf16_vector_add(residual, &projected)
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

    fn reference_bf16_dot(lhs: &[u16], rhs: &[u16]) -> u16 {
        let mut sum = Bf16MulWitness::new(lhs[0], rhs[0]).output.raw;
        for (&lhs, &rhs) in lhs[1..].iter().zip(&rhs[1..]) {
            let product = Bf16MulWitness::new(lhs, rhs).output.raw;
            sum = reference_bf16_add(sum, product);
        }
        sum
    }

    fn reference_bf16_linear(input: &[u16], weight: &[u16], bias: &[u16]) -> Vec<u16> {
        let output_features = bias.len();
        (0..output_features)
            .map(|output_index| {
                let column = weight
                    .chunks_exact(output_features)
                    .map(|row| row[output_index])
                    .collect::<Vec<_>>();
                let dot = reference_bf16_dot(input, &column);
                reference_bf16_add(dot, bias[output_index])
            })
            .collect()
    }

    fn reference_bf16_scaled_attention_score(query: &[u16], key: &[u16], scale: u16) -> u16 {
        let dot = reference_bf16_dot(query, key);
        Bf16MulWitness::new(dot, scale).output.raw
    }

    fn reference_bf16_softmax(values: &[u16]) -> Vec<u16> {
        let exponential_values = values
            .iter()
            .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Exponential, value).output)
            .collect::<Vec<_>>();
        let sum = exponential_values[1..]
            .iter()
            .fold(exponential_values[0], |sum, &value| reference_bf16_add(sum, value));
        exponential_values
            .into_iter()
            .map(|value| Bf16DivWitness::new(value, sum).output.raw)
            .collect()
    }

    fn reference_bf16_attention_weighted_sum(
        probabilities: &[u16],
        values: &[u16],
        head_dimension: usize,
    ) -> Vec<u16> {
        (0..head_dimension)
            .map(|feature| {
                let value_column = values
                    .chunks_exact(head_dimension)
                    .map(|value| value[feature])
                    .collect::<Vec<_>>();
                reference_bf16_dot(probabilities, &value_column)
            })
            .collect()
    }

    fn reference_bf16_attention_head(
        query: &[u16],
        keys: &[u16],
        values: &[u16],
        scale: u16,
    ) -> Vec<u16> {
        let scores = keys
            .chunks_exact(query.len())
            .map(|key| reference_bf16_scaled_attention_score(query, key, scale))
            .collect::<Vec<_>>();
        let probabilities = reference_bf16_softmax(&scores);
        reference_bf16_attention_weighted_sum(&probabilities, values, query.len())
    }

    fn reference_bf16_attention_output(
        heads: &[Vec<u16>],
        projection_weight: &[u16],
        projection_bias: &[u16],
        residual: &[u16],
    ) -> Vec<u16> {
        let merged = heads.iter().flatten().copied().collect::<Vec<_>>();
        let projected = reference_bf16_linear(&merged, projection_weight, projection_bias);
        residual
            .iter()
            .zip(projected)
            .map(|(&residual, projected)| reference_bf16_add(residual, projected))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_bf16_gpt2_mlp(
        residual: &[u16],
        layer_norm_weight: &[u16],
        layer_norm_bias: &[u16],
        epsilon: u16,
        expansion_weight: &[u16],
        expansion_bias: &[u16],
        projection_weight: &[u16],
        projection_bias: &[u16],
    ) -> Vec<u16> {
        let normalized =
            reference_bf16_layer_norm(residual, layer_norm_weight, layer_norm_bias, epsilon);
        let expanded = reference_bf16_linear(&normalized, expansion_weight, expansion_bias);
        let activated = expanded
            .into_iter()
            .map(|value| Bf16UnaryWitness::new(Bf16UnaryOpcode::GeluNew, value).output)
            .collect::<Vec<_>>();
        let projected = reference_bf16_linear(&activated, projection_weight, projection_bias);
        residual
            .iter()
            .zip(projected)
            .map(|(&residual, projected)| reference_bf16_add(residual, projected))
            .collect()
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
    fn compiles_and_executes_bf16_exponential() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0xbf80));
        let output = builder.bf16_exponential(input);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0x3ebc));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_unary_events.len(), 1);
        assert_eq!(executor.record.bf16_unary_events[0].opcode, Bf16UnaryOpcode::Exponential);
    }

    #[test]
    fn compiles_and_executes_bf16_gelu_new() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input: Felt<_> = builder.constant(SP1Field::from_canonical_u16(0xbf80));
        let output = builder.bf16_gelu_new(input);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(0xbe22));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_unary_events.len(), 1);
        assert_eq!(executor.record.bf16_unary_events[0].opcode, Bf16UnaryOpcode::GeluNew);
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
    fn compiles_and_executes_bf16_dot() {
        let raw_lhs = [0x3f80, 0x4000, 0xbf00];
        let raw_rhs = [0x4000, 0xbf80, 0x4080];
        let expected = reference_bf16_dot(&raw_lhs, &raw_rhs);
        assert_eq!(expected, 0xc000);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let lhs = raw_lhs
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let rhs = raw_rhs
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let output = builder.bf16_dot(&lhs, &rhs);
        builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), raw_lhs.len());
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_lhs.len() - 1);
    }

    #[test]
    fn compiles_and_executes_bf16_linear() {
        let raw_input = [0x3f80, 0x4000];
        let raw_weight = [
            0x3f80, 0x4000, 0xbf80, // input feature 0
            0x4040, 0xbf80, 0x3f00, // input feature 1
        ];
        let raw_bias = [0x3f00, 0x3f80, 0xc000];
        let expected = reference_bf16_linear(&raw_input, &raw_weight, &raw_bias);
        assert_eq!(expected, [0x40f0, 0x3f80, 0xc000]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input = raw_input
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let weight = raw_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let bias = raw_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let output = builder.bf16_linear(&input, &weight, &bias);
        assert_eq!(output.len(), expected.len());
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), raw_input.len() * raw_bias.len());
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_input.len() * raw_bias.len());
    }

    #[test]
    fn compiles_and_executes_bf16_attention_scores() {
        let raw_qkv = [
            0x3f80, 0x4000, 0x4040, 0x4080, // query heads
            0x3f00, 0xbf80, 0x4000, 0x3e80, // key heads
            0x40a0, 0x40c0, 0x40e0, 0x4100, // value heads
        ];
        let raw_scale = 0x3f00;
        let expected = [
            reference_bf16_scaled_attention_score(&raw_qkv[0..2], &raw_qkv[4..6], raw_scale),
            reference_bf16_scaled_attention_score(&raw_qkv[0..2], &raw_qkv[6..8], raw_scale),
        ];
        assert_eq!(expected, [0xbf40, 0x3fa0]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let qkv = raw_qkv
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let scale = builder.constant(SP1Field::from_canonical_u16(raw_scale));
        let (query, key, value) = builder.bf16_split_qkv_heads(&qkv, 2);
        assert_eq!(query.len(), 2);
        assert_eq!(key.len(), 2);
        assert_eq!(value.len(), 2);
        assert!(query.iter().chain(&key).chain(&value).all(|head| head.len() == 2));

        let key_cache = key.into_iter().flatten().collect::<Vec<_>>();
        let scores = builder.bf16_attention_scores(&query[0], &key_cache, scale);
        assert_eq!(scores.len(), expected.len());
        for (&score, &expected) in scores.iter().zip(&expected) {
            builder.assert_felt_eq(score, SP1Field::from_canonical_u16(expected));
        }
        builder.assert_felt_eq(value[1][1], SP1Field::from_canonical_u16(raw_qkv[11]));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 6);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 2);
    }

    #[test]
    fn compiles_and_executes_bf16_softmax() {
        let raw_values = [0xbf80, 0x0000];
        let expected = reference_bf16_softmax(&raw_values);
        assert_eq!(expected, [0x3e89, 0x3f3b]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let output = builder.bf16_softmax(&values);
        assert_eq!(output.len(), expected.len());
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_unary_events.len(), raw_values.len());
        assert!(executor
            .record
            .bf16_unary_events
            .iter()
            .all(|event| event.opcode == Bf16UnaryOpcode::Exponential));
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_values.len() - 1);
        assert_eq!(executor.record.bf16_div_events.len(), raw_values.len());
    }

    #[test]
    fn compiles_and_executes_bf16_attention_head() {
        let raw_query = [0x3f80, 0x0000];
        let raw_keys = [
            0xbf80, 0x0000, // cached key 0
            0x0000, 0x3f80, // cached key 1
        ];
        let raw_values = [
            0x3f80, 0x4000, // cached value 0
            0x4040, 0x4080, // cached value 1
        ];
        let raw_scale = 0x3f80;
        let expected = reference_bf16_attention_head(&raw_query, &raw_keys, &raw_values, raw_scale);
        assert_eq!(expected, [0x401d, 0x405d]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let query = raw_query
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let keys = raw_keys
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let scale = builder.constant(SP1Field::from_canonical_u16(raw_scale));
        let output = builder.bf16_attention_head(&query, &keys, &values, scale);
        assert_eq!(output.len(), raw_query.len());
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 10);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 5);
        assert_eq!(executor.record.bf16_unary_events.len(), 2);
        assert!(executor
            .record
            .bf16_unary_events
            .iter()
            .all(|event| event.opcode == Bf16UnaryOpcode::Exponential));
        assert_eq!(executor.record.bf16_div_events.len(), 2);
    }

    #[test]
    fn compiles_and_executes_bf16_attention_output() {
        let raw_heads = [vec![0x3f80, 0x4000], vec![0x4040, 0x4080]];
        let raw_projection_weight = [
            0x3f80, 0x0000, // merged feature 0
            0x0000, 0x3f80, // merged feature 1
            0x3f80, 0x0000, // merged feature 2
            0x0000, 0x3f80, // merged feature 3
        ];
        let raw_projection_bias = [0x3f00, 0xbf80];
        let raw_residual = [0x4120, 0x41a0];
        let expected = reference_bf16_attention_output(
            &raw_heads,
            &raw_projection_weight,
            &raw_projection_bias,
            &raw_residual,
        );
        assert_eq!(expected, [0x4168, 0x41c8]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let heads = raw_heads
            .iter()
            .map(|head| {
                head.iter()
                    .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let projection_weight = raw_projection_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let projection_bias = raw_projection_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let residual = raw_residual
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let output =
            builder.bf16_attention_output(&heads, &projection_weight, &projection_bias, &residual);
        assert_eq!(output.len(), raw_residual.len());
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 8);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 10);
        assert!(executor.record.bf16_unary_events.is_empty());
        assert!(executor.record.bf16_div_events.is_empty());
    }

    #[test]
    fn compiles_and_executes_bf16_gpt2_mlp() {
        let raw_residual = [0x3f80, 0xbf80];
        let raw_layer_norm_weight = [0x3f80, 0x3f80];
        let raw_layer_norm_bias = [0x0000, 0x0000];
        let raw_epsilon = 0x0000;
        let raw_expansion_weight = [
            0x3f80, 0x0000, // hidden feature 0
            0x0000, 0x3f80, // hidden feature 1
        ];
        let raw_expansion_bias = [0x0000, 0x0000];
        let raw_projection_weight = [
            0x3f80, 0x0000, // inner feature 0
            0x0000, 0x3f80, // inner feature 1
        ];
        let raw_projection_bias = [0x0000, 0x0000];
        let expected = reference_bf16_gpt2_mlp(
            &raw_residual,
            &raw_layer_norm_weight,
            &raw_layer_norm_bias,
            raw_epsilon,
            &raw_expansion_weight,
            &raw_expansion_bias,
            &raw_projection_weight,
            &raw_projection_bias,
        );
        assert_eq!(expected, [0x3feb, 0xbf94]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let residual = raw_residual
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let layer_norm_weight = raw_layer_norm_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let layer_norm_bias = raw_layer_norm_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let epsilon = builder.constant(SP1Field::from_canonical_u16(raw_epsilon));
        let expansion_weight = raw_expansion_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let expansion_bias = raw_expansion_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let projection_weight = raw_projection_weight
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let projection_bias = raw_projection_bias
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let output = builder.bf16_gpt2_mlp(
            &residual,
            &layer_norm_weight,
            &layer_norm_bias,
            epsilon,
            &expansion_weight,
            &expansion_bias,
            &projection_weight,
            &projection_bias,
        );
        assert_eq!(output.len(), raw_residual.len());
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 12);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 17);
        assert_eq!(executor.record.bf16_unary_events.len(), 5);
        assert_eq!(
            executor
                .record
                .bf16_unary_events
                .iter()
                .filter(|event| event.opcode == Bf16UnaryOpcode::GeluNew)
                .count(),
            raw_expansion_bias.len()
        );
        assert_eq!(executor.record.bf16_div_events.len(), 2);
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

    #[test]
    #[should_panic(expected = "BF16 dot product requires at least one value")]
    fn rejects_empty_bf16_dot() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        builder.bf16_dot(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "BF16 dot product length mismatch")]
    fn rejects_mismatched_bf16_dot() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_dot(&[value], &[]);
    }

    #[test]
    #[should_panic(expected = "BF16 linear requires at least one input feature")]
    fn rejects_empty_bf16_linear_input() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let bias = builder.constant(SP1Field::zero());
        builder.bf16_linear(&[], &[], &[bias]);
    }

    #[test]
    #[should_panic(expected = "BF16 linear requires at least one output feature")]
    fn rejects_empty_bf16_linear_output() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input = builder.constant(SP1Field::zero());
        builder.bf16_linear(&[input], &[], &[]);
    }

    #[test]
    #[should_panic(expected = "BF16 linear weight shape mismatch")]
    fn rejects_mismatched_bf16_linear_weight() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input = builder.constant(SP1Field::zero());
        let bias = builder.constant(SP1Field::zero());
        builder.bf16_linear(&[input], &[], &[bias]);
    }

    #[test]
    #[should_panic(expected = "BF16 QKV hidden size must be divisible by head count")]
    fn rejects_mismatched_bf16_qkv_heads() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_split_qkv_heads(&[value; 6], 4);
    }

    #[test]
    #[should_panic(expected = "BF16 attention key cache shape mismatch")]
    fn rejects_mismatched_bf16_attention_key_cache() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_attention_scores(&[value; 2], &[value; 3], value);
    }

    #[test]
    #[should_panic(expected = "BF16 softmax requires at least one value")]
    fn rejects_empty_bf16_softmax() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        builder.bf16_softmax(&[]);
    }

    #[test]
    #[should_panic(expected = "BF16 attention value cache shape mismatch")]
    fn rejects_mismatched_bf16_attention_value_cache() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_attention_head(&[value; 2], &[value; 4], &[value; 3], value);
    }

    #[test]
    #[should_panic(expected = "BF16 attention head dimension mismatch")]
    fn rejects_mismatched_bf16_attention_head_dimensions() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_merge_attention_heads(&[vec![value], vec![value; 2]]);
    }

    #[test]
    #[should_panic(expected = "BF16 attention residual length mismatch")]
    fn rejects_mismatched_bf16_attention_residual() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_attention_output(&[vec![value]], &[value], &[value], &[]);
    }

    #[test]
    #[should_panic(expected = "BF16 GPT-2 MLP projection weight shape mismatch")]
    fn rejects_mismatched_bf16_gpt2_mlp_projection() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_gpt2_mlp(
            &[value],
            &[value],
            &[value],
            value,
            &[value],
            &[value],
            &[],
            &[value],
        );
    }
}
