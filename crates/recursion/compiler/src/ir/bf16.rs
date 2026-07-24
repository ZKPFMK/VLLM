use std::sync::Arc;

use slop_algebra::AbstractField;
use sp1_primitives::SP1Field;
use sp1_recursion_executor::Bf16DivWitness;

use super::{
    Bf16DotProductOp, Bf16LinearNoBiasOp, Bf16MeanOp, Builder, Config, DslIr, Felt, IrIter,
};

/// Per-head key/value cache for one GPT-2 transformer layer.
#[derive(Clone, Debug, Default)]
pub struct Bf16KvCache {
    /// Each entry is a flattened `[cached_tokens, head_dimension]` key matrix for one head.
    pub keys: Vec<Vec<Felt<SP1Field>>>,
    /// Each entry is a flattened `[cached_tokens, head_dimension]` value matrix for one head.
    pub values: Vec<Vec<Felt<SP1Field>>>,
}

impl Bf16KvCache {
    /// Create an empty cache with one key and value vector per attention head.
    #[must_use]
    pub fn empty(num_heads: usize) -> Self {
        Self { keys: vec![Vec::new(); num_heads], values: vec![Vec::new(); num_heads] }
    }

    fn validate_shape(&self, num_heads: usize, head_dimension: usize) -> usize {
        assert!(num_heads > 0, "BF16 GPT-2 cache requires at least one attention head");
        assert!(head_dimension > 0, "BF16 GPT-2 cache requires a nonzero head dimension");
        assert_eq!(self.keys.len(), num_heads, "BF16 GPT-2 key cache head count mismatch");
        assert_eq!(self.values.len(), num_heads, "BF16 GPT-2 value cache head count mismatch");

        let cached_tokens = self.keys[0].len() / head_dimension;
        for head in 0..num_heads {
            assert_eq!(
                self.keys[head].len() % head_dimension,
                0,
                "BF16 GPT-2 key cache shape mismatch"
            );
            assert_eq!(
                self.values[head].len(),
                self.keys[head].len(),
                "BF16 GPT-2 value cache shape mismatch"
            );
            assert_eq!(
                self.keys[head].len() / head_dimension,
                cached_tokens,
                "BF16 GPT-2 cache token count mismatch"
            );
        }

        cached_tokens
    }
}

/// Borrowed BF16 parameters for one GPT-2 transformer layer.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Gpt2BlockParams<'a> {
    pub layer_norm_1_weight: &'a [Felt<SP1Field>],
    pub layer_norm_1_bias: &'a [Felt<SP1Field>],
    /// GPT-2 `attn.c_attn.weight`, laid out as `[hidden_size, 3 * hidden_size]`.
    pub attention_qkv_weight: &'a [Felt<SP1Field>],
    pub attention_qkv_bias: &'a [Felt<SP1Field>],
    /// GPT-2 `attn.c_proj.weight`, laid out as `[hidden_size, hidden_size]`.
    pub attention_projection_weight: &'a [Felt<SP1Field>],
    pub attention_projection_bias: &'a [Felt<SP1Field>],
    pub layer_norm_2_weight: &'a [Felt<SP1Field>],
    pub layer_norm_2_bias: &'a [Felt<SP1Field>],
    /// GPT-2 `mlp.c_fc.weight`, laid out as `[hidden_size, inner_size]`.
    pub mlp_expansion_weight: &'a [Felt<SP1Field>],
    pub mlp_expansion_bias: &'a [Felt<SP1Field>],
    /// GPT-2 `mlp.c_proj.weight`, laid out as `[inner_size, hidden_size]`.
    pub mlp_projection_weight: &'a [Felt<SP1Field>],
    pub mlp_projection_bias: &'a [Felt<SP1Field>],
    pub layer_norm_epsilon: Felt<SP1Field>,
    pub attention_scale: Felt<SP1Field>,
    pub num_heads: usize,
}

/// Output of one autoregressive GPT-2 transformer layer.
#[derive(Clone, Debug)]
pub struct Bf16Gpt2BlockOutput {
    pub hidden_state: Vec<Felt<SP1Field>>,
    pub cache: Bf16KvCache,
}

/// Key/value caches for every layer in a GPT-2 transformer stack.
#[derive(Clone, Debug, Default)]
pub struct Bf16Gpt2Cache {
    pub layers: Vec<Bf16KvCache>,
}

impl Bf16Gpt2Cache {
    /// Create an empty cache for a transformer whose layers all use the same head count.
    #[must_use]
    pub fn empty(num_layers: usize, num_heads: usize) -> Self {
        Self { layers: vec![Bf16KvCache::empty(num_heads); num_layers] }
    }
}

/// Externally supplied attention-score maxima for one autoregressive token.
///
/// `layers[layer][head]` contains one BF16 `max_hint`. The circuit uses it as a common softmax
/// shift for that head, but deliberately does not prove that it is the actual maximum score.
#[derive(Clone, Debug, Default)]
pub struct Bf16Gpt2AttentionMaxHints {
    pub layers: Vec<Vec<Felt<SP1Field>>>,
}

/// Borrowed BF16 parameters for the GPT-2 transformer blocks and final layer normalization.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Gpt2TransformerParams<'a> {
    pub blocks: &'a [Bf16Gpt2BlockParams<'a>],
    pub final_layer_norm_weight: &'a [Felt<SP1Field>],
    pub final_layer_norm_bias: &'a [Felt<SP1Field>],
    pub final_layer_norm_epsilon: Felt<SP1Field>,
}

/// Output of one token passing through every GPT-2 transformer block and `ln_f`.
#[derive(Clone, Debug)]
pub struct Bf16Gpt2TransformerOutput {
    pub hidden_state: Vec<Felt<SP1Field>>,
    pub cache: Bf16Gpt2Cache,
}

/// Borrowed BF16 parameters for GPT-2 inference starting after token/position embedding.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Gpt2ModelParams<'a> {
    pub transformer: Bf16Gpt2TransformerParams<'a>,
    /// GPT-2's tied `wte.weight`, laid out as `[vocab_size, hidden_size]`.
    pub lm_head_weight: &'a [Felt<SP1Field>],
}

/// GPT-2 output for one autoregressive token.
#[derive(Clone, Debug)]
pub struct Bf16Gpt2ModelOutput {
    /// Hidden state after all transformer blocks and `ln_f`.
    pub hidden_state: Vec<Felt<SP1Field>>,
    /// One BF16 logit per vocabulary entry.
    pub logits: Vec<Felt<SP1Field>>,
    pub cache: Bf16Gpt2Cache,
}

/// Borrowed BF16 parameters for one zkGPT-style transformer layer.
///
/// This deliberately follows the simplified computation implemented by the zkGPT prototype:
/// four bias-free matrices, no residual connections, an MLP width equal to the QKV width, and no
/// final layer normalization or language-model head. The input and output are flattened row-major
/// `[sequence_length, hidden_size]` matrices.
#[derive(Clone, Copy, Debug)]
pub struct Bf16ZkGptBlockParams<'a> {
    pub layer_norm_1_weight: &'a [Felt<SP1Field>],
    pub layer_norm_1_bias: &'a [Felt<SP1Field>],
    /// Bias-free `[hidden_size, 3 * hidden_size]` QKV projection.
    pub attention_qkv_weight: &'a [Felt<SP1Field>],
    /// Bias-free `[hidden_size, hidden_size]` attention projection.
    pub attention_projection_weight: &'a [Felt<SP1Field>],
    pub layer_norm_2_weight: &'a [Felt<SP1Field>],
    pub layer_norm_2_bias: &'a [Felt<SP1Field>],
    /// Bias-free `[hidden_size, linear_size]` MLP expansion.
    pub mlp_expansion_weight: &'a [Felt<SP1Field>],
    /// Bias-free `[linear_size, hidden_size]` MLP projection.
    pub mlp_projection_weight: &'a [Felt<SP1Field>],
    pub layer_norm_epsilon: Felt<SP1Field>,
    pub attention_scale: Felt<SP1Field>,
    pub num_heads: usize,
}

/// Externally supplied causal-attention maxima for a zkGPT-style transformer stack.
///
/// `layers[layer]` is row-major `[sequence_length, num_heads]`.
#[derive(Clone, Debug, Default)]
pub struct Bf16ZkGptAttentionMaxHints {
    pub layers: Vec<Vec<Felt<SP1Field>>>,
}

/// Borrowed parameters for a stack of zkGPT-style transformer layers.
#[derive(Clone, Copy, Debug)]
pub struct Bf16ZkGptStackParams<'a> {
    pub blocks: &'a [Bf16ZkGptBlockParams<'a>],
}

/// Encode a positive integer as BF16 using the same round-toward-zero policy as the arithmetic
/// chips. The common GPT-2 dimensions (including 768 and 3072) are represented exactly.
fn usize_to_bf16_raw(value: usize) -> u16 {
    assert!(value > 0, "BF16 integer conversion requires a positive value");

    let exponent = value.ilog2();
    let mantissa = if exponent <= 7 { value << (7 - exponent) } else { value >> (exponent - 7) };
    debug_assert!((128..256).contains(&mantissa));

    (((exponent + 127) as u16) << 7) | (mantissa as u16 - 128)
}

/// Compute the BF16-rounded reciprocal used when a divisor is known while building the circuit.
fn bf16_reciprocal_raw(divisor: u16) -> u16 {
    const BF16_ONE_RAW: u16 = 0x3f80;
    Bf16DivWitness::new(BF16_ONE_RAW, divisor).output.raw
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

        // Scalar intermediate addresses are reserved by the assembly batch instruction. Only the
        // externally visible result needs a DSL variable.
        let output = self.uninit();
        self.push_op(DslIr::Bf16DotProduct(Box::new(Bf16DotProductOp {
            lhs: lhs.into(),
            rhs: rhs.into(),
            output,
        })));
        output
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

    /// Apply a bias-free BF16 linear transformation.
    ///
    /// `weight` is row-major `[input_features, output_features]`. This matches the matrix boundary
    /// used by the zkGPT prototype, whose fully connected circuit explicitly omits bias terms.
    pub fn bf16_linear_no_bias(
        &mut self,
        input: &[Felt<SP1Field>],
        weight: &[Felt<SP1Field>],
        output_features: usize,
    ) -> Vec<Felt<SP1Field>> {
        self.bf16_linear_no_bias_shared(input, Arc::from(weight), output_features)
    }

    /// Build one compact bias-free BF16 linear operation.
    fn bf16_linear_no_bias_shared(
        &mut self,
        input: &[Felt<SP1Field>],
        weight: Arc<[Felt<SP1Field>]>,
        output_features: usize,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!input.is_empty(), "bias-free BF16 linear requires at least one input feature");
        assert!(output_features > 0, "bias-free BF16 linear requires an output feature");
        let expected_weight_len = input
            .len()
            .checked_mul(output_features)
            .expect("bias-free BF16 linear weight dimensions overflow");
        assert_eq!(
            weight.len(),
            expected_weight_len,
            "bias-free BF16 linear weight shape mismatch"
        );

        let outputs = (0..output_features).map(|_| self.uninit()).collect::<Vec<_>>();

        self.push_op(DslIr::Bf16LinearNoBias(Box::new(Bf16LinearNoBiasOp {
            input: input.into(),
            weight,
            input_features: input.len(),
            output_features,
            output: outputs.clone().into(),
        })));
        outputs
    }

    /// Apply one shared bias-free matrix to every row of a flattened BF16 matrix.
    pub fn bf16_linear_rows_no_bias(
        &mut self,
        input: &[Felt<SP1Field>],
        input_features: usize,
        weight: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        assert!(input_features > 0, "BF16 row-wise linear requires an input feature");
        assert!(!input.is_empty(), "BF16 row-wise linear requires at least one row");
        assert_eq!(input.len() % input_features, 0, "BF16 row-wise linear input shape mismatch");
        assert_eq!(weight.len() % input_features, 0, "BF16 row-wise linear weight shape mismatch");
        let output_features = weight.len() / input_features;
        assert!(output_features > 0, "BF16 row-wise linear requires an output feature");
        let rows = input.len() / input_features;
        let output = (0..rows * output_features).map(|_| self.uninit()).collect::<Vec<_>>();
        self.push_op(DslIr::Bf16LinearNoBias(Box::new(Bf16LinearNoBiasOp {
            input: input.into(),
            weight: Arc::from(weight),
            input_features,
            output_features,
            output: output.clone().into(),
        })));
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

    /// Compute BF16 softmax after subtracting one externally supplied maximum hint.
    ///
    /// The same `max_hint` is subtracted from every input before the exponential lookup. The hint
    /// is not compared with the inputs and is not constrained to equal their actual maximum.
    pub fn bf16_softmax(
        &mut self,
        values: &[Felt<SP1Field>],
        max_hint: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!values.is_empty(), "BF16 softmax requires at least one value");

        let exponential_values = values
            .iter()
            .map(|&value| {
                let shifted = self.bf16_sub(value, max_hint);
                self.bf16_exponential(shifted)
            })
            .collect::<Vec<_>>();
        let mut sum = exponential_values[0];
        for &value in &exponential_values[1..] {
            sum = self.bf16_add(sum, value);
        }

        // The normalization denominator is shared by every value in this softmax row. Compute its
        // BF16 reciprocal once, then reuse the multiplication chip for every probability.
        let one = self.constant(SP1Field::from_canonical_u16(0x3f80));
        let inverse_sum = self.bf16_div(one, sum);
        exponential_values.into_iter().map(|value| self.bf16_mul(value, inverse_sum)).collect()
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
        max_hint: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!query.is_empty(), "BF16 attention head requires a non-empty query");
        assert!(!keys.is_empty(), "BF16 attention head requires at least one key");
        assert_eq!(keys.len() % query.len(), 0, "BF16 attention key cache shape mismatch");
        assert_eq!(values.len(), keys.len(), "BF16 attention value cache shape mismatch");

        let scores = self.bf16_attention_scores(query, keys, scale);
        let probabilities = self.bf16_softmax(&scores, max_hint);
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

    /// Apply GPT-2's `gelu_new` activation independently to every row of a flattened matrix.
    pub fn bf16_gelu_new_rows(
        &mut self,
        values: &[Felt<SP1Field>],
        row_size: usize,
    ) -> Vec<Felt<SP1Field>> {
        assert!(row_size > 0, "BF16 row-wise GELU requires a nonzero row size");
        assert!(!values.is_empty(), "BF16 row-wise GELU requires at least one row");
        assert_eq!(values.len() % row_size, 0, "BF16 row-wise GELU input shape mismatch");

        let rows: Vec<Vec<_>> = values
            .chunks_exact(row_size)
            .ir_par_map_collect(self, |builder, row| builder.bf16_gelu_new_vector(row));
        rows.into_iter().flatten().collect()
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

    /// Evaluate one autoregressive GPT-2 transformer layer and update its KV cache.
    ///
    /// The input represents one token. The method performs pre-attention layer normalization, QKV
    /// projection, per-head causal attention over the supplied cache plus the current token,
    /// attention projection and residual addition, followed by [`Self::bf16_gpt2_mlp`].
    pub fn bf16_gpt2_block(
        &mut self,
        hidden_state: &[Felt<SP1Field>],
        cache: &Bf16KvCache,
        attention_max_hints: &[Felt<SP1Field>],
        params: &Bf16Gpt2BlockParams<'_>,
    ) -> Bf16Gpt2BlockOutput {
        assert!(!hidden_state.is_empty(), "BF16 GPT-2 block requires a non-empty hidden state");
        assert!(params.num_heads > 0, "BF16 GPT-2 block requires at least one attention head");
        assert_eq!(
            attention_max_hints.len(),
            params.num_heads,
            "BF16 GPT-2 attention max hint head count mismatch"
        );

        let hidden_size = hidden_state.len();
        assert_eq!(
            hidden_size % params.num_heads,
            0,
            "BF16 GPT-2 hidden size must be divisible by head count"
        );
        let head_dimension = hidden_size / params.num_heads;
        assert_eq!(
            params.layer_norm_1_weight.len(),
            hidden_size,
            "BF16 GPT-2 ln_1 weight length mismatch"
        );
        assert_eq!(
            params.layer_norm_1_bias.len(),
            hidden_size,
            "BF16 GPT-2 ln_1 bias length mismatch"
        );

        let qkv_size = hidden_size.checked_mul(3).expect("BF16 GPT-2 QKV dimensions overflow");
        assert_eq!(
            params.attention_qkv_bias.len(),
            qkv_size,
            "BF16 GPT-2 QKV bias length mismatch"
        );
        let expected_qkv_weight_len =
            hidden_size.checked_mul(qkv_size).expect("BF16 GPT-2 QKV weight dimensions overflow");
        assert_eq!(
            params.attention_qkv_weight.len(),
            expected_qkv_weight_len,
            "BF16 GPT-2 QKV weight shape mismatch"
        );
        assert_eq!(
            params.attention_projection_bias.len(),
            hidden_size,
            "BF16 GPT-2 attention projection bias length mismatch"
        );
        let expected_attention_projection_weight_len = hidden_size
            .checked_mul(hidden_size)
            .expect("BF16 GPT-2 attention projection dimensions overflow");
        assert_eq!(
            params.attention_projection_weight.len(),
            expected_attention_projection_weight_len,
            "BF16 GPT-2 attention projection weight shape mismatch"
        );

        assert_eq!(
            params.layer_norm_2_weight.len(),
            hidden_size,
            "BF16 GPT-2 ln_2 weight length mismatch"
        );
        assert_eq!(
            params.layer_norm_2_bias.len(),
            hidden_size,
            "BF16 GPT-2 ln_2 bias length mismatch"
        );
        let inner_size = params.mlp_expansion_bias.len();
        assert!(inner_size > 0, "BF16 GPT-2 block requires a non-empty MLP expansion bias");
        let expected_mlp_expansion_weight_len = hidden_size
            .checked_mul(inner_size)
            .expect("BF16 GPT-2 MLP expansion dimensions overflow");
        assert_eq!(
            params.mlp_expansion_weight.len(),
            expected_mlp_expansion_weight_len,
            "BF16 GPT-2 MLP expansion weight shape mismatch"
        );
        assert_eq!(
            params.mlp_projection_bias.len(),
            hidden_size,
            "BF16 GPT-2 MLP projection bias length mismatch"
        );
        let expected_mlp_projection_weight_len = inner_size
            .checked_mul(hidden_size)
            .expect("BF16 GPT-2 MLP projection dimensions overflow");
        assert_eq!(
            params.mlp_projection_weight.len(),
            expected_mlp_projection_weight_len,
            "BF16 GPT-2 MLP projection weight shape mismatch"
        );

        cache.validate_shape(params.num_heads, head_dimension);

        let normalized = self.bf16_layer_norm(
            hidden_state,
            params.layer_norm_1_weight,
            params.layer_norm_1_bias,
            params.layer_norm_epsilon,
        );
        let qkv =
            self.bf16_linear(&normalized, params.attention_qkv_weight, params.attention_qkv_bias);
        let (queries, current_keys, current_values) =
            self.bf16_split_qkv_heads(&qkv, params.num_heads);

        let mut updated_cache = cache.clone();
        let mut heads = Vec::with_capacity(params.num_heads);
        for head in 0..params.num_heads {
            updated_cache.keys[head].extend_from_slice(&current_keys[head]);
            updated_cache.values[head].extend_from_slice(&current_values[head]);
            heads.push(self.bf16_attention_head(
                &queries[head],
                &updated_cache.keys[head],
                &updated_cache.values[head],
                params.attention_scale,
                attention_max_hints[head],
            ));
        }

        let attention_output = self.bf16_attention_output(
            &heads,
            params.attention_projection_weight,
            params.attention_projection_bias,
            hidden_state,
        );
        let hidden_state = self.bf16_gpt2_mlp(
            &attention_output,
            params.layer_norm_2_weight,
            params.layer_norm_2_bias,
            params.layer_norm_epsilon,
            params.mlp_expansion_weight,
            params.mlp_expansion_bias,
            params.mlp_projection_weight,
            params.mlp_projection_bias,
        );

        Bf16Gpt2BlockOutput { hidden_state, cache: updated_cache }
    }

    /// Apply one shared BF16 layer normalization to every row of a flattened matrix.
    pub fn bf16_layer_norm_rows(
        &mut self,
        values: &[Felt<SP1Field>],
        row_size: usize,
        weight: &[Felt<SP1Field>],
        bias: &[Felt<SP1Field>],
        epsilon: Felt<SP1Field>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(row_size > 0, "BF16 row-wise layer norm requires a nonzero row size");
        assert!(!values.is_empty(), "BF16 row-wise layer norm requires at least one row");
        assert_eq!(values.len() % row_size, 0, "BF16 row-wise layer norm input shape mismatch");
        assert_eq!(weight.len(), row_size, "BF16 row-wise layer norm weight shape mismatch");
        assert_eq!(bias.len(), row_size, "BF16 row-wise layer norm bias shape mismatch");

        let rows: Vec<Vec<_>> =
            values.chunks_exact(row_size).ir_par_map_collect(self, |builder, row| {
                builder.bf16_layer_norm(row, weight, bias, epsilon)
            });
        rows.into_iter().flatten().collect()
    }

    /// Evaluate one full-sequence transformer layer with the simplified zkGPT architecture.
    ///
    /// Unlike [`Self::bf16_gpt2_block`], this method consumes every token at once, constructs the
    /// complete causal attention triangle, omits all four linear biases and both residual additions,
    /// and uses the QKV width as the MLP expansion width. These are the computation boundaries in
    /// the zkGPT prototype; BF16 arithmetic remains in use so the proof-system comparison changes
    /// one independent variable at a time.
    pub fn bf16_zkgpt_block(
        &mut self,
        hidden_states: &[Felt<SP1Field>],
        attention_max_hints: &[Felt<SP1Field>],
        params: &Bf16ZkGptBlockParams<'_>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!hidden_states.is_empty(), "BF16 zkGPT block requires a hidden-state matrix");
        assert!(params.num_heads > 0, "BF16 zkGPT block requires at least one attention head");

        let hidden_size = params.layer_norm_1_weight.len();
        assert!(hidden_size > 0, "BF16 zkGPT block requires a nonzero hidden size");
        assert_eq!(hidden_states.len() % hidden_size, 0, "BF16 zkGPT hidden-state shape mismatch");
        assert_eq!(
            hidden_size % params.num_heads,
            0,
            "BF16 zkGPT hidden size must be divisible by its head count"
        );
        let sequence_length = hidden_states.len() / hidden_size;
        let head_dimension = hidden_size / params.num_heads;
        assert_eq!(
            attention_max_hints.len(),
            sequence_length * params.num_heads,
            "BF16 zkGPT attention max hint shape mismatch"
        );
        assert_eq!(
            params.layer_norm_1_bias.len(),
            hidden_size,
            "BF16 zkGPT ln_1 bias shape mismatch"
        );
        assert_eq!(
            params.layer_norm_2_weight.len(),
            hidden_size,
            "BF16 zkGPT ln_2 weight shape mismatch"
        );
        assert_eq!(
            params.layer_norm_2_bias.len(),
            hidden_size,
            "BF16 zkGPT ln_2 bias shape mismatch"
        );
        let qkv_size = hidden_size.checked_mul(3).expect("BF16 zkGPT QKV dimensions overflow");
        assert_eq!(
            params.attention_qkv_weight.len(),
            hidden_size * qkv_size,
            "BF16 zkGPT QKV weight shape mismatch"
        );
        assert_eq!(
            params.attention_projection_weight.len(),
            hidden_size * hidden_size,
            "BF16 zkGPT attention projection weight shape mismatch"
        );
        assert_eq!(
            params.mlp_expansion_weight.len() % hidden_size,
            0,
            "BF16 zkGPT MLP expansion weight shape mismatch"
        );
        let linear_size = params.mlp_expansion_weight.len() / hidden_size;
        assert_eq!(
            linear_size, qkv_size,
            "zkGPT uses one shared 3 * hidden width for QKV and MLP expansion"
        );
        assert_eq!(
            params.mlp_projection_weight.len(),
            linear_size * hidden_size,
            "BF16 zkGPT MLP projection weight shape mismatch"
        );

        let normalized = self.bf16_layer_norm_rows(
            hidden_states,
            hidden_size,
            params.layer_norm_1_weight,
            params.layer_norm_1_bias,
            params.layer_norm_epsilon,
        );
        let qkv =
            self.bf16_linear_rows_no_bias(&normalized, hidden_size, params.attention_qkv_weight);

        let attention_rows: Vec<Vec<_>> =
            (0..sequence_length).ir_par_map_collect(self, |builder, query_token| {
                let query_row = &qkv[query_token * qkv_size..(query_token + 1) * qkv_size];
                let mut token_heads = Vec::with_capacity(params.num_heads);
                for head in 0..params.num_heads {
                    let head_start = head * head_dimension;
                    let head_end = head_start + head_dimension;
                    let query = &query_row[head_start..head_end];
                    let mut causal_keys = Vec::with_capacity((query_token + 1) * head_dimension);
                    let mut causal_values = Vec::with_capacity((query_token + 1) * head_dimension);
                    for key_token in 0..=query_token {
                        let key_value_row = &qkv[key_token * qkv_size..(key_token + 1) * qkv_size];
                        causal_keys.extend_from_slice(
                            &key_value_row[hidden_size + head_start..hidden_size + head_end],
                        );
                        causal_values.extend_from_slice(
                            &key_value_row
                                [2 * hidden_size + head_start..2 * hidden_size + head_end],
                        );
                    }
                    token_heads.push(builder.bf16_attention_head(
                        query,
                        &causal_keys,
                        &causal_values,
                        params.attention_scale,
                        attention_max_hints[query_token * params.num_heads + head],
                    ));
                }
                builder.bf16_merge_attention_heads(&token_heads)
            });
        let attention_context = attention_rows.into_iter().flatten().collect::<Vec<_>>();

        let projected_attention = self.bf16_linear_rows_no_bias(
            &attention_context,
            hidden_size,
            params.attention_projection_weight,
        );
        let normalized_mlp = self.bf16_layer_norm_rows(
            &projected_attention,
            hidden_size,
            params.layer_norm_2_weight,
            params.layer_norm_2_bias,
            params.layer_norm_epsilon,
        );
        let expanded = self.bf16_linear_rows_no_bias(
            &normalized_mlp,
            hidden_size,
            params.mlp_expansion_weight,
        );
        let activated = self.bf16_gelu_new_rows(&expanded, linear_size);
        self.bf16_linear_rows_no_bias(&activated, linear_size, params.mlp_projection_weight)
    }

    /// Evaluate a complete stack of simplified zkGPT-style transformer layers.
    pub fn bf16_zkgpt_stack(
        &mut self,
        hidden_states: &[Felt<SP1Field>],
        attention_max_hints: &Bf16ZkGptAttentionMaxHints,
        params: &Bf16ZkGptStackParams<'_>,
    ) -> Vec<Felt<SP1Field>> {
        assert!(!params.blocks.is_empty(), "BF16 zkGPT stack requires at least one block");
        assert_eq!(
            attention_max_hints.layers.len(),
            params.blocks.len(),
            "BF16 zkGPT attention hint layer count mismatch"
        );

        let mut hidden_states = hidden_states.to_vec();
        for (block, layer_hints) in params.blocks.iter().zip(&attention_max_hints.layers) {
            hidden_states = self.bf16_zkgpt_block(&hidden_states, layer_hints, block);
        }
        hidden_states
    }

    /// Evaluate one token through all GPT-2 transformer blocks and the final `ln_f`.
    ///
    /// Each block consumes and updates only its own KV cache. All layers must have the same head
    /// count and cached-token count, matching GPT-2's autoregressive cache layout.
    pub fn bf16_gpt2_transformer(
        &mut self,
        hidden_state: &[Felt<SP1Field>],
        cache: &Bf16Gpt2Cache,
        attention_max_hints: &Bf16Gpt2AttentionMaxHints,
        params: &Bf16Gpt2TransformerParams<'_>,
    ) -> Bf16Gpt2TransformerOutput {
        assert!(!hidden_state.is_empty(), "BF16 GPT-2 transformer requires a hidden state");
        assert!(!params.blocks.is_empty(), "BF16 GPT-2 transformer requires at least one block");
        assert_eq!(
            cache.layers.len(),
            params.blocks.len(),
            "BF16 GPT-2 transformer cache layer count mismatch"
        );
        assert_eq!(
            attention_max_hints.layers.len(),
            params.blocks.len(),
            "BF16 GPT-2 attention max hint layer count mismatch"
        );
        assert_eq!(
            params.final_layer_norm_weight.len(),
            hidden_state.len(),
            "BF16 GPT-2 final layer norm weight length mismatch"
        );
        assert_eq!(
            params.final_layer_norm_bias.len(),
            hidden_state.len(),
            "BF16 GPT-2 final layer norm bias length mismatch"
        );

        let num_heads = params.blocks[0].num_heads;
        assert!(num_heads > 0, "BF16 GPT-2 transformer requires at least one attention head");
        assert_eq!(
            hidden_state.len() % num_heads,
            0,
            "BF16 GPT-2 hidden size must be divisible by head count"
        );
        let head_dimension = hidden_state.len() / num_heads;
        let cached_tokens = cache.layers[0].validate_shape(num_heads, head_dimension);
        for (layer, (block, layer_hints)) in
            params.blocks.iter().zip(&attention_max_hints.layers).enumerate()
        {
            assert_eq!(
                layer_hints.len(),
                block.num_heads,
                "BF16 GPT-2 attention max hint head count mismatch at layer {layer}"
            );
        }
        for (layer, (block, layer_cache)) in
            params.blocks.iter().zip(&cache.layers).enumerate().skip(1)
        {
            assert_eq!(
                block.num_heads, num_heads,
                "BF16 GPT-2 head count mismatch at layer {layer}"
            );
            assert_eq!(
                layer_cache.validate_shape(num_heads, head_dimension),
                cached_tokens,
                "BF16 GPT-2 cached-token count mismatch at layer {layer}"
            );
        }

        let mut hidden_state = hidden_state.to_vec();
        let mut updated_layers = Vec::with_capacity(params.blocks.len());
        for ((block, layer_cache), layer_hints) in
            params.blocks.iter().zip(&cache.layers).zip(&attention_max_hints.layers)
        {
            let output = self.bf16_gpt2_block(&hidden_state, layer_cache, layer_hints, block);
            hidden_state = output.hidden_state;
            updated_layers.push(output.cache);
        }
        let hidden_state = self.bf16_layer_norm(
            &hidden_state,
            params.final_layer_norm_weight,
            params.final_layer_norm_bias,
            params.final_layer_norm_epsilon,
        );

        Bf16Gpt2TransformerOutput { hidden_state, cache: Bf16Gpt2Cache { layers: updated_layers } }
    }

    /// Project one final GPT-2 hidden state to vocabulary logits using the tied token embedding.
    ///
    /// Unlike GPT-2's internal `Conv1D` matrices, `wte.weight` is row-major
    /// `[vocab_size, hidden_size]`. GPT-2 has no LM Head bias, so every output is exactly one BF16
    /// dot product accumulated from left to right.
    pub fn bf16_gpt2_lm_head(
        &mut self,
        hidden_state: &[Felt<SP1Field>],
        weight: &[Felt<SP1Field>],
    ) -> Vec<Felt<SP1Field>> {
        assert!(!hidden_state.is_empty(), "BF16 GPT-2 LM Head requires a hidden state");
        assert!(!weight.is_empty(), "BF16 GPT-2 LM Head requires at least one vocabulary row");
        assert_eq!(
            weight.len() % hidden_state.len(),
            0,
            "BF16 GPT-2 LM Head weight shape mismatch"
        );

        weight
            .chunks_exact(hidden_state.len())
            .map(|vocabulary_row| self.bf16_dot(hidden_state, vocabulary_row))
            .collect()
    }

    /// Evaluate GPT-2 from an already embedded single-token hidden state through vocabulary logits.
    ///
    /// Token and position embedding are intentionally outside this circuit-facing interface.
    pub fn bf16_gpt2_model(
        &mut self,
        hidden_state: &[Felt<SP1Field>],
        cache: &Bf16Gpt2Cache,
        attention_max_hints: &Bf16Gpt2AttentionMaxHints,
        params: &Bf16Gpt2ModelParams<'_>,
    ) -> Bf16Gpt2ModelOutput {
        let transformer = self.bf16_gpt2_transformer(
            hidden_state,
            cache,
            attention_max_hints,
            &params.transformer,
        );
        let logits = self.bf16_gpt2_lm_head(&transformer.hidden_state, params.lm_head_weight);

        Bf16Gpt2ModelOutput {
            hidden_state: transformer.hidden_state,
            logits,
            cache: transformer.cache,
        }
    }

    /// Compute the mean of a non-empty BF16 vector.
    ///
    /// Values are added from left to right, then the final sum is multiplied by the BF16-rounded
    /// reciprocal of `values.len()`. The length is known while building the circuit, so this avoids
    /// a runtime division event.
    pub fn bf16_mean(&mut self, values: &[Felt<SP1Field>]) -> Felt<SP1Field> {
        assert!(!values.is_empty(), "BF16 mean requires at least one value");

        let divisor_raw = usize_to_bf16_raw(values.len());
        let reciprocal_raw = bf16_reciprocal_raw(divisor_raw);
        let output = self.uninit();
        self.push_op(DslIr::Bf16Mean(Box::new(Bf16MeanOp {
            values: values.into(),
            reciprocal: SP1Field::from_canonical_u16(reciprocal_raw),
            output,
        })));
        output
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
        Bf16UnaryWitness, Executor, Instruction,
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

    fn bf16_constants(
        builder: &mut Builder<crate::circuit::AsmConfig>,
        values: &[u16],
    ) -> Vec<Felt<SP1Field>> {
        values.iter().map(|&value| builder.constant(SP1Field::from_canonical_u16(value))).collect()
    }

    fn reference_bf16_add(lhs: u16, rhs: u16) -> u16 {
        Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Add).output.raw
    }

    fn reference_bf16_sub(lhs: u16, rhs: u16) -> u16 {
        Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Sub).output.raw
    }

    fn reference_bf16_mean(values: &[u16]) -> u16 {
        let sum = values[1..].iter().fold(values[0], |sum, &value| reference_bf16_add(sum, value));
        let reciprocal = bf16_reciprocal_raw(usize_to_bf16_raw(values.len()));
        Bf16MulWitness::new(sum, reciprocal).output.raw
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

    fn reference_bf16_linear_no_bias(
        input: &[u16],
        weight: &[u16],
        output_features: usize,
    ) -> Vec<u16> {
        (0..output_features)
            .map(|output_index| {
                let column = weight
                    .chunks_exact(output_features)
                    .map(|row| row[output_index])
                    .collect::<Vec<_>>();
                reference_bf16_dot(input, &column)
            })
            .collect()
    }

    fn reference_bf16_scaled_attention_score(query: &[u16], key: &[u16], scale: u16) -> u16 {
        let dot = reference_bf16_dot(query, key);
        Bf16MulWitness::new(dot, scale).output.raw
    }

    fn reference_bf16_softmax(values: &[u16], max_hint: u16) -> Vec<u16> {
        let shifted_values =
            values.iter().map(|&value| reference_bf16_sub(value, max_hint)).collect::<Vec<_>>();
        let exponential_values = shifted_values
            .iter()
            .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Exponential, value).output)
            .collect::<Vec<_>>();
        let sum = exponential_values[1..]
            .iter()
            .fold(exponential_values[0], |sum, &value| reference_bf16_add(sum, value));
        let inverse_sum = Bf16DivWitness::new(0x3f80, sum).output.raw;
        exponential_values
            .into_iter()
            .map(|value| Bf16MulWitness::new(value, inverse_sum).output.raw)
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
        max_hint: u16,
    ) -> Vec<u16> {
        let scores = keys
            .chunks_exact(query.len())
            .map(|key| reference_bf16_scaled_attention_score(query, key, scale))
            .collect::<Vec<_>>();
        let probabilities = reference_bf16_softmax(&scores, max_hint);
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

        let root_block = builder.into_root_block();
        assert!(
            root_block.ops.iter().any(|op| matches!(op, DslIr::Bf16DotProduct(_))),
            "dot product must remain compact until assembly lowering"
        );
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(root_block).validate().unwrap());
        assert_eq!(
            program
                .inner
                .iter()
                .filter(|instruction| {
                    matches!(instruction.inner(), Instruction::Bf16LinearBatch(_))
                })
                .count(),
            1
        );
        assert!(!program.inner.iter().any(|instruction| {
            matches!(instruction.inner(), Instruction::Bf16Mul(_) | Instruction::Bf16AddSub(_))
        }));
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), raw_lhs.len());
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_lhs.len() - 1);
    }

    #[test]
    fn compiles_and_executes_singleton_bf16_batches() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let lhs = builder.constant(SP1Field::from_canonical_u16(0x4000));
        let rhs = builder.constant(SP1Field::from_canonical_u16(0x4040));
        let dot = builder.bf16_dot(&[lhs], &[rhs]);
        let mean = builder.bf16_mean(&[dot]);
        builder.assert_felt_eq(dot, SP1Field::from_canonical_u16(0x40c0));
        builder.assert_felt_eq(mean, SP1Field::from_canonical_u16(0x40c0));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 2);
        assert!(executor.record.bf16_add_sub_events.is_empty());
        assert!(executor.record.bf16_div_events.is_empty());
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
    fn compact_bf16_linear_rows_preserve_results_and_events() {
        const INPUT_FEATURES: usize = 2;
        const OUTPUT_FEATURES: usize = 3;
        let raw_input = [
            0x3f80, 0x4000, // row 0
            0xbf00, 0x4040, // row 1
        ];
        let raw_weight = [
            0x3f80, 0x4000, 0xbf80, // input feature 0
            0x4040, 0xbf80, 0x3f00, // input feature 1
        ];
        let expected = raw_input
            .chunks_exact(INPUT_FEATURES)
            .flat_map(|row| reference_bf16_linear_no_bias(row, &raw_weight, OUTPUT_FEATURES))
            .collect::<Vec<_>>();

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let input = bf16_constants(&mut builder, &raw_input);
        let weight = bf16_constants(&mut builder, &raw_weight);
        let output = builder.bf16_linear_rows_no_bias(&input, INPUT_FEATURES, &weight);
        for (&output, &expected) in output.iter().zip(&expected) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }

        let root_block = builder.into_root_block();
        let linear = root_block
            .ops
            .iter()
            .find_map(|op| match op {
                DslIr::Bf16LinearNoBias(op)
                    if op.input_features == INPUT_FEATURES
                        && op.output_features == OUTPUT_FEATURES =>
                {
                    Some(op)
                }
                _ => None,
            })
            .expect("row-wise linear must use one compact DSL operation for all rows");
        assert_eq!(linear.input.len() / linear.input_features, 2);

        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(root_block).validate().unwrap());
        assert_eq!(
            program
                .inner
                .iter()
                .filter(|instruction| {
                    matches!(instruction.inner(), Instruction::Bf16LinearBatch(_))
                })
                .count(),
            1
        );
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 2 * INPUT_FEATURES * OUTPUT_FEATURES);
        assert_eq!(
            executor.record.bf16_add_sub_events.len(),
            2 * (INPUT_FEATURES - 1) * OUTPUT_FEATURES
        );
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
        let raw_max_hint = 0x0000;
        let expected = reference_bf16_softmax(&raw_values, raw_max_hint);
        assert_eq!(expected, [0x3e89, 0x3f3b]);

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let values = raw_values
            .iter()
            .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
            .collect::<Vec<_>>();
        let max_hint = builder.constant(SP1Field::from_canonical_u16(raw_max_hint));
        let output = builder.bf16_softmax(&values, max_hint);
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
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_values.len() * 2 - 1);
        assert_eq!(executor.record.bf16_mul_events.len(), raw_values.len());
        assert_eq!(executor.record.bf16_div_events.len(), 1);
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
        let raw_max_hint = 0x0000;
        let expected = reference_bf16_attention_head(
            &raw_query,
            &raw_keys,
            &raw_values,
            raw_scale,
            raw_max_hint,
        );
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
        let max_hint = builder.constant(SP1Field::from_canonical_u16(raw_max_hint));
        let output = builder.bf16_attention_head(&query, &keys, &values, scale, max_hint);
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
        assert_eq!(executor.record.bf16_mul_events.len(), 12);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 7);
        assert_eq!(executor.record.bf16_unary_events.len(), 2);
        assert!(executor
            .record
            .bf16_unary_events
            .iter()
            .all(|event| event.opcode == Bf16UnaryOpcode::Exponential));
        assert_eq!(executor.record.bf16_div_events.len(), 1);
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
        assert_eq!(executor.record.bf16_mul_events.len(), 14);
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
        assert!(executor.record.bf16_div_events.is_empty());
    }

    #[test]
    fn compiles_and_executes_bf16_gpt2_block() {
        let raw_hidden_state = [0x3f80, 0xbf80];
        let raw_layer_norm_weight = [0x3f80, 0x3f80];
        let raw_layer_norm_bias = [0x0000, 0x0000];
        let raw_attention_qkv_weight = [
            0x3f80, 0x0000, 0x3f80, 0x0000, 0x0000, 0x0000, // hidden feature 0
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // hidden feature 1
        ];
        let raw_attention_qkv_bias = [0x0000; 6];
        let raw_identity_weight = [0x3f80, 0x0000, 0x0000, 0x3f80];
        let raw_zero_bias = [0x0000, 0x0000];

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let hidden_state = bf16_constants(&mut builder, &raw_hidden_state);
        let layer_norm_1_weight = bf16_constants(&mut builder, &raw_layer_norm_weight);
        let layer_norm_1_bias = bf16_constants(&mut builder, &raw_layer_norm_bias);
        let attention_qkv_weight = bf16_constants(&mut builder, &raw_attention_qkv_weight);
        let attention_qkv_bias = bf16_constants(&mut builder, &raw_attention_qkv_bias);
        let attention_projection_weight = bf16_constants(&mut builder, &raw_identity_weight);
        let attention_projection_bias = bf16_constants(&mut builder, &raw_zero_bias);
        let layer_norm_2_weight = bf16_constants(&mut builder, &raw_layer_norm_weight);
        let layer_norm_2_bias = bf16_constants(&mut builder, &raw_layer_norm_bias);
        let mlp_expansion_weight = bf16_constants(&mut builder, &raw_identity_weight);
        let mlp_expansion_bias = bf16_constants(&mut builder, &raw_zero_bias);
        let mlp_projection_weight = bf16_constants(&mut builder, &raw_identity_weight);
        let mlp_projection_bias = bf16_constants(&mut builder, &raw_zero_bias);
        let layer_norm_epsilon = builder.constant(SP1Field::from_canonical_u16(0x0000));
        let attention_scale = builder.constant(SP1Field::from_canonical_u16(0x3f80));
        let attention_max_hint = builder.constant(SP1Field::from_canonical_u16(0x3f80));
        let params = Bf16Gpt2BlockParams {
            layer_norm_1_weight: &layer_norm_1_weight,
            layer_norm_1_bias: &layer_norm_1_bias,
            attention_qkv_weight: &attention_qkv_weight,
            attention_qkv_bias: &attention_qkv_bias,
            attention_projection_weight: &attention_projection_weight,
            attention_projection_bias: &attention_projection_bias,
            layer_norm_2_weight: &layer_norm_2_weight,
            layer_norm_2_bias: &layer_norm_2_bias,
            mlp_expansion_weight: &mlp_expansion_weight,
            mlp_expansion_bias: &mlp_expansion_bias,
            mlp_projection_weight: &mlp_projection_weight,
            mlp_projection_bias: &mlp_projection_bias,
            layer_norm_epsilon,
            attention_scale,
            num_heads: 1,
        };
        let output = builder.bf16_gpt2_block(
            &hidden_state,
            &Bf16KvCache::empty(params.num_heads),
            &[attention_max_hint],
            &params,
        );
        assert_eq!(output.hidden_state.len(), raw_hidden_state.len());
        for (&output, expected) in output.hidden_state.iter().zip([0x3feb, 0xbf94]) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }
        assert_eq!(output.cache.keys.len(), 1);
        assert_eq!(output.cache.values.len(), 1);
        assert_eq!(output.cache.keys[0].len(), 2);
        assert_eq!(output.cache.values[0].len(), 2);
        for (&key, expected) in output.cache.keys[0].iter().zip([0x3f80, 0x0000]) {
            builder.assert_felt_eq(key, SP1Field::from_canonical_u16(expected));
        }
        for (&value, expected) in output.cache.values[0].iter().zip([0x0000, 0x0000]) {
            builder.assert_felt_eq(value, SP1Field::from_canonical_u16(expected));
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 42);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 44);
        assert_eq!(executor.record.bf16_unary_events.len(), 9);
        assert_eq!(executor.record.bf16_div_events.len(), 1);
    }

    #[test]
    fn compiles_and_executes_bf16_zkgpt_stack() {
        const SEQUENCE_LENGTH: usize = 2;
        const HIDDEN_SIZE: usize = 4;
        const NUM_HEADS: usize = 2;
        const LINEAR_SIZE: usize = 3 * HIDDEN_SIZE;
        const NUM_BLOCKS: usize = 2;

        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let hidden_states =
            bf16_constants(&mut builder, &vec![0x0000; SEQUENCE_LENGTH * HIDDEN_SIZE]);
        let layer_norm_weight = bf16_constants(&mut builder, &vec![0x3f80; HIDDEN_SIZE]);
        let layer_norm_bias = bf16_constants(&mut builder, &vec![0x0000; HIDDEN_SIZE]);
        let attention_qkv_weight =
            bf16_constants(&mut builder, &vec![0x3f80; HIDDEN_SIZE * 3 * HIDDEN_SIZE]);
        let attention_projection_weight =
            bf16_constants(&mut builder, &vec![0x3f80; HIDDEN_SIZE * HIDDEN_SIZE]);
        let mlp_expansion_weight =
            bf16_constants(&mut builder, &vec![0x3f80; HIDDEN_SIZE * LINEAR_SIZE]);
        let mlp_projection_weight =
            bf16_constants(&mut builder, &vec![0x3f80; LINEAR_SIZE * HIDDEN_SIZE]);
        let epsilon = builder.constant(SP1Field::from_canonical_u16(0x3f80));
        let attention_scale = builder.constant(SP1Field::from_canonical_u16(0x3f80));
        let zero = builder.constant(SP1Field::zero());
        let block = Bf16ZkGptBlockParams {
            layer_norm_1_weight: &layer_norm_weight,
            layer_norm_1_bias: &layer_norm_bias,
            attention_qkv_weight: &attention_qkv_weight,
            attention_projection_weight: &attention_projection_weight,
            layer_norm_2_weight: &layer_norm_weight,
            layer_norm_2_bias: &layer_norm_bias,
            mlp_expansion_weight: &mlp_expansion_weight,
            mlp_projection_weight: &mlp_projection_weight,
            layer_norm_epsilon: epsilon,
            attention_scale,
            num_heads: NUM_HEADS,
        };
        let blocks = [block; NUM_BLOCKS];
        let params = Bf16ZkGptStackParams { blocks: &blocks };
        let attention_max_hints = Bf16ZkGptAttentionMaxHints {
            layers: vec![vec![zero; SEQUENCE_LENGTH * NUM_HEADS]; NUM_BLOCKS],
        };
        let output = builder.bf16_zkgpt_stack(&hidden_states, &attention_max_hints, &params);
        assert_eq!(output.len(), hidden_states.len());
        for &value in &output {
            builder.assert_felt_eq(value, SP1Field::zero());
        }

        let root_block = builder.into_root_block();
        let parallel_stages = root_block
            .ops
            .iter()
            .filter(|op| matches!(op, DslIr::Parallel(rows) if rows.len() == SEQUENCE_LENGTH))
            .count();
        let linear_batches =
            root_block.ops.iter().filter(|op| matches!(op, DslIr::Bf16LinearNoBias(_))).count();
        assert_eq!(linear_batches, 4 * NUM_BLOCKS);
        assert_eq!(parallel_stages, 4 * NUM_BLOCKS);

        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(root_block).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();

        // One block has 396 multiplications, 334 additions/subtractions, 50 unary lookups, and
        // four divisions for this shape. The stack must repeat the exact computation twice.
        assert_eq!(executor.record.bf16_mul_events.len(), 396 * NUM_BLOCKS);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 334 * NUM_BLOCKS);
        assert_eq!(executor.record.bf16_unary_events.len(), 50 * NUM_BLOCKS);
        assert_eq!(executor.record.bf16_div_events.len(), 4 * NUM_BLOCKS);
    }

    #[test]
    fn compiles_and_executes_bf16_gpt2_model_with_cache() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let hidden_state = bf16_constants(&mut builder, &[0x3f80, 0xbf80]);
        let layer_norm_weight = bf16_constants(&mut builder, &[0x3f80, 0x3f80]);
        let layer_norm_bias = bf16_constants(&mut builder, &[0x0000, 0x0000]);
        let attention_qkv_weight = bf16_constants(&mut builder, &[0x0000; 12]);
        let attention_qkv_bias = bf16_constants(&mut builder, &[0x0000; 6]);
        let attention_projection_weight = bf16_constants(&mut builder, &[0x0000; 4]);
        let attention_projection_bias = bf16_constants(&mut builder, &[0x3f00, 0x0000]);
        let mlp_expansion_weight = bf16_constants(&mut builder, &[0x0000; 4]);
        let mlp_expansion_bias = bf16_constants(&mut builder, &[0x0000, 0x0000]);
        let mlp_projection_weight = bf16_constants(&mut builder, &[0x0000; 4]);
        let mlp_projection_bias = bf16_constants(&mut builder, &[0x0000, 0x3f00]);
        let final_layer_norm_weight = bf16_constants(&mut builder, &[0x4000, 0x3f00]);
        let final_layer_norm_bias = bf16_constants(&mut builder, &[0x3f00, 0xbf00]);
        let lm_head_weight =
            bf16_constants(&mut builder, &[0x3f80, 0x0000, 0x0000, 0x3f80, 0x3f80, 0x3f80]);
        let epsilon = builder.constant(SP1Field::from_canonical_u16(0x0000));
        let attention_scale = builder.constant(SP1Field::from_canonical_u16(0x3f80));
        let block = Bf16Gpt2BlockParams {
            layer_norm_1_weight: &layer_norm_weight,
            layer_norm_1_bias: &layer_norm_bias,
            attention_qkv_weight: &attention_qkv_weight,
            attention_qkv_bias: &attention_qkv_bias,
            attention_projection_weight: &attention_projection_weight,
            attention_projection_bias: &attention_projection_bias,
            layer_norm_2_weight: &layer_norm_weight,
            layer_norm_2_bias: &layer_norm_bias,
            mlp_expansion_weight: &mlp_expansion_weight,
            mlp_expansion_bias: &mlp_expansion_bias,
            mlp_projection_weight: &mlp_projection_weight,
            mlp_projection_bias: &mlp_projection_bias,
            layer_norm_epsilon: epsilon,
            attention_scale,
            num_heads: 1,
        };
        let blocks = [block; 2];
        let transformer = Bf16Gpt2TransformerParams {
            blocks: &blocks,
            final_layer_norm_weight: &final_layer_norm_weight,
            final_layer_norm_bias: &final_layer_norm_bias,
            final_layer_norm_epsilon: epsilon,
        };
        let params = Bf16Gpt2ModelParams { transformer, lm_head_weight: &lm_head_weight };
        let attention_max_hints =
            Bf16Gpt2AttentionMaxHints { layers: vec![vec![epsilon]; blocks.len()] };

        let first = builder.bf16_gpt2_model(
            &hidden_state,
            &Bf16Gpt2Cache::empty(blocks.len(), block.num_heads),
            &attention_max_hints,
            &params,
        );
        for (&output, expected) in first.hidden_state.iter().zip([0x4020, 0xbf80]) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }
        assert_eq!(first.logits.len(), 3);
        for (&logit, expected) in first.logits.iter().zip([0x4020, 0xbf80, 0x3fc0]) {
            builder.assert_felt_eq(logit, SP1Field::from_canonical_u16(expected));
        }
        assert_eq!(first.cache.layers.len(), blocks.len());
        for layer in &first.cache.layers {
            assert_eq!(layer.keys.len(), block.num_heads);
            assert_eq!(layer.values.len(), block.num_heads);
            assert_eq!(layer.keys[0].len(), 2);
            assert_eq!(layer.values[0].len(), 2);
            for &value in layer.keys[0].iter().chain(&layer.values[0]) {
                builder.assert_felt_eq(value, SP1Field::zero());
            }
        }

        let second =
            builder.bf16_gpt2_model(&hidden_state, &first.cache, &attention_max_hints, &params);
        for (&output, expected) in second.hidden_state.iter().zip([0x4020, 0xbf80]) {
            builder.assert_felt_eq(output, SP1Field::from_canonical_u16(expected));
        }
        assert_eq!(second.logits.len(), 3);
        for (&logit, expected) in second.logits.iter().zip([0x4020, 0xbf80, 0x3fc0]) {
            builder.assert_felt_eq(logit, SP1Field::from_canonical_u16(expected));
        }
        assert_eq!(second.cache.layers.len(), blocks.len());
        for layer in &second.cache.layers {
            assert_eq!(layer.keys.len(), block.num_heads);
            assert_eq!(layer.values.len(), block.num_heads);
            assert_eq!(layer.keys[0].len(), 4);
            assert_eq!(layer.values[0].len(), 4);
            for &value in layer.keys[0].iter().chain(&layer.values[0]) {
                builder.assert_felt_eq(value, SP1Field::zero());
            }
        }

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_mul_events.len(), 204);
        assert_eq!(executor.record.bf16_add_sub_events.len(), 206);
        assert_eq!(executor.record.bf16_unary_events.len(), 44);
        assert_eq!(executor.record.bf16_div_events.len(), 4);
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
        let expected_mean = reference_bf16_mean(&raw_values);
        builder.assert_felt_eq(mean, SP1Field::from_canonical_u16(expected_mean));

        let root_block = builder.into_root_block();
        assert!(
            root_block.ops.iter().any(|op| matches!(op, DslIr::Bf16Mean(_))),
            "mean must remain compact until assembly lowering"
        );
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(root_block).validate().unwrap());
        assert_eq!(
            program
                .inner
                .iter()
                .filter(|instruction| {
                    matches!(instruction.inner(), Instruction::Bf16MeanBatch(_))
                })
                .count(),
            1
        );
        assert!(!program.inner.iter().any(|instruction| {
            matches!(instruction.inner(), Instruction::Bf16AddSub(_) | Instruction::Bf16Div(_))
        }));
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), raw_values.len() - 1);
        assert_eq!(executor.record.bf16_mul_events.len(), 1);
        assert!(executor.record.bf16_div_events.is_empty());
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
        let expected_mean = reference_bf16_mean(&raw_values);
        let centered = raw_values
            .iter()
            .map(|&value| reference_bf16_sub(value, expected_mean))
            .collect::<Vec<_>>();
        let squared = centered
            .iter()
            .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, value).output)
            .collect::<Vec<_>>();
        let expected_variance = reference_bf16_mean(&squared);
        builder.assert_felt_eq(mean, SP1Field::from_canonical_u16(expected_mean));
        builder.assert_felt_eq(variance, SP1Field::from_canonical_u16(expected_variance));

        let mut compiler = AsmCompiler::default();
        let program =
            Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.run().unwrap();
        assert_eq!(executor.record.bf16_add_sub_events.len(), 2302);
        assert_eq!(executor.record.bf16_unary_events.len(), 768);
        assert_eq!(executor.record.bf16_mul_events.len(), 2);
        assert!(executor.record.bf16_div_events.is_empty());
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
        assert_eq!(executor.record.bf16_mul_events.len(), 1538);
        assert!(executor.record.bf16_div_events.is_empty());
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
    #[should_panic(expected = "BF16 GPT-2 LM Head weight shape mismatch")]
    fn rejects_mismatched_bf16_gpt2_lm_head_weight() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_gpt2_lm_head(&[value; 2], &[value; 3]);
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
        let max_hint = builder.constant(SP1Field::zero());
        builder.bf16_softmax(&[], max_hint);
    }

    #[test]
    #[should_panic(expected = "BF16 attention value cache shape mismatch")]
    fn rejects_mismatched_bf16_attention_value_cache() {
        let mut builder: Builder<crate::circuit::AsmConfig> = AsmBuilder::default();
        let value = builder.constant(SP1Field::zero());
        builder.bf16_attention_head(&[value; 2], &[value; 4], &[value; 3], value, value);
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
