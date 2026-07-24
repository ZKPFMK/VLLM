#![allow(clippy::print_stdout)]

use std::{
    borrow::BorrowMut,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use slop_algebra::{AbstractField, PrimeField32};
use slop_symmetric::Permutation;
use sp1_hypercube::inner_perm;
use sp1_primitives::{SP1DiffusionMatrix, SP1ExtensionField, SP1Field};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler, CircuitV2Builder},
    prelude::{Bf16Gpt2BlockParams, Bf16KvCache, Builder, Felt},
};
use sp1_recursion_executor::{
    Bf16AddSubOpcode, Bf16AddSubWitness, Bf16DivWitness, Bf16MulWitness, Bf16UnaryOpcode,
    Bf16UnaryWitness, Executor, RecursionPublicValues, DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH,
    RECURSIVE_PROOF_NUM_PV_ELTS,
};

const NUM_BLOCKS: usize = 12;
const EVENTS_PER_ROW: usize = 12;
const HIDDEN_SIZE: usize = 768;
const INNER_SIZE: usize = 3072;
const NUM_HEADS: usize = 12;
const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;
const GPT2_ATTENTION_SCALE: u16 = 0x3e00;

const MUL_EVENTS_PER_BLOCK: usize = 7_082_524;
const ADD_SUB_EVENTS_PER_BLOCK: usize = 7_086_334;
const UNARY_EVENTS_PER_BLOCK: usize = 4_622;
const DIV_EVENTS_PER_BLOCK: usize = 12;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Mode {
    #[default]
    Build,
    Execute,
}

#[derive(Debug)]
struct Arguments {
    data_root: PathBuf,
    output_dir: PathBuf,
    mode: Mode,
}

#[derive(Debug)]
struct BlockData {
    hidden_state: Vec<u16>,
    layer_norm_1_weight: Vec<u16>,
    layer_norm_1_bias: Vec<u16>,
    attention_qkv_weight: Vec<u16>,
    attention_qkv_bias: Vec<u16>,
    attention_projection_weight: Vec<u16>,
    attention_projection_bias: Vec<u16>,
    layer_norm_2_weight: Vec<u16>,
    layer_norm_2_bias: Vec<u16>,
    mlp_expansion_weight: Vec<u16>,
    mlp_expansion_bias: Vec<u16>,
    mlp_projection_weight: Vec<u16>,
    mlp_projection_bias: Vec<u16>,
}

fn parse_arguments() -> Arguments {
    let mut data_root = None;
    let mut mode = Mode::Build;
    for argument in std::env::args_os().skip(1) {
        match argument.to_str() {
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some(value) if value.starts_with('-') => panic!("unknown option: {value}"),
            _ if data_root.is_none() => data_root = Some(PathBuf::from(argument)),
            _ => panic!("only one data root may be supplied"),
        }
    }
    let output_dir = std::env::var_os("SP1_GPT2_12_BLOCK_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-gpt2-12-block-output"));
    Arguments {
        data_root: data_root.unwrap_or_else(|| std::env::temp_dir().join("sp1-gpt2-12-layer-real")),
        output_dir,
        mode,
    }
}

fn read_bf16_binary(path: &Path) -> Vec<u16> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 2, 0, "{} has an odd byte length", path.display());
    bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect()
}

fn load_data(data_dir: &Path) -> BlockData {
    let read = |name: &str| read_bf16_binary(&data_dir.join(name));
    BlockData {
        hidden_state: read("hidden_state.bf16.bin"),
        layer_norm_1_weight: read("ln_1_weight.bf16.bin"),
        layer_norm_1_bias: read("ln_1_bias.bf16.bin"),
        attention_qkv_weight: read("attention_qkv_weight.bf16.bin"),
        attention_qkv_bias: read("attention_qkv_bias.bf16.bin"),
        attention_projection_weight: read("attention_projection_weight.bf16.bin"),
        attention_projection_bias: read("attention_projection_bias.bf16.bin"),
        layer_norm_2_weight: read("ln_2_weight.bf16.bin"),
        layer_norm_2_bias: read("ln_2_bias.bf16.bin"),
        mlp_expansion_weight: read("mlp_expansion_weight.bf16.bin"),
        mlp_expansion_bias: read("mlp_expansion_bias.bf16.bin"),
        mlp_projection_weight: read("mlp_projection_weight.bf16.bin"),
        mlp_projection_bias: read("mlp_projection_bias.bf16.bin"),
    }
}

fn validate_data(data: &BlockData) {
    assert_eq!(data.hidden_state.len(), HIDDEN_SIZE);
    assert_eq!(data.layer_norm_1_weight.len(), HIDDEN_SIZE);
    assert_eq!(data.layer_norm_1_bias.len(), HIDDEN_SIZE);
    assert_eq!(data.attention_qkv_weight.len(), HIDDEN_SIZE * 3 * HIDDEN_SIZE);
    assert_eq!(data.attention_qkv_bias.len(), 3 * HIDDEN_SIZE);
    assert_eq!(data.attention_projection_weight.len(), HIDDEN_SIZE * HIDDEN_SIZE);
    assert_eq!(data.attention_projection_bias.len(), HIDDEN_SIZE);
    assert_eq!(data.layer_norm_2_weight.len(), HIDDEN_SIZE);
    assert_eq!(data.layer_norm_2_bias.len(), HIDDEN_SIZE);
    assert_eq!(data.mlp_expansion_weight.len(), HIDDEN_SIZE * INNER_SIZE);
    assert_eq!(data.mlp_expansion_bias.len(), INNER_SIZE);
    assert_eq!(data.mlp_projection_weight.len(), INNER_SIZE * HIDDEN_SIZE);
    assert_eq!(data.mlp_projection_bias.len(), HIDDEN_SIZE);
}

fn constants(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[u16],
) -> Vec<Felt<SP1Field>> {
    values.iter().map(|&value| builder.constant(SP1Field::from_canonical_u16(value))).collect()
}

fn usize_to_bf16_raw(value: usize) -> u16 {
    assert!(value > 0);
    let exponent = value.ilog2();
    let mantissa = if exponent <= 7 { value << (7 - exponent) } else { value >> (exponent - 7) };
    (((exponent + 127) as u16) << 7) | (mantissa as u16 - 128)
}

fn reference_add(lhs: u16, rhs: u16) -> u16 {
    Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Add).output.raw
}

fn reference_sub(lhs: u16, rhs: u16) -> u16 {
    Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Sub).output.raw
}

fn reference_mul(lhs: u16, rhs: u16) -> u16 {
    Bf16MulWitness::new(lhs, rhs).output.raw
}

fn reference_dot(lhs: &[u16], rhs: &[u16]) -> u16 {
    assert!(!lhs.is_empty());
    assert_eq!(lhs.len(), rhs.len());
    let mut sum = reference_mul(lhs[0], rhs[0]);
    for (&lhs, &rhs) in lhs[1..].iter().zip(&rhs[1..]) {
        sum = reference_add(sum, reference_mul(lhs, rhs));
    }
    sum
}

fn reference_mean(values: &[u16]) -> u16 {
    assert!(!values.is_empty());
    let mut sum = values[0];
    for &value in &values[1..] {
        sum = reference_add(sum, value);
    }
    let reciprocal = Bf16DivWitness::new(0x3f80, usize_to_bf16_raw(values.len())).output.raw;
    reference_mul(sum, reciprocal)
}

fn reference_layer_norm(values: &[u16], weight: &[u16], bias: &[u16]) -> Vec<u16> {
    let mean = reference_mean(values);
    let centered = values.iter().map(|&value| reference_sub(value, mean)).collect::<Vec<_>>();
    let squared = centered
        .iter()
        .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, value).output)
        .collect::<Vec<_>>();
    let variance = reference_mean(&squared);
    let variance_with_epsilon = reference_add(variance, GPT2_LAYER_NORM_EPSILON);
    let inverse_standard_deviation =
        Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, variance_with_epsilon).output;
    centered
        .into_iter()
        .zip(weight)
        .zip(bias)
        .map(|((value, &weight), &bias)| {
            reference_add(
                reference_mul(reference_mul(value, inverse_standard_deviation), weight),
                bias,
            )
        })
        .collect()
}

fn reference_linear(input: &[u16], weight: &[u16], bias: &[u16]) -> Vec<u16> {
    let output_features = bias.len();
    assert_eq!(weight.len(), input.len() * output_features);
    (0..output_features)
        .map(|output| {
            let mut sum = reference_mul(input[0], weight[output]);
            for input_index in 1..input.len() {
                let product = reference_mul(
                    input[input_index],
                    weight[input_index * output_features + output],
                );
                sum = reference_add(sum, product);
            }
            reference_add(sum, bias[output])
        })
        .collect()
}

fn reference_vector_add(lhs: &[u16], rhs: &[u16]) -> Vec<u16> {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs).map(|(&lhs, &rhs)| reference_add(lhs, rhs)).collect()
}

fn reference_block(data: &BlockData) -> (Vec<u16>, Vec<u16>) {
    let normalized = reference_layer_norm(
        &data.hidden_state,
        &data.layer_norm_1_weight,
        &data.layer_norm_1_bias,
    );
    let qkv = reference_linear(&normalized, &data.attention_qkv_weight, &data.attention_qkv_bias);
    let (queries, key_values) = qkv.split_at(HIDDEN_SIZE);
    let (keys, values) = key_values.split_at(HIDDEN_SIZE);
    let head_dimension = HIDDEN_SIZE / NUM_HEADS;
    let mut attention_max_hints = Vec::with_capacity(NUM_HEADS);
    let mut heads = Vec::with_capacity(HIDDEN_SIZE);
    for head in 0..NUM_HEADS {
        let range = head * head_dimension..(head + 1) * head_dimension;
        let score = reference_mul(
            reference_dot(&queries[range.clone()], &keys[range.clone()]),
            GPT2_ATTENTION_SCALE,
        );
        attention_max_hints.push(score);
        let shifted_score = reference_sub(score, score);
        let exponential = Bf16UnaryWitness::new(Bf16UnaryOpcode::Exponential, shifted_score).output;
        let inverse_sum = Bf16DivWitness::new(0x3f80, exponential).output.raw;
        let probability = reference_mul(exponential, inverse_sum);
        heads.extend(values[range].iter().map(|&value| reference_mul(probability, value)));
    }

    let projected_attention = reference_linear(
        &heads,
        &data.attention_projection_weight,
        &data.attention_projection_bias,
    );
    let attention_residual = reference_vector_add(&data.hidden_state, &projected_attention);
    let normalized_mlp = reference_layer_norm(
        &attention_residual,
        &data.layer_norm_2_weight,
        &data.layer_norm_2_bias,
    );
    let expanded =
        reference_linear(&normalized_mlp, &data.mlp_expansion_weight, &data.mlp_expansion_bias);
    let activated = expanded
        .into_iter()
        .map(|value| Bf16UnaryWitness::new(Bf16UnaryOpcode::GeluNew, value).output)
        .collect::<Vec<_>>();
    let projected_mlp =
        reference_linear(&activated, &data.mlp_projection_weight, &data.mlp_projection_bias);
    (reference_vector_add(&attention_residual, &projected_mlp), attention_max_hints)
}

fn poseidon2_hash_raw_bf16(values: &[u16]) -> [SP1Field; DIGEST_SIZE] {
    let mut state = [SP1Field::zero(); PERMUTATION_WIDTH];
    for chunk in values.chunks(HASH_RATE) {
        for (destination, &value) in state.iter_mut().zip(chunk) {
            *destination = SP1Field::from_canonical_u16(value);
        }
        inner_perm().permute_mut(&mut state);
    }
    state[..DIGEST_SIZE].try_into().unwrap()
}

fn commit_bf16_output(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[Felt<SP1Field>],
) {
    let zero = builder.constant(SP1Field::zero());
    let mut state = [zero; PERMUTATION_WIDTH];
    for chunk in values.chunks(HASH_RATE) {
        state[..chunk.len()].copy_from_slice(chunk);
        state = builder.poseidon2_permute_v2(state);
    }
    let digest: [Felt<SP1Field>; DIGEST_SIZE] = state[..DIGEST_SIZE].try_into().unwrap();
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = digest;
    builder.commit_public_values_v2(*public_values);
}

fn padded_rows(events: usize) -> usize {
    events.div_ceil(EVENTS_PER_ROW).next_multiple_of(32)
}

fn block_data_dir(root: &Path, block: usize) -> PathBuf {
    root.join(format!("block-{block}-data"))
}

fn main() {
    let arguments = parse_arguments();
    println!("data_root={}", arguments.data_root.display());
    println!("output_dir={}", arguments.output_dir.display());
    println!("mode={:?}", arguments.mode);
    println!("blocks={NUM_BLOCKS} events_per_row={EVENTS_PER_ROW}");

    let expected_mul_events = MUL_EVENTS_PER_BLOCK * NUM_BLOCKS;
    let expected_add_sub_events = ADD_SUB_EVENTS_PER_BLOCK * NUM_BLOCKS;
    let expected_unary_events = UNARY_EVENTS_PER_BLOCK * NUM_BLOCKS;
    let expected_div_events = DIV_EVENTS_PER_BLOCK * NUM_BLOCKS;
    println!(
        "estimated BF16 events: mul={expected_mul_events} add_sub={expected_add_sub_events} \
         unary={expected_unary_events} div={expected_div_events}"
    );
    println!(
        "packed trace rows (before proof shape): mul={} add_sub={} max_2^23={}",
        padded_rows(expected_mul_events),
        padded_rows(expected_add_sub_events),
        1usize << 23,
    );

    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let layer_norm_epsilon =
        builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let attention_scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));
    let mut circuit_hidden = None;
    let mut host_hidden = None;

    for layer in 0..NUM_BLOCKS {
        let layer_started = Instant::now();
        let data_dir = block_data_dir(&arguments.data_root, layer);
        let data = load_data(&data_dir);
        validate_data(&data);
        if let Some(expected_input) = &host_hidden {
            assert_eq!(
                &data.hidden_state, expected_input,
                "block {layer} fixture is not chained from the previous exact BF16 output"
            );
        } else {
            circuit_hidden = Some(constants(&mut builder, &data.hidden_state));
        }

        let reference_started = Instant::now();
        let (reference_output, attention_max_hints_raw) = reference_block(&data);
        let reference_elapsed = reference_started.elapsed().as_secs_f64();

        let layer_norm_1_weight = constants(&mut builder, &data.layer_norm_1_weight);
        let layer_norm_1_bias = constants(&mut builder, &data.layer_norm_1_bias);
        let attention_qkv_weight = constants(&mut builder, &data.attention_qkv_weight);
        let attention_qkv_bias = constants(&mut builder, &data.attention_qkv_bias);
        let attention_projection_weight =
            constants(&mut builder, &data.attention_projection_weight);
        let attention_projection_bias = constants(&mut builder, &data.attention_projection_bias);
        let layer_norm_2_weight = constants(&mut builder, &data.layer_norm_2_weight);
        let layer_norm_2_bias = constants(&mut builder, &data.layer_norm_2_bias);
        let mlp_expansion_weight = constants(&mut builder, &data.mlp_expansion_weight);
        let mlp_expansion_bias = constants(&mut builder, &data.mlp_expansion_bias);
        let mlp_projection_weight = constants(&mut builder, &data.mlp_projection_weight);
        let mlp_projection_bias = constants(&mut builder, &data.mlp_projection_bias);
        let attention_max_hints = constants(&mut builder, &attention_max_hints_raw);
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
            num_heads: NUM_HEADS,
        };
        let output = builder.bf16_gpt2_block(
            circuit_hidden.as_ref().unwrap(),
            &Bf16KvCache::empty(NUM_HEADS),
            &attention_max_hints,
            &params,
        );
        circuit_hidden = Some(output.hidden_state);
        host_hidden = Some(reference_output);
        println!(
            "built layer={layer} reference={reference_elapsed:.3}s total_layer={:.3}s",
            layer_started.elapsed().as_secs_f64()
        );
    }

    let circuit_hidden = circuit_hidden.unwrap();
    let host_hidden = host_hidden.unwrap();
    let expected_output_digest = poseidon2_hash_raw_bf16(&host_hidden);
    commit_bf16_output(&mut builder, &circuit_hidden);
    for &value in &circuit_hidden {
        builder.print_f(value);
    }
    let block = builder.into_root_block();
    println!(
        "built all layers: ir_ops={} elapsed={:.3}s",
        block.ops.len(),
        build_started.elapsed().as_secs_f64()
    );
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    println!("compiled: elapsed={:.3}s", compile_started.elapsed().as_secs_f64());
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let print_path = arguments.output_dir.join("circuit_12_block_output.txt");
    let mut executor =
        Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
    executor.debug_stdout = Box::new(File::create(&print_path).unwrap());
    let execute_started = Instant::now();
    executor.run().unwrap();
    println!("executed: elapsed={:.3}s", execute_started.elapsed().as_secs_f64());
    assert_eq!(executor.record.bf16_mul_events.len(), expected_mul_events);
    assert_eq!(executor.record.bf16_add_sub_events.len(), expected_add_sub_events);
    assert_eq!(executor.record.bf16_unary_events.len(), expected_unary_events);
    assert_eq!(executor.record.bf16_div_events.len(), expected_div_events);
    assert_eq!(executor.record.public_values.digest, expected_output_digest);
    println!(
        "events: mul={} add_sub={} unary={} div={}",
        executor.record.bf16_mul_events.len(),
        executor.record.bf16_add_sub_events.len(),
        executor.record.bf16_unary_events.len(),
        executor.record.bf16_div_events.len(),
    );
    drop(executor);

    let circuit_output = fs::read_to_string(&print_path)
        .unwrap()
        .lines()
        .map(|line| {
            line.strip_prefix("PRINTF=")
                .unwrap_or_else(|| panic!("unexpected circuit output line: {line}"))
                .parse::<u16>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(circuit_output, host_hidden, "12-block circuit output differs from host reference");
    println!("exact BF16 host-reference matches: {HIDDEN_SIZE}/{HIDDEN_SIZE}");
    println!(
        "public output digest: {}",
        expected_output_digest
            .iter()
            .map(|value| format!("{:08X}", value.as_canonical_u32()))
            .collect::<Vec<_>>()
            .join(":")
    );
    println!("circuit output: {}", print_path.display());
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}
