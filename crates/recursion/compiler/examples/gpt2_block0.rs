#![allow(clippy::print_stdout)]

use std::{
    borrow::BorrowMut,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use slop_algebra::{AbstractField, PrimeField32};
use slop_basefold::FriConfig;
use slop_symmetric::Permutation;
use sp1_hypercube::{
    air::MachineAir, inner_perm, prover::simple_prover, MachineProof, ShardVerifier,
};
use sp1_primitives::{
    fri_params::{unique_decoding_queries, SP1_PROOF_OF_WORK_BITS},
    SP1DiffusionMatrix, SP1ExtensionField, SP1Field,
};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler, CircuitV2Builder},
    prelude::{Bf16Gpt2BlockParams, Bf16KvCache, Builder, Felt},
};
use sp1_recursion_executor::{
    Bf16AddSubOpcode, Bf16AddSubWitness, Bf16DivWitness, Bf16MulWitness, Bf16UnaryOpcode,
    Bf16UnaryWitness, ExecutionRecord, Executor, RecursionProgram, RecursionPublicValues,
    DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH, RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::RecursionAir;

const HIDDEN_SIZE: usize = 768;
const INNER_SIZE: usize = 3072;
const NUM_HEADS: usize = 12;
const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;
const GPT2_ATTENTION_SCALE: u16 = 0x3e00;

const EXPECTED_MUL_EVENTS: usize = 7_082_524;
const EXPECTED_ADD_SUB_EVENTS: usize = 7_086_334;
const EXPECTED_UNARY_EVENTS: usize = 4_622;
const EXPECTED_DIV_EVENTS: usize = 12;
const EXPECTED_IR_OPS: usize = 21_263_028;
const PROOF_LOG_BLOWUP: usize = 1;
const PROOF_LOG_STACKING_HEIGHT: u32 = 22;
const PROOF_MAX_LOG_ROW_COUNT: usize = 23;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Mode {
    #[default]
    Check,
    Build,
    Execute,
    Prove,
}

#[derive(Debug)]
struct Arguments {
    data_dir: PathBuf,
    output_dir: PathBuf,
    mode: Mode,
}

#[derive(Debug)]
struct Block0Data {
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
    pytorch_attention_max_hints: Vec<u16>,
    pytorch_output: Vec<u16>,
}

fn default_data_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .ancestors()
        .nth(3)
        .expect("compiler crate must be nested under the repository root");
    repo_root
        .parent()
        .expect("repository must have a parent directory")
        .join("sp1-models/gpt2-bf16/recursion/block0-once")
}

fn parse_arguments() -> Arguments {
    let mut data_dir = None;
    let mut mode = Mode::Check;
    for argument in std::env::args_os().skip(1) {
        match argument.to_str() {
            Some("--check") => mode = Mode::Check,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some(value) if value.starts_with('-') => panic!("unknown option: {value}"),
            _ if data_dir.is_none() => data_dir = Some(PathBuf::from(argument)),
            _ => panic!("only one data directory may be supplied"),
        }
    }
    let output_dir = std::env::var_os("SP1_GPT2_BLOCK0_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-gpt2-block0-output"));
    Arguments { data_dir: data_dir.unwrap_or_else(default_data_dir), output_dir, mode }
}

fn read_bf16_binary(path: &Path) -> Vec<u16> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 2, 0, "{} has an odd byte length", path.display());
    bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect()
}

fn load_data(data_dir: &Path) -> Block0Data {
    let read = |name: &str| read_bf16_binary(&data_dir.join(name));
    Block0Data {
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
        pytorch_attention_max_hints: read("pytorch_attention_max_hints.bf16.bin"),
        pytorch_output: read("pytorch_block0_output.bf16.bin"),
    }
}

fn validate_data(data: &Block0Data) {
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
    assert_eq!(data.pytorch_attention_max_hints.len(), NUM_HEADS);
    assert_eq!(data.pytorch_output.len(), HIDDEN_SIZE);
}

fn constants(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[u16],
) -> Vec<Felt<SP1Field>> {
    values.iter().map(|&value| builder.constant(SP1Field::from_canonical_u16(value))).collect()
}

fn bf16_to_f64(raw: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(raw) << 16))
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

fn reference_layer_norm(values: &[u16], weight: &[u16], bias: &[u16], epsilon: u16) -> Vec<u16> {
    assert_eq!(values.len(), weight.len());
    assert_eq!(values.len(), bias.len());
    let mean = reference_mean(values);
    let centered = values.iter().map(|&value| reference_sub(value, mean)).collect::<Vec<_>>();
    let squared = centered
        .iter()
        .map(|&value| Bf16UnaryWitness::new(Bf16UnaryOpcode::Square, value).output)
        .collect::<Vec<_>>();
    let variance = reference_mean(&squared);
    let variance_with_epsilon = reference_add(variance, epsilon);
    let inverse_standard_deviation =
        Bf16UnaryWitness::new(Bf16UnaryOpcode::Rsqrt, variance_with_epsilon).output;
    centered
        .into_iter()
        .zip(weight)
        .zip(bias)
        .map(|((value, &weight), &bias)| {
            let normalized = reference_mul(value, inverse_standard_deviation);
            let scaled = reference_mul(normalized, weight);
            reference_add(scaled, bias)
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

fn reference_block0(data: &Block0Data) -> (Vec<u16>, Vec<u16>) {
    let normalized = reference_layer_norm(
        &data.hidden_state,
        &data.layer_norm_1_weight,
        &data.layer_norm_1_bias,
        GPT2_LAYER_NORM_EPSILON,
    );
    let qkv = reference_linear(&normalized, &data.attention_qkv_weight, &data.attention_qkv_bias);
    let (queries, key_values) = qkv.split_at(HIDDEN_SIZE);
    let (keys, values) = key_values.split_at(HIDDEN_SIZE);
    let head_dimension = HIDDEN_SIZE / NUM_HEADS;
    let mut attention_max_hints = Vec::with_capacity(NUM_HEADS);
    let mut heads = Vec::with_capacity(NUM_HEADS);
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
        GPT2_LAYER_NORM_EPSILON,
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

fn compare_outputs(actual: &[u16], reference: &[u16]) {
    assert_eq!(actual.len(), reference.len());
    let exact_matches =
        actual.iter().zip(reference).filter(|(actual, expected)| actual == expected).count();
    let mut absolute_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut max_absolute_error = 0.0;
    let mut max_absolute_error_index = 0;
    for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let error = (bf16_to_f64(actual) - bf16_to_f64(reference)).abs();
        absolute_sum += error;
        squared_sum += error * error;
        if error > max_absolute_error {
            max_absolute_error = error;
            max_absolute_error_index = index;
        }
    }
    let count = actual.len() as f64;
    println!("exact PyTorch BF16 matches: {exact_matches}/{}", actual.len());
    println!(
        "vs PyTorch BF16: MAE={:.9} RMSE={:.9} max_abs={max_absolute_error:.9}@{max_absolute_error_index}",
        absolute_sum / count,
        (squared_sum / count).sqrt(),
    );
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

fn digest_hex(digest: &[SP1Field; DIGEST_SIZE]) -> String {
    digest.iter().map(|value| format!("{:08X}\n", value.as_canonical_u32())).collect()
}

async fn prove_and_verify(
    program: Arc<RecursionProgram<SP1Field>>,
    record: ExecutionRecord<SP1Field>,
    output_dir: &Path,
) {
    type A = RecursionAir<SP1Field, 3, 2>;

    let machine = A::verillm_machine();
    println!(
        "proof config: log_blowup={PROOF_LOG_BLOWUP} \
         log_stacking_height={PROOF_LOG_STACKING_HEIGHT} \
         max_log_row_count={PROOF_MAX_LOG_ROW_COUNT}"
    );
    let max_rows = 1usize << PROOF_MAX_LOG_ROW_COUNT;
    for chip in machine.chips() {
        if chip.included(&record) {
            let rows = chip.num_rows(&record).unwrap_or_default();
            println!("trace: {:<18} rows={rows}", chip.name());
            assert!(
                rows <= max_rows,
                "{} needs {rows} rows, exceeding the configured maximum {max_rows}",
                chip.name()
            );
        }
    }

    let verifier = ShardVerifier::from_basefold_parameters(
        FriConfig::new(
            PROOF_LOG_BLOWUP,
            unique_decoding_queries(PROOF_LOG_BLOWUP),
            SP1_PROOF_OF_WORK_BITS,
        ),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        machine,
    );
    let prover = simple_prover(verifier);
    let shape = prover.shape_from_record(&record).expect("VeriLLM machine has no proof shape");
    println!("proof shape: {shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    println!("proof setup: elapsed={:.3}s", setup_started.elapsed().as_secs_f64());
    // The development prover has one permit. Setup is complete, so release its permit before
    // proving the main trace, matching `run_test_recursion` in the machine test helpers.
    let pk = unsafe { pk.into_inner() };

    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    println!("proof generated: elapsed={:.3}s", prove_started.elapsed().as_secs_f64());

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated Block 0 proof must verify");
    println!("proof verified: elapsed={:.3}s", verify_started.elapsed().as_secs_f64());

    let proof_path = output_dir.join("gpt2_block0.proof.bin");
    let vk_path = output_dir.join("gpt2_block0.vk.bin");
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    println!(
        "proof artifacts: proof={} ({} bytes) vk={} ({} bytes)",
        proof_path.display(),
        fs::metadata(&proof_path).unwrap().len(),
        vk_path.display(),
        fs::metadata(&vk_path).unwrap().len(),
    );
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let load_started = Instant::now();
    let data = load_data(&arguments.data_dir);
    validate_data(&data);
    let parameter_count = data.layer_norm_1_weight.len()
        + data.layer_norm_1_bias.len()
        + data.attention_qkv_weight.len()
        + data.attention_qkv_bias.len()
        + data.attention_projection_weight.len()
        + data.attention_projection_bias.len()
        + data.layer_norm_2_weight.len()
        + data.layer_norm_2_bias.len()
        + data.mlp_expansion_weight.len()
        + data.mlp_expansion_bias.len()
        + data.mlp_projection_weight.len()
        + data.mlp_projection_bias.len();
    println!("data_dir={}", arguments.data_dir.display());
    println!("output_dir={}", arguments.output_dir.display());
    println!("mode={:?}", arguments.mode);
    println!(
        "loaded: parameters={parameter_count} hidden={} max_hints={} elapsed={:.3}s",
        data.hidden_state.len(),
        data.pytorch_attention_max_hints.len(),
        load_started.elapsed().as_secs_f64(),
    );
    println!(
        "estimated BF16 events: mul={EXPECTED_MUL_EVENTS} add_sub={EXPECTED_ADD_SUB_EVENTS} \
         unary={EXPECTED_UNARY_EVENTS} div={EXPECTED_DIV_EVENTS}"
    );
    println!("expected IR operations: {EXPECTED_IR_OPS}");
    if arguments.mode == Mode::Check {
        println!(
            "data validation complete; pass --build, --execute, or --prove to construct the full block"
        );
        return;
    }

    let reference_started = Instant::now();
    let (reference_output, attention_max_hints_raw) = reference_block0(&data);
    let expected_output_digest = poseidon2_hash_raw_bf16(&reference_output);
    let pytorch_hint_matches = attention_max_hints_raw
        .iter()
        .zip(&data.pytorch_attention_max_hints)
        .filter(|(reference, pytorch)| reference == pytorch)
        .count();
    println!(
        "host reference: elapsed={:.3}s PyTorch_max_hint_matches={pytorch_hint_matches}/{NUM_HEADS}",
        reference_started.elapsed().as_secs_f64()
    );

    let build_started = Instant::now();
    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let hidden_state = constants(&mut builder, &data.hidden_state);
    let layer_norm_1_weight = constants(&mut builder, &data.layer_norm_1_weight);
    let layer_norm_1_bias = constants(&mut builder, &data.layer_norm_1_bias);
    let attention_qkv_weight = constants(&mut builder, &data.attention_qkv_weight);
    let attention_qkv_bias = constants(&mut builder, &data.attention_qkv_bias);
    let attention_projection_weight = constants(&mut builder, &data.attention_projection_weight);
    let attention_projection_bias = constants(&mut builder, &data.attention_projection_bias);
    let layer_norm_2_weight = constants(&mut builder, &data.layer_norm_2_weight);
    let layer_norm_2_bias = constants(&mut builder, &data.layer_norm_2_bias);
    let mlp_expansion_weight = constants(&mut builder, &data.mlp_expansion_weight);
    let mlp_expansion_bias = constants(&mut builder, &data.mlp_expansion_bias);
    let mlp_projection_weight = constants(&mut builder, &data.mlp_projection_weight);
    let mlp_projection_bias = constants(&mut builder, &data.mlp_projection_bias);
    let attention_max_hints = constants(&mut builder, &attention_max_hints_raw);
    let layer_norm_epsilon =
        builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let attention_scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));
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
        &hidden_state,
        &Bf16KvCache::empty(NUM_HEADS),
        &attention_max_hints,
        &params,
    );
    commit_bf16_output(&mut builder, &output.hidden_state);
    for &value in &output.hidden_state {
        builder.print_f(value);
    }
    let block = builder.into_root_block();
    println!(
        "built: ir_ops={} elapsed={:.3}s",
        block.ops.len(),
        build_started.elapsed().as_secs_f64()
    );
    assert_eq!(block.ops.len(), EXPECTED_IR_OPS);
    if arguments.mode == Mode::Build {
        return;
    }

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    println!("compiled: elapsed={:.3}s", compile_started.elapsed().as_secs_f64());
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let print_path = arguments.output_dir.join("circuit_block0_output.txt");
    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.debug_stdout = Box::new(File::create(&print_path).unwrap());
    let execute_started = Instant::now();
    executor.run().unwrap();
    println!("executed: elapsed={:.3}s", execute_started.elapsed().as_secs_f64());
    assert_eq!(executor.record.bf16_mul_events.len(), EXPECTED_MUL_EVENTS);
    assert_eq!(executor.record.bf16_add_sub_events.len(), EXPECTED_ADD_SUB_EVENTS);
    assert_eq!(executor.record.bf16_unary_events.len(), EXPECTED_UNARY_EVENTS);
    assert_eq!(executor.record.bf16_div_events.len(), EXPECTED_DIV_EVENTS);
    assert_eq!(executor.record.poseidon2_events.len(), HIDDEN_SIZE.div_ceil(HASH_RATE));
    assert_eq!(executor.record.commit_pv_hash_events.len(), 1);
    assert_eq!(executor.record.public_values.digest, expected_output_digest);
    println!(
        "events: mul={} add_sub={} unary={} div={}",
        executor.record.bf16_mul_events.len(),
        executor.record.bf16_add_sub_events.len(),
        executor.record.bf16_unary_events.len(),
        executor.record.bf16_div_events.len(),
    );
    println!(
        "public output digest: {}",
        expected_output_digest
            .iter()
            .map(|value| format!("{:08X}", value.as_canonical_u32()))
            .collect::<Vec<_>>()
            .join(":")
    );
    let proof_record = if arguments.mode == Mode::Prove {
        Some(std::mem::take(&mut executor.record))
    } else {
        None
    };
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
    assert_eq!(circuit_output.len(), HIDDEN_SIZE);
    assert_eq!(circuit_output, reference_output, "circuit output differs from BF16 host reference");
    println!("exact BF16 host-reference matches: {HIDDEN_SIZE}/{HIDDEN_SIZE}");
    fs::write(
        arguments.output_dir.join("circuit_block0_output.bf16.hex"),
        circuit_output.iter().map(|value| format!("{value:04X}\n")).collect::<String>(),
    )
    .unwrap();
    fs::write(
        arguments.output_dir.join("circuit_block0_output.poseidon2.hex"),
        digest_hex(&expected_output_digest),
    )
    .unwrap();
    compare_outputs(&circuit_output, &data.pytorch_output);
    println!("circuit output: {}", print_path.display());
    if let Some(record) = proof_record {
        prove_and_verify(program, record, &arguments.output_dir).await;
    }
}
