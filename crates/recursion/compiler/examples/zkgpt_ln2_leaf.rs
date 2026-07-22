#![allow(clippy::print_stdout)]

use std::{
    borrow::{Borrow, BorrowMut},
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use slop_algebra::{AbstractField, PrimeField32};
use slop_basefold::FriConfig;
use slop_symmetric::Permutation;
use sp1_hypercube::{
    air::MachineAir, inner_perm, prover::simple_prover, MachineProof, MachineVerifyingKey,
    SP1PcsProofInner, ShardVerifier,
};
use sp1_primitives::{
    fri_params::{unique_decoding_queries, SP1_PROOF_OF_WORK_BITS},
    SP1DiffusionMatrix, SP1ExtensionField, SP1Field, SP1GlobalContext,
};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler, AsmConfig, CircuitV2Builder},
    prelude::{Builder, Felt},
};
use sp1_recursion_executor::{
    Bf16AddSubEvent, Bf16AddSubOpcode, Bf16AddSubWitness, Bf16DivEvent, Bf16DivWitness,
    Bf16MulEvent, Bf16MulWitness, Bf16UnaryEvent, Bf16UnaryOpcode, Bf16UnaryWitness,
    ExecutionRecord, Executor, RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE,
    PERMUTATION_WIDTH, RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::{
    chips::bf16::{NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW},
    RecursionAir,
};

const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_LAYER: usize = 0;

const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;

// These stage and domain constants extend the commitment chain established by
// `zkgpt_c_proj_leaf` and `zkgpt_c_proj_join`.
const PROTOCOL_VERSION: u32 = 1;
const LN2_STAGE: u32 = 5;
const DOMAIN_C_PROJ_GROUP_OUTPUT: u32 = 0x1212;
const DOMAIN_LN2_PARAMETERS: u32 = 0x1301;
const DOMAIN_LN2_OUTPUT: u32 = 0x1302;
const DOMAIN_LN2_TRANSCRIPT: u32 = 0x1303;

const PROOF_LOG_BLOWUP: usize = 1;
const FULL_MAX_LOG_ROWS: usize = 16;
const SMALL_MAX_LOG_ROWS: usize = 16;
const UPSTREAM_JOIN_MAX_LOG_ROWS: usize = 16;

type Ln2Builder = Builder<AsmConfig>;
type Digest = [SP1Field; DIGEST_SIZE];
type StoredProof = MachineProof<SP1GlobalContext, SP1PcsProofInner>;
type StoredVerifyingKey = MachineVerifyingKey<SP1GlobalContext>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Mode {
    #[default]
    Estimate,
    Build,
    Execute,
    Prove,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    sequence_length: usize,
    hidden_size: usize,
}

impl Shape {
    fn full() -> Self {
        Self { sequence_length: DEFAULT_SEQUENCE_LENGTH, hidden_size: DEFAULT_HIDDEN_SIZE }
    }

    fn small() -> Self {
        Self { sequence_length: 2, hidden_size: 8 }
    }

    fn validate(self) {
        assert!(self.sequence_length > 0, "LN2 sequence length must be nonzero");
        assert!(self.hidden_size > 0, "LN2 hidden size must be nonzero");
    }

    fn max_log_rows(self) -> usize {
        if self.sequence_length == DEFAULT_SEQUENCE_LENGTH
            && self.hidden_size == DEFAULT_HIDDEN_SIZE
        {
            FULL_MAX_LOG_ROWS
        } else {
            SMALL_MAX_LOG_ROWS
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EventCounts {
    mul: usize,
    add_sub: usize,
    unary: usize,
    div: usize,
}

impl EventCounts {
    fn for_shape(shape: Shape) -> Self {
        shape.validate();
        Self {
            mul: shape.sequence_length * 2 * shape.hidden_size,
            add_sub: shape.sequence_length * (4 * shape.hidden_size - 1),
            unary: shape.sequence_length * (shape.hidden_size + 1),
            div: shape.sequence_length * 2,
        }
    }

    fn total(self) -> usize {
        self.mul + self.add_sub + self.unary + self.div
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    shape: Shape,
    layer: usize,
    synthetic: bool,
    c_proj_dir: PathBuf,
    join_dir: PathBuf,
    data_dir: PathBuf,
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct UpstreamCommitments {
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Debug)]
struct Ln2Data {
    input: Vec<u16>,
    weight: Vec<u16>,
    bias: Vec<u16>,
    upstream: UpstreamCommitments,
}

#[derive(Clone, Copy, Debug)]
struct Ln2Commitments {
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Debug, Default)]
struct Report {
    proof_path: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    build_seconds: f64,
    load_seconds: f64,
    upstream_verify_seconds: f64,
    reference_seconds: f64,
    compile_seconds: f64,
    execute_seconds: f64,
    setup_seconds: Option<f64>,
    prove_seconds: Option<f64>,
    verify_seconds: Option<f64>,
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
        .join("sp1-models/gpt2-bf16/recursion/zkgpt-like-12x30-real-bf16")
}

fn parse_usize(argument: Option<std::ffi::OsString>, option: &str) -> usize {
    argument
        .unwrap_or_else(|| panic!("{option} requires a value"))
        .to_str()
        .unwrap_or_else(|| panic!("{option} must be valid UTF-8"))
        .parse()
        .unwrap_or_else(|error| panic!("invalid {option}: {error}"))
}

fn parse_arguments() -> Arguments {
    let mut mode = Mode::Estimate;
    let mut shape = Shape::full();
    let mut layer = DEFAULT_LAYER;
    let mut synthetic = false;
    let mut c_proj_dir = std::env::var_os("SP1_ZKGPT_C_PROJ_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-c-proj-output"));
    let mut join_dir = std::env::var_os("SP1_ZKGPT_C_PROJ_JOIN_DIR").map(PathBuf::from);
    let mut data_dir = std::env::var_os("SP1_ZKGPT_LIKE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    let mut output_dir = std::env::var_os("SP1_ZKGPT_LN2_OUTPUT_DIR").map(PathBuf::from);
    let mut arguments = std::env::args_os().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--estimate") => mode = Mode::Estimate,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--small") => {
                shape = Shape::small();
                synthetic = true;
            }
            Some("--synthetic") => synthetic = true,
            Some("--layer") => layer = parse_usize(arguments.next(), "--layer"),
            Some("--c-proj-dir") => {
                c_proj_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--c-proj-dir requires a value")),
                );
            }
            Some("--join-dir") => {
                join_dir = Some(PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--join-dir requires a value")),
                ));
            }
            Some("--data-dir") => {
                data_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--data-dir requires a value")),
                );
                synthetic = false;
            }
            Some("--output-dir") => {
                output_dir = Some(PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--output-dir requires a value")),
                ));
            }
            Some(value) => panic!("unknown option: {value}"),
            None => panic!("command-line options must be valid UTF-8"),
        }
    }

    shape.validate();
    if !synthetic {
        assert_eq!(
            (shape.sequence_length, shape.hidden_size),
            (DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE),
            "real LN2 data uses the fixed 30 x 768 shape"
        );
    }
    let join_dir = join_dir.unwrap_or_else(|| c_proj_dir.clone());
    let output_dir =
        output_dir.unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-ln2-output"));
    Arguments { mode, shape, layer, synthetic, c_proj_dir, join_dir, data_dir, output_dir }
}

fn parse_digest(value: &str) -> Digest {
    let limbs = value
        .split(':')
        .map(|limb| {
            let limb = limb.strip_prefix("0x").unwrap_or(limb);
            u32::from_str_radix(limb, 16)
                .unwrap_or_else(|error| panic!("invalid digest limb {limb}: {error}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(limbs.len(), DIGEST_SIZE, "a digest requires {DIGEST_SIZE} limbs");
    limbs.into_iter().map(SP1Field::from_canonical_u32).collect::<Vec<_>>().try_into().unwrap()
}

fn digest_hex(digest: &Digest) -> String {
    digest
        .iter()
        .map(|value| format!("{:08X}", value.as_canonical_u32()))
        .collect::<Vec<_>>()
        .join(":")
}

fn read_bf16_binary(path: &Path) -> Vec<u16> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 2, 0, "{} has an odd byte length", path.display());
    bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect()
}

fn json_string_field(path: &Path, key: &str) -> String {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let prefix = format!("\"{key}\":");
    let line = contents
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("{} is missing {key}", path.display()));
    line[prefix.len()..].trim().trim_end_matches(',').trim_matches('"').to_owned()
}

fn host_commit_fields(domain: u32, values: &[SP1Field]) -> Digest {
    let mut state = [SP1Field::zero(); PERMUTATION_WIDTH];
    state[0] = SP1Field::from_canonical_u32(domain);
    state[1] = SP1Field::from_canonical_usize(values.len());
    inner_perm().permute_mut(&mut state);
    for chunk in values.chunks(HASH_RATE) {
        state[..HASH_RATE].fill(SP1Field::zero());
        state[..chunk.len()].copy_from_slice(chunk);
        inner_perm().permute_mut(&mut state);
    }
    state[..DIGEST_SIZE].try_into().unwrap()
}

fn host_commit_u16(domain: u32, values: &[u16]) -> Digest {
    let values =
        values.iter().map(|&value| SP1Field::from_canonical_u16(value)).collect::<Vec<_>>();
    host_commit_fields(domain, &values)
}

fn verify_upstream_join(arguments: &Arguments) -> (UpstreamCommitments, f64) {
    type A = RecursionAir<SP1Field, 3, 2>;

    let manifest_path =
        arguments.join_dir.join(format!("zkgpt_c_proj_join_l{:02}.manifest.json", arguments.layer));
    let upstream = UpstreamCommitments {
        output: parse_digest(&json_string_field(&manifest_path, "output_commitment")),
        transcript: parse_digest(&json_string_field(&manifest_path, "transcript_commitment")),
    };
    let verifier = ShardVerifier::from_basefold_parameters(
        FriConfig::new(
            PROOF_LOG_BLOWUP,
            unique_decoding_queries(PROOF_LOG_BLOWUP),
            SP1_PROOF_OF_WORK_BITS,
        ),
        UPSTREAM_JOIN_MAX_LOG_ROWS as u32,
        UPSTREAM_JOIN_MAX_LOG_ROWS,
        A::verillm_machine(),
    );
    let prover = simple_prover(verifier);
    let vk_path =
        arguments.join_dir.join(format!("zkgpt_c_proj_join_l{:02}.vk.bin", arguments.layer));
    let proof_path =
        arguments.join_dir.join(format!("zkgpt_c_proj_join_l{:02}.proof.bin", arguments.layer));
    let vk: StoredVerifyingKey = bincode::deserialize_from(
        File::open(&vk_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", vk_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", vk_path.display()));
    let proof: StoredProof = bincode::deserialize_from(
        File::open(&proof_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", proof_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", proof_path.display()));
    assert_eq!(proof.shard_proofs.len(), 1, "c_proj join proof must contain one shard");
    let public_values: &RecursionPublicValues<SP1Field> =
        proof.shard_proofs[0].public_values.as_slice().borrow();
    assert_eq!(
        public_values.digest, upstream.transcript,
        "c_proj join proof public digest differs from its manifest"
    );
    let started = Instant::now();
    prover.verify(&vk, &proof).expect("upstream c_proj join proof must verify");
    let seconds = started.elapsed().as_secs_f64();
    println!("upstream c_proj join proof verified: elapsed={seconds:.3}s");
    (upstream, seconds)
}

fn load_data(arguments: &Arguments) -> (Ln2Data, f64) {
    if arguments.synthetic {
        let input = (0..arguments.shape.sequence_length * arguments.shape.hidden_size)
            .map(|index| [0x3f80, 0x4000, 0x4040, 0xbf80][index % 4])
            .collect::<Vec<_>>();
        let upstream = UpstreamCommitments {
            output: host_commit_u16(DOMAIN_C_PROJ_GROUP_OUTPUT, &input),
            transcript: host_commit_u16(0x13ff, &[arguments.layer as u16]),
        };
        return (
            Ln2Data {
                input,
                weight: vec![0x3f80; arguments.shape.hidden_size],
                bias: vec![0; arguments.shape.hidden_size],
                upstream,
            },
            0.0,
        );
    }

    let (upstream, upstream_verify_seconds) = verify_upstream_join(arguments);
    let input_path = arguments
        .c_proj_dir
        .join(format!("zkgpt_c_proj_l{:02}.output.private.bf16.bin", arguments.layer));
    let input = read_bf16_binary(&input_path);
    assert_eq!(input.len(), arguments.shape.sequence_length * arguments.shape.hidden_size);
    let input_digest = host_commit_u16(DOMAIN_C_PROJ_GROUP_OUTPUT, &input);
    assert_eq!(
        input_digest, upstream.output,
        "private c_proj output does not match the join proof commitment"
    );

    let layer_dir = arguments.data_dir.join(format!("layer-{:02}", arguments.layer));
    let weight = read_bf16_binary(&layer_dir.join("ln_2_weight.bf16.bin"));
    let bias = read_bf16_binary(&layer_dir.join("ln_2_bias.bf16.bin"));
    assert_eq!(weight.len(), arguments.shape.hidden_size);
    assert_eq!(bias.len(), arguments.shape.hidden_size);
    (Ln2Data { input, weight, bias, upstream }, upstream_verify_seconds)
}

fn usize_to_bf16_raw(value: usize) -> u16 {
    assert!(value > 0, "BF16 integer conversion requires a positive value");
    let exponent = value.ilog2();
    let mantissa = if exponent <= 7 { value << (7 - exponent) } else { value >> (exponent - 7) };
    debug_assert!((128..256).contains(&mantissa));
    (((exponent + 127) as u16) << 7) | (mantissa as u16 - 128)
}

fn reference_add(lhs: u16, rhs: u16) -> u16 {
    Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Add).output.raw
}

fn reference_sub(lhs: u16, rhs: u16) -> u16 {
    Bf16AddSubWitness::new(lhs, rhs, Bf16AddSubOpcode::Sub).output.raw
}

fn reference_mean(values: &[u16]) -> u16 {
    let sum = values[1..].iter().fold(values[0], |sum, &value| reference_add(sum, value));
    Bf16DivWitness::new(sum, usize_to_bf16_raw(values.len())).output.raw
}

fn reference_layer_norm_row(
    values: &[u16],
    weight: &[u16],
    bias: &[u16],
    epsilon: u16,
) -> Vec<u16> {
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
        .zip(weight.iter().copied())
        .zip(bias.iter().copied())
        .map(|((value, weight), bias)| {
            let normalized = Bf16MulWitness::new(value, inverse_standard_deviation).output.raw;
            let scaled = Bf16MulWitness::new(normalized, weight).output.raw;
            reference_add(scaled, bias)
        })
        .collect()
}

fn reference_ln2(shape: Shape, data: &Ln2Data) -> Vec<u16> {
    let mut output = Vec::with_capacity(data.input.len());
    for row in data.input.chunks_exact(shape.hidden_size) {
        output.extend(reference_layer_norm_row(
            row,
            &data.weight,
            &data.bias,
            GPT2_LAYER_NORM_EPSILON,
        ));
    }
    output
}

fn transcript_fields_host(
    shape: Shape,
    layer: usize,
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
) -> Vec<SP1Field> {
    let mut fields = [
        PROTOCOL_VERSION,
        LN2_STAGE,
        layer as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(upstream);
    fields.extend(input);
    fields.extend(parameters);
    fields.extend(output);
    fields
}

fn compute_commitments(
    shape: Shape,
    layer: usize,
    data: &Ln2Data,
    output: &[u16],
) -> Ln2Commitments {
    let input = host_commit_u16(DOMAIN_C_PROJ_GROUP_OUTPUT, &data.input);
    assert_eq!(input, data.upstream.output, "LN2 input differs from c_proj join output");
    let mut parameter_values = Vec::with_capacity(data.weight.len() + data.bias.len());
    parameter_values.extend_from_slice(&data.weight);
    parameter_values.extend_from_slice(&data.bias);
    let parameters = host_commit_u16(DOMAIN_LN2_PARAMETERS, &parameter_values);
    let output = host_commit_u16(DOMAIN_LN2_OUTPUT, output);
    let transcript = host_commit_fields(
        DOMAIN_LN2_TRANSCRIPT,
        &transcript_fields_host(shape, layer, data.upstream.transcript, input, parameters, output),
    );
    Ln2Commitments { upstream: data.upstream.transcript, input, parameters, output, transcript }
}

fn circuit_commit_fields(
    builder: &mut Ln2Builder,
    domain: u32,
    values: &[Felt<SP1Field>],
) -> [Felt<SP1Field>; DIGEST_SIZE] {
    let zero = builder.constant(SP1Field::zero());
    let mut state = [zero; PERMUTATION_WIDTH];
    state[0] = builder.constant(SP1Field::from_canonical_u32(domain));
    state[1] = builder.constant(SP1Field::from_canonical_usize(values.len()));
    state = builder.poseidon2_permute_v2(state);
    for chunk in values.chunks(HASH_RATE) {
        state[..HASH_RATE].fill(zero);
        state[..chunk.len()].copy_from_slice(chunk);
        state = builder.poseidon2_permute_v2(state);
    }
    state[..DIGEST_SIZE].try_into().unwrap()
}

fn build_ln2(builder: &mut Ln2Builder, shape: Shape) {
    let layer = builder.hint_felts_v2(1)[0];
    let upstream: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let expected_input: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let input = builder.hint_felts_v2(shape.sequence_length * shape.hidden_size);
    let weight = builder.hint_felts_v2(shape.hidden_size);
    let bias = builder.hint_felts_v2(shape.hidden_size);

    let input_digest = circuit_commit_fields(builder, DOMAIN_C_PROJ_GROUP_OUTPUT, &input);
    for (&computed, &expected) in input_digest.iter().zip(&expected_input) {
        builder.assert_felt_eq(computed, expected);
    }
    let mut parameter_values = Vec::with_capacity(weight.len() + bias.len());
    parameter_values.extend_from_slice(&weight);
    parameter_values.extend_from_slice(&bias);
    let parameters = circuit_commit_fields(builder, DOMAIN_LN2_PARAMETERS, &parameter_values);
    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let output = builder.bf16_layer_norm_rows(&input, shape.hidden_size, &weight, &bias, epsilon);
    let output_digest = circuit_commit_fields(builder, DOMAIN_LN2_OUTPUT, &output);

    let constants = [
        PROTOCOL_VERSION,
        LN2_STAGE,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut transcript_fields = vec![constants[0], constants[1], layer];
    transcript_fields.extend_from_slice(&constants[2..]);
    transcript_fields.extend(upstream);
    transcript_fields.extend(expected_input);
    transcript_fields.extend(parameters);
    transcript_fields.extend(output_digest);
    let transcript = circuit_commit_fields(builder, DOMAIN_LN2_TRANSCRIPT, &transcript_fields);

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = transcript;
    builder.commit_public_values_v2(*public_values);
}

fn witness_stream(layer: usize, data: &Ln2Data) -> Vec<sp1_recursion_executor::Block<SP1Field>> {
    let mut values = Vec::with_capacity(
        1 + 2 * DIGEST_SIZE + data.input.len() + data.weight.len() + data.bias.len(),
    );
    values.push(SP1Field::from_canonical_usize(layer));
    values.extend(data.upstream.transcript);
    values.extend(data.upstream.output);
    values.extend(data.input.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.weight.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.bias.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.into_iter().map(Into::into).collect()
}

fn padded_rows(events: usize, lanes: usize) -> usize {
    events.div_ceil(lanes).next_multiple_of(32)
}

fn print_estimate(shape: Shape, events: EventCounts) {
    let max_rows = 1usize << shape.max_log_rows();
    let event_bytes = events.mul * std::mem::size_of::<Bf16MulEvent<SP1Field>>()
        + events.add_sub * std::mem::size_of::<Bf16AddSubEvent<SP1Field>>()
        + events.unary * std::mem::size_of::<Bf16UnaryEvent<SP1Field>>()
        + events.div * std::mem::size_of::<Bf16DivEvent<SP1Field>>();
    println!(
        "LN2 shape: seq_len={} hidden={} output_values={}",
        shape.sequence_length,
        shape.hidden_size,
        shape.sequence_length * shape.hidden_size
    );
    println!(
        "events: mul={} add_sub={} unary={} div={} total={}",
        events.mul,
        events.add_sub,
        events.unary,
        events.div,
        events.total()
    );
    println!(
        "12-lane rows: mul={} add_sub={} max_2^{}={max_rows}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW),
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW),
        shape.max_log_rows()
    );
    println!(
        "executor-record lower bound: bytes={event_bytes} mib={:.2}",
        event_bytes as f64 / 1024_f64.powi(2)
    );
}

fn print_trace_rows(record: &ExecutionRecord<SP1Field>, max_log_rows: usize) {
    type A = RecursionAir<SP1Field, 3, 2>;
    let machine = A::verillm_machine();
    let max_rows = 1usize << max_log_rows;
    for chip in machine.chips() {
        if chip.included(record) {
            let rows = chip.num_rows(record).unwrap_or_default();
            println!("trace: {:<22} rows={rows}", chip.name());
            assert!(
                rows <= max_rows,
                "{} needs {rows} rows, exceeding LN2 maximum {max_rows}",
                chip.name()
            );
        }
    }
}

fn optional_seconds(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| format!("{value:.6}"))
}

fn optional_path(path: Option<&Path>) -> String {
    path.map_or_else(
        || "null".to_owned(),
        |path| format!("\"{}\"", path.file_name().unwrap().to_string_lossy()),
    )
}

fn optional_bytes(path: Option<&Path>) -> String {
    path.map_or_else(|| "null".to_owned(), |path| fs::metadata(path).unwrap().len().to_string())
}

fn write_artifacts(
    arguments: &Arguments,
    output: &[u16],
    commitments: Ln2Commitments,
    report: &Report,
) {
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_ln2_l{:02}", arguments.layer);
    let output_file = format!("{stem}.output.private.bf16.bin");
    let output_path = arguments.output_dir.join(&output_file);
    let output_bytes = output.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    fs::write(&output_path, output_bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));

    let manifest = format!(
        concat!(
            "{{\n",
            "  \"version\": {protocol_version},\n",
            "  \"stage\": \"ln_2\",\n",
            "  \"layer\": {},\n",
            "  \"sequence_length\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"epsilon_bf16_raw\": {},\n",
            "  \"upstream_transcript\": \"{}\",\n",
            "  \"input_commitment\": \"{}\",\n",
            "  \"parameters_commitment\": \"{}\",\n",
            "  \"output_commitment\": \"{}\",\n",
            "  \"transcript_commitment\": \"{}\",\n",
            "  \"private_output_file\": \"{}\",\n",
            "  \"output_values\": {},\n",
            "  \"build_seconds\": {build_seconds:.6},\n",
            "  \"load_seconds\": {load_seconds:.6},\n",
            "  \"upstream_verify_seconds\": {upstream_verify_seconds:.6},\n",
            "  \"reference_seconds\": {reference_seconds:.6},\n",
            "  \"compile_seconds\": {compile_seconds:.6},\n",
            "  \"execute_seconds\": {execute_seconds:.6},\n",
            "  \"setup_seconds\": {},\n",
            "  \"prove_seconds\": {},\n",
            "  \"verify_seconds\": {},\n",
            "  \"proof_file\": {},\n",
            "  \"proof_bytes\": {},\n",
            "  \"verifying_key_file\": {},\n",
            "  \"verifying_key_bytes\": {}\n",
            "}}\n"
        ),
        arguments.layer,
        arguments.shape.sequence_length,
        arguments.shape.hidden_size,
        GPT2_LAYER_NORM_EPSILON,
        digest_hex(&commitments.upstream),
        digest_hex(&commitments.input),
        digest_hex(&commitments.parameters),
        digest_hex(&commitments.output),
        digest_hex(&commitments.transcript),
        output_file,
        output.len(),
        optional_seconds(report.setup_seconds),
        optional_seconds(report.prove_seconds),
        optional_seconds(report.verify_seconds),
        optional_path(report.proof_path.as_deref()),
        optional_bytes(report.proof_path.as_deref()),
        optional_path(report.vk_path.as_deref()),
        optional_bytes(report.vk_path.as_deref()),
        protocol_version = PROTOCOL_VERSION,
        build_seconds = report.build_seconds,
        load_seconds = report.load_seconds,
        upstream_verify_seconds = report.upstream_verify_seconds,
        reference_seconds = report.reference_seconds,
        compile_seconds = report.compile_seconds,
        execute_seconds = report.execute_seconds,
    );
    let manifest_path = arguments.output_dir.join(format!("{stem}.manifest.json"));
    fs::write(&manifest_path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
    println!("LN2 private output: {}", output_path.display());
    println!("LN2 manifest: {}", manifest_path.display());
}

async fn prove_ln2(
    program: Arc<RecursionProgram<SP1Field>>,
    record: ExecutionRecord<SP1Field>,
    arguments: &Arguments,
) -> (PathBuf, PathBuf, f64, f64, f64) {
    type A = RecursionAir<SP1Field, 3, 2>;
    let max_log_rows = arguments.shape.max_log_rows();
    let verifier = ShardVerifier::from_basefold_parameters(
        FriConfig::new(
            PROOF_LOG_BLOWUP,
            unique_decoding_queries(PROOF_LOG_BLOWUP),
            SP1_PROOF_OF_WORK_BITS,
        ),
        max_log_rows as u32,
        max_log_rows,
        A::verillm_machine(),
    );
    let prover = simple_prover(verifier);
    let proof_shape = prover.shape_from_record(&record).expect("LN2 machine has no proof shape");
    println!("proof shape: {proof_shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    println!("LN2 setup: elapsed={setup_seconds:.3}s");
    let pk = unsafe { pk.into_inner() };

    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    let prove_seconds = prove_started.elapsed().as_secs_f64();
    println!("LN2 proof generated: elapsed={prove_seconds:.3}s");

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated LN2 proof must verify");
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    println!("LN2 proof verified: elapsed={verify_seconds:.3}s");

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_ln2_l{:02}", arguments.layer);
    let proof_path = arguments.output_dir.join(format!("{stem}.proof.bin"));
    let vk_path = arguments.output_dir.join(format!("{stem}.vk.bin"));
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    println!(
        "LN2 proof artifacts: proof={} ({} bytes) vk={} ({} bytes)",
        proof_path.display(),
        fs::metadata(&proof_path).unwrap().len(),
        vk_path.display(),
        fs::metadata(&vk_path).unwrap().len()
    );
    (proof_path, vk_path, setup_seconds, prove_seconds, verify_seconds)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let events = EventCounts::for_shape(arguments.shape);
    println!(
        "mode={:?} layer={} synthetic={} c_proj_dir={} join_dir={}",
        arguments.mode,
        arguments.layer,
        arguments.synthetic,
        arguments.c_proj_dir.display(),
        arguments.join_dir.display()
    );
    print_estimate(arguments.shape, events);
    if arguments.mode == Mode::Estimate {
        return;
    }

    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: Ln2Builder = AsmBuilder::default();
    build_ln2(&mut builder, arguments.shape);
    let block = builder.into_root_block();
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("built LN2: ir_ops={} elapsed={build_seconds:.3}s", block.ops.len());
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let load_started = Instant::now();
    let (data, upstream_verify_seconds) = load_data(&arguments);
    let load_seconds = load_started.elapsed().as_secs_f64();
    println!(
        "loaded LN2 data: input={} weight={} bias={} elapsed={load_seconds:.3}s",
        data.input.len(),
        data.weight.len(),
        data.bias.len()
    );

    let reference_started = Instant::now();
    let output = reference_ln2(arguments.shape, &data);
    let commitments = compute_commitments(arguments.shape, arguments.layer, &data, &output);
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    println!("LN2 input commitment: {}", digest_hex(&commitments.input));
    println!("LN2 output commitment: {}", digest_hex(&commitments.output));
    println!("LN2 transcript: {}", digest_hex(&commitments.transcript));
    println!("host BF16 reference: elapsed={reference_seconds:.3}s");

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    println!("compiled LN2: elapsed={compile_seconds:.3}s");

    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.witness_stream = witness_stream(arguments.layer, &data).into();
    let execute_started = Instant::now();
    executor.run().expect("valid LN2 witness must execute");
    let execute_seconds = execute_started.elapsed().as_secs_f64();
    println!("executed LN2: elapsed={execute_seconds:.3}s");
    assert_eq!(executor.record.bf16_mul_events.len(), events.mul);
    assert_eq!(executor.record.bf16_add_sub_events.len(), events.add_sub);
    assert_eq!(executor.record.bf16_unary_events.len(), events.unary);
    assert_eq!(executor.record.bf16_div_events.len(), events.div);
    assert_eq!(
        executor.record.public_values.digest, commitments.transcript,
        "LN2 circuit transcript differs from host transcript"
    );
    println!("LN2 circuit output matches the independent BF16 reference commitment");
    print_trace_rows(&executor.record, arguments.shape.max_log_rows());

    let mut report = Report {
        build_seconds,
        load_seconds,
        upstream_verify_seconds,
        reference_seconds,
        compile_seconds,
        execute_seconds,
        ..Report::default()
    };
    if arguments.mode == Mode::Prove {
        let record = std::mem::take(&mut executor.record);
        drop(executor);
        let result = prove_ln2(program, record, &arguments).await;
        report.proof_path = Some(result.0);
        report.vk_path = Some(result.1);
        report.setup_seconds = Some(result.2);
        report.prove_seconds = Some(result.3);
        report.verify_seconds = Some(result.4);
    }
    write_artifacts(&arguments, &output, commitments, &report);
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(shape: Shape) -> Ln2Data {
        let input = (0..shape.sequence_length * shape.hidden_size)
            .map(|index| [0x3f80, 0x4000, 0x4040, 0xbf80][index % 4])
            .collect::<Vec<_>>();
        Ln2Data {
            upstream: UpstreamCommitments {
                output: host_commit_u16(DOMAIN_C_PROJ_GROUP_OUTPUT, &input),
                transcript: host_commit_u16(0x13ff, &[0]),
            },
            input,
            weight: vec![0x3f80; shape.hidden_size],
            bias: vec![0; shape.hidden_size],
        }
    }

    #[test]
    fn ln2_event_count_matches_formula() {
        let events = EventCounts::for_shape(Shape::full());
        assert_eq!(events.mul, 46_080);
        assert_eq!(events.add_sub, 92_130);
        assert_eq!(events.unary, 23_070);
        assert_eq!(events.div, 60);
        assert_eq!(events.total(), 161_340);
    }

    #[test]
    fn ln2_circuit_matches_reference_and_rejects_wrong_upstream_input() {
        let shape = Shape::small();
        let valid = synthetic_data(shape);
        let output = reference_ln2(shape, &valid);
        let expected = compute_commitments(shape, DEFAULT_LAYER, &valid, &output);

        let mut builder: Ln2Builder = AsmBuilder::default();
        build_ln2(&mut builder, shape);
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(
            compiler.compile_inner(builder.into_root_block()).validate().expect("valid LN2 IR"),
        );
        let execute = |data: &Ln2Data| {
            let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
                program.clone(),
                inner_perm(),
            );
            executor.witness_stream = witness_stream(DEFAULT_LAYER, data).into();
            let result = executor.run();
            (result.is_ok(), executor.record.public_values.digest)
        };

        let (succeeded, digest) = execute(&valid);
        assert!(succeeded);
        assert_eq!(digest, expected.transcript);

        let mut modified_input = valid.clone();
        modified_input.input[0] ^= 1;
        assert!(!execute(&modified_input).0, "input not committed by c_proj join must fail");
    }
}
