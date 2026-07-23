#![allow(clippy::print_stdout)]

use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Write as _,
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
    Bf16AddSubEvent, Bf16AddSubOpcode, Bf16AddSubWitness, Bf16MulEvent, Bf16MulWitness,
    ExecutionRecord, Executor, RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE,
    PERMUTATION_WIDTH, RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::{
    chips::bf16::{NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW},
    sharding::{plan_linear_output_columns, ShardLimits, DEFAULT_MAX_TRACE_AREA},
    RecursionAir,
};

const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_LAYER: usize = 0;
const DEFAULT_TILE: usize = 0;

const PROTOCOL_VERSION: u32 = 1;
const C_PROJ_TILE_STAGE: u32 = 3;
const C_PROJ_GROUP_STAGE: u32 = 4;
const DOMAIN_ATTENTION_GROUP_OUTPUT: u32 = 0x1103;
const DOMAIN_C_PROJ_PARAMETERS: u32 = 0x1201;
const DOMAIN_C_PROJ_TILE_OUTPUT: u32 = 0x1202;
const DOMAIN_C_PROJ_TILE_TRANSCRIPT: u32 = 0x1203;
const DOMAIN_C_PROJ_GROUP_PARAMETERS: u32 = 0x1211;
const DOMAIN_C_PROJ_GROUP_OUTPUT: u32 = 0x1212;
const DOMAIN_C_PROJ_GROUP_TRANSCRIPT: u32 = 0x1213;

const PROOF_LOG_BLOWUP: usize = 1;
const FULL_MAX_LOG_ROWS: usize = 19;
const SMALL_MAX_LOG_ROWS: usize = 16;
const UPSTREAM_JOIN_MAX_LOG_ROWS: usize = 16;

type TileBuilder = Builder<AsmConfig>;
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
    num_tiles: usize,
}

impl Shape {
    fn full() -> Self {
        let plan = plan_linear_output_columns(
            DEFAULT_SEQUENCE_LENGTH,
            DEFAULT_HIDDEN_SIZE,
            DEFAULT_HIDDEN_SIZE,
            0,
            ShardLimits::full(),
        );
        Self {
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            num_tiles: plan.shard_count,
        }
    }

    fn small() -> Self {
        Self { sequence_length: 2, hidden_size: 8, num_tiles: 2 }
    }

    fn validate(self) {
        assert!(self.sequence_length > 0, "c_proj sequence length must be nonzero");
        assert!(self.hidden_size > 0, "c_proj hidden size must be nonzero");
        assert!(self.num_tiles > 0, "c_proj tile count must be nonzero");
        assert_eq!(
            self.hidden_size % self.num_tiles,
            0,
            "c_proj hidden size must be divisible by tile count"
        );
    }

    fn tile_width(self) -> usize {
        self.hidden_size / self.num_tiles
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

#[derive(Clone, Copy, Debug)]
struct EventCounts {
    mul: usize,
    add_sub: usize,
}

impl EventCounts {
    fn for_shape(shape: Shape) -> Self {
        let outputs = shape.sequence_length * shape.tile_width();
        Self { mul: outputs * shape.hidden_size, add_sub: outputs * (shape.hidden_size - 1) }
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    shape: Shape,
    layer: usize,
    tile: usize,
    all_tiles: bool,
    synthetic: bool,
    attention_dir: PathBuf,
    join_dir: PathBuf,
    data_dir: PathBuf,
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct UpstreamCommitments {
    output: Digest,
    transcript: Digest,
}

#[derive(Debug)]
struct SourceData {
    input: Vec<u16>,
    full_weight: Vec<u16>,
    upstream: UpstreamCommitments,
}

#[derive(Debug)]
struct TileData {
    input: Vec<u16>,
    weight: Vec<u16>,
    upstream: UpstreamCommitments,
}

#[derive(Clone, Copy, Debug)]
struct TileCommitments {
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
    transcript: Digest,
}

struct TileExecution {
    tile: usize,
    output: Vec<u16>,
    commitments: TileCommitments,
    record: ExecutionRecord<SP1Field>,
    reference_seconds: f64,
    execute_seconds: f64,
}

#[derive(Debug)]
struct TileSummary {
    tile: usize,
    commitments: TileCommitments,
    reference_seconds: f64,
    execute_seconds: f64,
    prove_seconds: Option<f64>,
    verify_seconds: Option<f64>,
    proof_file: Option<String>,
    proof_bytes: Option<u64>,
}

#[derive(Debug)]
struct CompletedTile {
    output: Vec<u16>,
    summary: TileSummary,
}

#[derive(Clone, Copy, Debug)]
struct GroupCommitments {
    input: Digest,
    parameters: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Debug, Default)]
struct BatchMetrics {
    build_seconds: f64,
    compile_seconds: f64,
    setup_seconds: Option<f64>,
    batch_seconds: f64,
    verifying_key_file: Option<String>,
    verifying_key_bytes: Option<u64>,
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
    let mut tile = DEFAULT_TILE;
    let mut tile_was_set = false;
    let mut all_tiles = false;
    let mut synthetic = false;
    let mut attention_dir = std::env::var_os("SP1_ZKGPT_LEAF_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-leaf-output"));
    let mut join_dir = std::env::var_os("SP1_ZKGPT_ATTENTION_JOIN_DIR").map(PathBuf::from);
    let mut data_dir = std::env::var_os("SP1_ZKGPT_LIKE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    let mut output_dir = std::env::var_os("SP1_ZKGPT_C_PROJ_OUTPUT_DIR").map(PathBuf::from);
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
            Some("--layer") => layer = parse_usize(arguments.next(), "--layer"),
            Some("--tile") => {
                tile = parse_usize(arguments.next(), "--tile");
                tile_was_set = true;
            }
            Some("--all-tiles") => all_tiles = true,
            Some("--attention-dir") => {
                attention_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--attention-dir requires a value")),
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
    assert!(!(all_tiles && tile_was_set), "--all-tiles cannot be combined with --tile");
    assert!(tile < shape.num_tiles, "tile index {tile} is out of range");
    if !synthetic {
        let full = Shape::full();
        assert_eq!(
            (shape.sequence_length, shape.hidden_size, shape.num_tiles),
            (DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE, full.num_tiles),
            "real c_proj data uses the shape-aware full 30 x 768 plan"
        );
    }
    let join_dir = join_dir.unwrap_or_else(|| attention_dir.clone());
    let output_dir =
        output_dir.unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-c-proj-output"));
    Arguments {
        mode,
        shape,
        layer,
        tile,
        all_tiles,
        synthetic,
        attention_dir,
        join_dir,
        data_dir,
        output_dir,
    }
}

fn selected_tiles(arguments: &Arguments) -> Vec<usize> {
    if arguments.all_tiles {
        (0..arguments.shape.num_tiles).collect()
    } else {
        vec![arguments.tile]
    }
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

fn extract_weight_tile(full_weight: &[u16], shape: Shape, tile: usize) -> Vec<u16> {
    assert_eq!(full_weight.len(), shape.hidden_size * shape.hidden_size);
    assert!(tile < shape.num_tiles);
    let width = shape.tile_width();
    let start = tile * width;
    let mut weight = Vec::with_capacity(shape.hidden_size * width);
    for row in full_weight.chunks_exact(shape.hidden_size) {
        weight.extend_from_slice(&row[start..start + width]);
    }
    weight
}

fn reference_linear_rows_no_bias(input: &[u16], input_features: usize, weight: &[u16]) -> Vec<u16> {
    let output_features = weight.len() / input_features;
    let mut output = Vec::with_capacity(input.len() / input_features * output_features);
    for input_row in input.chunks_exact(input_features) {
        for output_index in 0..output_features {
            let mut sum = Bf16MulWitness::new(input_row[0], weight[output_index]).output.raw;
            for input_index in 1..input_features {
                let product = Bf16MulWitness::new(
                    input_row[input_index],
                    weight[input_index * output_features + output_index],
                )
                .output
                .raw;
                sum = Bf16AddSubWitness::new(sum, product, Bf16AddSubOpcode::Add).output.raw;
            }
            output.push(sum);
        }
    }
    output
}

fn verify_upstream_join(arguments: &Arguments) -> UpstreamCommitments {
    type A = RecursionAir<SP1Field, 3, 2>;

    let manifest_path = arguments
        .join_dir
        .join(format!("zkgpt_attention_join_l{:02}.manifest.json", arguments.layer));
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
        arguments.join_dir.join(format!("zkgpt_attention_join_l{:02}.vk.bin", arguments.layer));
    let proof_path =
        arguments.join_dir.join(format!("zkgpt_attention_join_l{:02}.proof.bin", arguments.layer));
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
    assert_eq!(proof.shard_proofs.len(), 1, "attention join proof must contain one shard");
    let public_values: &RecursionPublicValues<SP1Field> =
        proof.shard_proofs[0].public_values.as_slice().borrow();
    assert_eq!(
        public_values.digest, upstream.transcript,
        "attention join proof public digest differs from its manifest"
    );
    prover.verify(&vk, &proof).expect("upstream attention join proof must verify");
    println!("upstream attention join proof verified");
    upstream
}

fn load_source_data(arguments: &Arguments) -> SourceData {
    let upstream = verify_upstream_join(arguments);
    let input_path = arguments
        .attention_dir
        .join(format!("zkgpt_attention_l{:02}.output.private.bf16.bin", arguments.layer));
    let input = read_bf16_binary(&input_path);
    assert_eq!(input.len(), arguments.shape.sequence_length * arguments.shape.hidden_size);
    let input_digest = host_commit_u16(DOMAIN_ATTENTION_GROUP_OUTPUT, &input);
    assert_eq!(
        input_digest, upstream.output,
        "private attention output does not match the join proof commitment"
    );

    let full_weight = if arguments.synthetic {
        vec![0x3f80; arguments.shape.hidden_size * arguments.shape.hidden_size]
    } else {
        let path = arguments
            .data_dir
            .join(format!("layer-{:02}/attention_projection_weight.bf16.bin", arguments.layer));
        read_bf16_binary(&path)
    };
    assert_eq!(full_weight.len(), arguments.shape.hidden_size * arguments.shape.hidden_size);
    SourceData { input, full_weight, upstream }
}

fn tile_data(source: &SourceData, shape: Shape, tile: usize) -> TileData {
    TileData {
        input: source.input.clone(),
        weight: extract_weight_tile(&source.full_weight, shape, tile),
        upstream: source.upstream,
    }
}

fn tile_transcript_fields_host(
    shape: Shape,
    layer: usize,
    tile: usize,
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
) -> Vec<SP1Field> {
    let mut fields = [
        PROTOCOL_VERSION,
        C_PROJ_TILE_STAGE,
        layer as u32,
        tile as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_tiles as u32,
        shape.tile_width() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(upstream);
    fields.extend(input);
    fields.extend(parameters);
    fields.extend(output);
    fields
}

fn compute_tile_commitments(
    shape: Shape,
    layer: usize,
    tile: usize,
    data: &TileData,
    output: &[u16],
) -> TileCommitments {
    let input = host_commit_u16(DOMAIN_ATTENTION_GROUP_OUTPUT, &data.input);
    assert_eq!(input, data.upstream.output, "c_proj input differs from attention join output");
    let parameters = host_commit_u16(DOMAIN_C_PROJ_PARAMETERS, &data.weight);
    let output = host_commit_u16(DOMAIN_C_PROJ_TILE_OUTPUT, output);
    let transcript = host_commit_fields(
        DOMAIN_C_PROJ_TILE_TRANSCRIPT,
        &tile_transcript_fields_host(
            shape,
            layer,
            tile,
            data.upstream.transcript,
            input,
            parameters,
            output,
        ),
    );
    TileCommitments { upstream: data.upstream.transcript, input, parameters, output, transcript }
}

fn circuit_commit_fields(
    builder: &mut TileBuilder,
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

fn build_tile(builder: &mut TileBuilder, shape: Shape) {
    let metadata = builder.hint_felts_v2(2);
    let upstream: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let expected_input: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let input = builder.hint_felts_v2(shape.sequence_length * shape.hidden_size);
    let weight = builder.hint_felts_v2(shape.hidden_size * shape.tile_width());

    let input_digest = circuit_commit_fields(builder, DOMAIN_ATTENTION_GROUP_OUTPUT, &input);
    for (&computed, &expected) in input_digest.iter().zip(&expected_input) {
        builder.assert_felt_eq(computed, expected);
    }
    let parameters = circuit_commit_fields(builder, DOMAIN_C_PROJ_PARAMETERS, &weight);
    let output = builder.bf16_linear_rows_no_bias(&input, shape.hidden_size, &weight);
    let output_digest = circuit_commit_fields(builder, DOMAIN_C_PROJ_TILE_OUTPUT, &output);

    let constants = [
        PROTOCOL_VERSION,
        C_PROJ_TILE_STAGE,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_tiles as u32,
        shape.tile_width() as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut transcript_fields = vec![constants[0], constants[1], metadata[0], metadata[1]];
    transcript_fields.extend_from_slice(&constants[2..]);
    transcript_fields.extend(upstream);
    transcript_fields.extend(expected_input);
    transcript_fields.extend(parameters);
    transcript_fields.extend(output_digest);
    let transcript =
        circuit_commit_fields(builder, DOMAIN_C_PROJ_TILE_TRANSCRIPT, &transcript_fields);

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = transcript;
    builder.commit_public_values_v2(*public_values);
}

fn witness_stream(
    arguments: &Arguments,
    tile: usize,
    data: &TileData,
) -> Vec<sp1_recursion_executor::Block<SP1Field>> {
    let mut values = Vec::with_capacity(2 + 2 * DIGEST_SIZE + data.input.len() + data.weight.len());
    values.push(SP1Field::from_canonical_usize(arguments.layer));
    values.push(SP1Field::from_canonical_usize(tile));
    values.extend(data.upstream.transcript);
    values.extend(data.upstream.output);
    values.extend(data.input.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.weight.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.into_iter().map(Into::into).collect()
}

fn padded_rows(events: usize, lanes: usize) -> usize {
    events.div_ceil(lanes).next_multiple_of(32)
}

fn print_estimate(shape: Shape, events: EventCounts) {
    let max_rows = 1usize << shape.max_log_rows();
    let plan = plan_linear_output_columns(
        shape.sequence_length,
        shape.hidden_size,
        shape.hidden_size,
        0,
        ShardLimits { max_log_rows: shape.max_log_rows(), ..ShardLimits::full() },
    );
    let event_bytes = events.mul * std::mem::size_of::<Bf16MulEvent<SP1Field>>()
        + events.add_sub * std::mem::size_of::<Bf16AddSubEvent<SP1Field>>();
    println!(
        "c_proj tile shape: seq_len={} input={} output={} tiles={}",
        shape.sequence_length,
        shape.hidden_size,
        shape.tile_width(),
        shape.num_tiles
    );
    println!("events per tile: mul={} add_sub={}", events.mul, events.add_sub);
    println!(
        "12-lane rows: mul={} add_sub={} max_2^{}={max_rows}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW),
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW),
        shape.max_log_rows()
    );
    println!(
        "executor-record lower bound per tile: bytes={event_bytes} gib={:.2}",
        event_bytes as f64 / 1024_f64.powi(3)
    );
    println!(
        "shape-aware plan: natural_unit=output_column units_per_leaf={} leaves={} estimated_rows={} estimated_trace_area={} limit={}",
        plan.units_per_shard,
        plan.shard_count,
        plan.estimate.max_rows(),
        plan.estimate.estimated_trace_area,
        plan.limits.max_trace_area,
    );
}

fn write_tile_manifest(arguments: &Arguments, tile: usize, commitments: TileCommitments) {
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let manifest = format!(
        "version={PROTOCOL_VERSION}\nstage=attention_c_proj_tile\nsharding=shape_aware_output_columns\nlayer={}\ntile={tile}\noutput_start={}\noutput_width={}\nupstream={}\ninput={}\nparameters={}\noutput={}\ntranscript={}\n",
        arguments.layer,
        tile * arguments.shape.tile_width(),
        arguments.shape.tile_width(),
        digest_hex(&commitments.upstream),
        digest_hex(&commitments.input),
        digest_hex(&commitments.parameters),
        digest_hex(&commitments.output),
        digest_hex(&commitments.transcript),
    );
    let path = arguments
        .output_dir
        .join(format!("zkgpt_c_proj_l{:02}_t{tile:02}.commitments.txt", arguments.layer));
    fs::write(&path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    println!("tile commitment manifest: {}", path.display());
}

fn execute_tile(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    events: EventCounts,
    source: &SourceData,
    tile: usize,
) -> TileExecution {
    let data = tile_data(source, arguments.shape, tile);
    let reference_started = Instant::now();
    let output =
        reference_linear_rows_no_bias(&data.input, arguments.shape.hidden_size, &data.weight);
    let commitments =
        compute_tile_commitments(arguments.shape, arguments.layer, tile, &data, &output);
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    println!(
        "host reference: tile={tile} output_values={} elapsed={reference_seconds:.3}s",
        output.len()
    );
    write_tile_manifest(arguments, tile, commitments);

    let mut executor =
        Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
    executor.witness_stream = witness_stream(arguments, tile, &data).into();
    let execute_started = Instant::now();
    executor.run().expect("valid c_proj tile witness must execute");
    let execute_seconds = execute_started.elapsed().as_secs_f64();
    println!("executed c_proj tile: tile={tile} elapsed={execute_seconds:.3}s");
    assert_eq!(executor.record.bf16_mul_events.len(), events.mul);
    assert_eq!(executor.record.bf16_add_sub_events.len(), events.add_sub);
    assert_eq!(
        executor.record.public_values.digest, commitments.transcript,
        "c_proj circuit transcript differs from host transcript"
    );
    let record = std::mem::take(&mut executor.record);
    drop(executor);
    TileExecution { tile, output, commitments, record, reference_seconds, execute_seconds }
}

fn execute_tiles(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    events: EventCounts,
    source: &SourceData,
    tiles: &[usize],
) -> (Vec<CompletedTile>, f64) {
    let started = Instant::now();
    let mut completed = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        let execution = execute_tile(program.clone(), arguments, events, source, tile);
        let TileExecution { tile, output, commitments, record, reference_seconds, execute_seconds } =
            execution;
        drop(record);
        completed.push(CompletedTile {
            output,
            summary: TileSummary {
                tile,
                commitments,
                reference_seconds,
                execute_seconds,
                prove_seconds: None,
                verify_seconds: None,
                proof_file: None,
                proof_bytes: None,
            },
        });
    }
    (completed, started.elapsed().as_secs_f64())
}

async fn prove_tiles(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    events: EventCounts,
    source: &SourceData,
    tiles: &[usize],
) -> (Vec<CompletedTile>, BatchMetrics) {
    type A = RecursionAir<SP1Field, 3, 2>;

    let max_log_rows = arguments.shape.max_log_rows();
    let max_rows = 1usize << max_log_rows;
    let machine = A::verillm_machine();
    let verifier = ShardVerifier::from_basefold_parameters(
        FriConfig::new(
            PROOF_LOG_BLOWUP,
            unique_decoding_queries(PROOF_LOG_BLOWUP),
            SP1_PROOF_OF_WORK_BITS,
        ),
        max_log_rows as u32,
        max_log_rows,
        machine,
    );
    let prover = simple_prover(verifier);
    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program.clone()).await;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    println!("c_proj shared setup: elapsed={setup_seconds:.3}s");
    let pk = unsafe { pk.into_inner() };

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let vk_stem = if arguments.all_tiles {
        format!("zkgpt_c_proj_l{:02}.shared", arguments.layer)
    } else {
        format!("zkgpt_c_proj_l{:02}_t{:02}", arguments.layer, tiles[0])
    };
    let vk_file = format!("{vk_stem}.vk.bin");
    let vk_path = arguments.output_dir.join(&vk_file);
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    let vk_bytes = fs::metadata(&vk_path).unwrap().len();
    println!("shared c_proj verifying key: {} ({vk_bytes} bytes)", vk_path.display());

    let batch_started = Instant::now();
    let mut completed = Vec::with_capacity(tiles.len());
    for (position, &tile) in tiles.iter().enumerate() {
        println!("c_proj tile {}/{}: tile={tile}", position + 1, tiles.len());
        let execution = execute_tile(program.clone(), arguments, events, source, tile);
        for chip in prover.machine().chips() {
            if chip.included(&execution.record) {
                let rows = chip.num_rows(&execution.record).unwrap_or_default();
                if position == 0 {
                    println!("trace: {:<22} rows={rows}", chip.name());
                }
                assert!(
                    rows <= max_rows,
                    "{} needs {rows} rows, exceeding tile maximum {max_rows}",
                    chip.name()
                );
            }
        }
        let proof_shape = prover
            .shape_from_record(&execution.record)
            .expect("c_proj tile machine has no proof shape");
        let trace_area = proof_shape.preprocessed_area + proof_shape.main_area;
        assert!(
            trace_area <= DEFAULT_MAX_TRACE_AREA,
            "c_proj leaf trace area {trace_area} exceeds shape-aware limit {DEFAULT_MAX_TRACE_AREA}"
        );
        println!("proof shape: tile={tile} {proof_shape:?}");

        let TileExecution { tile, output, commitments, record, reference_seconds, execute_seconds } =
            execution;
        let prove_started = Instant::now();
        let shard_proof = prover.prove_shard(pk.clone(), record).await;
        let proof = MachineProof::from(vec![shard_proof]);
        let prove_seconds = prove_started.elapsed().as_secs_f64();
        println!("c_proj proof generated: tile={tile} elapsed={prove_seconds:.3}s");
        let verify_started = Instant::now();
        prover.verify(&vk, &proof).expect("generated c_proj tile proof must verify");
        let verify_seconds = verify_started.elapsed().as_secs_f64();
        println!("c_proj proof verified: tile={tile} elapsed={verify_seconds:.3}s");

        let proof_file = format!("zkgpt_c_proj_l{:02}_t{tile:02}.proof.bin", arguments.layer);
        let proof_path = arguments.output_dir.join(&proof_file);
        bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
        let proof_bytes = fs::metadata(&proof_path).unwrap().len();
        println!("c_proj proof artifact: {} ({proof_bytes} bytes)", proof_path.display());
        completed.push(CompletedTile {
            output,
            summary: TileSummary {
                tile,
                commitments,
                reference_seconds,
                execute_seconds,
                prove_seconds: Some(prove_seconds),
                verify_seconds: Some(verify_seconds),
                proof_file: Some(proof_file),
                proof_bytes: Some(proof_bytes),
            },
        });
    }
    (
        completed,
        BatchMetrics {
            setup_seconds: Some(setup_seconds),
            batch_seconds: batch_started.elapsed().as_secs_f64(),
            verifying_key_file: Some(vk_file),
            verifying_key_bytes: Some(vk_bytes),
            ..BatchMetrics::default()
        },
    )
}

fn compute_group(
    shape: Shape,
    layer: usize,
    upstream: UpstreamCommitments,
    completed: &[CompletedTile],
) -> (Vec<u16>, GroupCommitments) {
    assert_eq!(completed.len(), shape.num_tiles, "c_proj group requires every tile");
    let mut parameter_fields = Vec::with_capacity(shape.num_tiles * (DIGEST_SIZE + 1));
    for (expected_tile, tile) in completed.iter().enumerate() {
        assert_eq!(tile.summary.tile, expected_tile, "c_proj tiles must be ordered");
        assert_eq!(tile.summary.commitments.input, upstream.output, "c_proj inputs differ");
        assert_eq!(
            tile.summary.commitments.upstream, upstream.transcript,
            "c_proj upstream transcripts differ"
        );
        parameter_fields.push(SP1Field::from_canonical_usize(expected_tile));
        parameter_fields.extend(tile.summary.commitments.parameters);
    }
    let parameters = host_commit_fields(DOMAIN_C_PROJ_GROUP_PARAMETERS, &parameter_fields);
    let tile_width = shape.tile_width();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for tile in completed {
            assert_eq!(tile.output.len(), shape.sequence_length * tile_width);
            let start = token * tile_width;
            output.extend_from_slice(&tile.output[start..start + tile_width]);
        }
    }
    let output_digest = host_commit_u16(DOMAIN_C_PROJ_GROUP_OUTPUT, &output);
    let mut fields = [
        PROTOCOL_VERSION,
        C_PROJ_GROUP_STAGE,
        layer as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_tiles as u32,
        shape.tile_width() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(upstream.transcript);
    fields.extend(upstream.output);
    fields.extend(parameters);
    fields.extend(output_digest);
    for tile in completed {
        fields.push(SP1Field::from_canonical_usize(tile.summary.tile));
        fields.extend(tile.summary.commitments.transcript);
    }
    let transcript = host_commit_fields(DOMAIN_C_PROJ_GROUP_TRANSCRIPT, &fields);
    (
        output,
        GroupCommitments { input: upstream.output, parameters, output: output_digest, transcript },
    )
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| format!("{value:.6}"))
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| value.to_string())
}

fn optional_json_string(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| format!("\"{value}\""))
}

fn write_group_artifacts(
    arguments: &Arguments,
    upstream: UpstreamCommitments,
    completed: &[CompletedTile],
    commitments: GroupCommitments,
    output: &[u16],
    metrics: &BatchMetrics,
) {
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_c_proj_l{:02}", arguments.layer);
    let output_file = format!("{stem}.output.private.bf16.bin");
    let output_path = arguments.output_dir.join(&output_file);
    let output_bytes = output.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    fs::write(&output_path, output_bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));

    let mut manifest = String::new();
    writeln!(manifest, "{{").unwrap();
    writeln!(manifest, "  \"version\": {PROTOCOL_VERSION},").unwrap();
    writeln!(manifest, "  \"stage\": \"attention_c_proj_group\",").unwrap();
    writeln!(manifest, "  \"layer\": {},", arguments.layer).unwrap();
    writeln!(manifest, "  \"sequence_length\": {},", arguments.shape.sequence_length).unwrap();
    writeln!(manifest, "  \"hidden_size\": {},", arguments.shape.hidden_size).unwrap();
    writeln!(manifest, "  \"num_tiles\": {},", arguments.shape.num_tiles).unwrap();
    writeln!(manifest, "  \"sharding_strategy\": \"shape_aware_output_columns\",").unwrap();
    writeln!(manifest, "  \"natural_unit\": \"output_column\",").unwrap();
    writeln!(manifest, "  \"units_per_leaf\": {},", arguments.shape.tile_width()).unwrap();
    writeln!(manifest, "  \"max_log_rows\": {},", arguments.shape.max_log_rows()).unwrap();
    writeln!(manifest, "  \"max_trace_area\": {DEFAULT_MAX_TRACE_AREA},").unwrap();
    writeln!(manifest, "  \"upstream_transcript\": \"{}\",", digest_hex(&upstream.transcript))
        .unwrap();
    writeln!(manifest, "  \"input_commitment\": \"{}\",", digest_hex(&commitments.input)).unwrap();
    writeln!(manifest, "  \"parameters_commitment\": \"{}\",", digest_hex(&commitments.parameters))
        .unwrap();
    writeln!(manifest, "  \"output_commitment\": \"{}\",", digest_hex(&commitments.output))
        .unwrap();
    writeln!(manifest, "  \"transcript_commitment\": \"{}\",", digest_hex(&commitments.transcript))
        .unwrap();
    writeln!(manifest, "  \"private_output_file\": \"{output_file}\",").unwrap();
    writeln!(manifest, "  \"output_values\": {},", output.len()).unwrap();
    writeln!(manifest, "  \"build_seconds\": {:.6},", metrics.build_seconds).unwrap();
    writeln!(manifest, "  \"compile_seconds\": {:.6},", metrics.compile_seconds).unwrap();
    writeln!(manifest, "  \"setup_seconds\": {},", optional_f64(metrics.setup_seconds)).unwrap();
    writeln!(manifest, "  \"batch_seconds\": {:.6},", metrics.batch_seconds).unwrap();
    writeln!(
        manifest,
        "  \"verifying_key_file\": {},",
        optional_json_string(metrics.verifying_key_file.as_deref())
    )
    .unwrap();
    writeln!(manifest, "  \"verifying_key_bytes\": {},", optional_u64(metrics.verifying_key_bytes))
        .unwrap();
    writeln!(manifest, "  \"tiles\": [").unwrap();
    for (index, completed_tile) in completed.iter().enumerate() {
        let summary = &completed_tile.summary;
        writeln!(manifest, "    {{").unwrap();
        writeln!(manifest, "      \"tile\": {},", summary.tile).unwrap();
        writeln!(
            manifest,
            "      \"parameters_commitment\": \"{}\",",
            digest_hex(&summary.commitments.parameters)
        )
        .unwrap();
        writeln!(
            manifest,
            "      \"output_commitment\": \"{}\",",
            digest_hex(&summary.commitments.output)
        )
        .unwrap();
        writeln!(
            manifest,
            "      \"transcript_commitment\": \"{}\",",
            digest_hex(&summary.commitments.transcript)
        )
        .unwrap();
        writeln!(manifest, "      \"reference_seconds\": {:.6},", summary.reference_seconds)
            .unwrap();
        writeln!(manifest, "      \"execute_seconds\": {:.6},", summary.execute_seconds).unwrap();
        writeln!(manifest, "      \"prove_seconds\": {},", optional_f64(summary.prove_seconds))
            .unwrap();
        writeln!(manifest, "      \"verify_seconds\": {},", optional_f64(summary.verify_seconds))
            .unwrap();
        writeln!(
            manifest,
            "      \"proof_file\": {},",
            optional_json_string(summary.proof_file.as_deref())
        )
        .unwrap();
        writeln!(manifest, "      \"proof_bytes\": {}", optional_u64(summary.proof_bytes)).unwrap();
        let suffix = if index + 1 == completed.len() { "" } else { "," };
        writeln!(manifest, "    }}{suffix}").unwrap();
    }
    writeln!(manifest, "  ]").unwrap();
    writeln!(manifest, "}}").unwrap();

    let manifest_path = arguments.output_dir.join(format!("{stem}.manifest.json"));
    fs::write(&manifest_path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
    println!("c_proj private output: {}", output_path.display());
    println!("c_proj group manifest: {}", manifest_path.display());
    println!("c_proj output commitment: {}", digest_hex(&commitments.output));
    println!("c_proj group transcript: {}", digest_hex(&commitments.transcript));
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let tiles = selected_tiles(&arguments);
    let events = EventCounts::for_shape(arguments.shape);
    println!(
        "mode={:?} layer={} tiles={tiles:?} reuse_setup={}",
        arguments.mode, arguments.layer, arguments.all_tiles
    );
    print_estimate(arguments.shape, events);
    if arguments.mode == Mode::Estimate {
        return;
    }

    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: TileBuilder = AsmBuilder::default();
    build_tile(&mut builder, arguments.shape);
    let block = builder.into_root_block();
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("built c_proj tile: ir_ops={} elapsed={build_seconds:.3}s", block.ops.len());
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let load_started = Instant::now();
    let source = load_source_data(&arguments);
    println!(
        "loaded source: input={} full_weight={} elapsed={:.3}s",
        source.input.len(),
        source.full_weight.len(),
        load_started.elapsed().as_secs_f64()
    );
    println!("upstream transcript: {}", digest_hex(&source.upstream.transcript));
    println!("c_proj input commitment: {}", digest_hex(&source.upstream.output));

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    println!("compiled c_proj tile: elapsed={compile_seconds:.3}s");

    let (completed, mut metrics) = match arguments.mode {
        Mode::Execute => {
            let (completed, batch_seconds) =
                execute_tiles(program, &arguments, events, &source, &tiles);
            (completed, BatchMetrics { batch_seconds, ..BatchMetrics::default() })
        }
        Mode::Prove => prove_tiles(program, &arguments, events, &source, &tiles).await,
        Mode::Estimate | Mode::Build => unreachable!(),
    };
    metrics.build_seconds = build_seconds;
    metrics.compile_seconds = compile_seconds;

    if arguments.all_tiles {
        let (output, commitments) =
            compute_group(arguments.shape, arguments.layer, source.upstream, &completed);
        write_group_artifacts(
            &arguments,
            source.upstream,
            &completed,
            commitments,
            &output,
            &metrics,
        );
    }
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_arguments(shape: Shape) -> Arguments {
        Arguments {
            mode: Mode::Execute,
            shape,
            layer: DEFAULT_LAYER,
            tile: 0,
            all_tiles: false,
            synthetic: true,
            attention_dir: PathBuf::new(),
            join_dir: PathBuf::new(),
            data_dir: PathBuf::new(),
            output_dir: PathBuf::new(),
        }
    }

    #[test]
    fn extracts_row_major_output_columns() {
        let shape = Shape::small();
        let full = (0..shape.hidden_size * shape.hidden_size)
            .map(|value| value as u16)
            .collect::<Vec<_>>();
        let tile = extract_weight_tile(&full, shape, 1);
        let expected = full
            .chunks_exact(shape.hidden_size)
            .flat_map(|row| row[shape.tile_width()..].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(tile, expected);
    }

    #[test]
    fn tile_circuit_rejects_wrong_upstream_input() {
        let shape = Shape::small();
        let arguments = test_arguments(shape);
        let input = vec![0; shape.sequence_length * shape.hidden_size];
        let upstream = UpstreamCommitments {
            output: host_commit_u16(DOMAIN_ATTENTION_GROUP_OUTPUT, &input),
            transcript: host_commit_fields(0x2201, &[SP1Field::one()]),
        };
        let data = TileData {
            input,
            weight: vec![0x3f80; shape.hidden_size * shape.tile_width()],
            upstream,
        };
        let output = reference_linear_rows_no_bias(&data.input, shape.hidden_size, &data.weight);
        let expected = compute_tile_commitments(shape, DEFAULT_LAYER, DEFAULT_TILE, &data, &output);

        let mut builder: TileBuilder = AsmBuilder::default();
        build_tile(&mut builder, shape);
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(
            compiler.compile_inner(builder.into_root_block()).validate().expect("valid tile IR"),
        );
        let execute = |data: &TileData| {
            let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
                program.clone(),
                inner_perm(),
            );
            executor.witness_stream = witness_stream(&arguments, DEFAULT_TILE, data).into();
            let result = executor.run();
            (result.is_ok(), executor.record.public_values.digest)
        };

        let (succeeded, digest) = execute(&data);
        assert!(succeeded);
        assert_eq!(digest, expected.transcript);

        let mut tampered = data;
        tampered.input[0] ^= 1;
        assert!(!execute(&tampered).0, "input differing from join commitment must fail");
    }
}
