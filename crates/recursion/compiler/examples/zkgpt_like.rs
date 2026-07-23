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
    circuit::{AsmBuilder, AsmCompiler, CircuitV2Builder},
    prelude::{Bf16ZkGptBlockParams, Builder, Felt},
};
use sp1_recursion_executor::{
    Bf16AddSubEvent, Bf16DivEvent, Bf16MulEvent, Bf16UnaryEvent, ExecutionRecord, Executor,
    RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH,
    RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::{
    chips::bf16::{NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW},
    RecursionAir,
};

const DEFAULT_LAYERS: usize = 12;
const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_NUM_HEADS: usize = 12;
const DEFAULT_LINEAR_SIZE: usize = 2304;
const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;
const GPT2_ATTENTION_SCALE: u16 = 0x3e00;
const SAFE_MATERIALIZED_EVENT_LIMIT: u128 = 100_000_000;
const PROOF_LOG_BLOWUP: usize = 1;
const PROOF_LOG_STACKING_HEIGHT: u32 = 22;
// The full 12 x 30 computation has about 2^28 rows in its largest BF16 chips. The verifier
// requires max_log_row_count < 29, so 28 is both necessary and the largest supported value.
const PROOF_MAX_LOG_ROW_COUNT: usize = 28;
const BLOCK_PROTOCOL_VERSION: u32 = 1;
const BLOCK_STAGE_GENESIS: u32 = 0x1600;
const BLOCK_STAGE_CHAINED: u32 = 0x1601;
const DOMAIN_BLOCK_INPUT: u32 = 0x1610;
const DOMAIN_BLOCK_PARAMETERS: u32 = 0x1611;
const DOMAIN_BLOCK_HINTS: u32 = 0x1612;
const DOMAIN_BLOCK_OUTPUT: u32 = 0x1613;
const DOMAIN_BLOCK_TRANSCRIPT: u32 = 0x1614;

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
    CheckData,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    layers: usize,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            layers: DEFAULT_LAYERS,
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            num_heads: DEFAULT_NUM_HEADS,
            linear_size: DEFAULT_LINEAR_SIZE,
        }
    }
}

impl Shape {
    fn validate(self) {
        assert!(self.layers > 0, "zkGPT-like inference requires at least one layer");
        assert!(self.sequence_length > 0, "zkGPT-like inference requires at least one token");
        assert!(self.hidden_size > 0, "zkGPT-like inference requires a hidden dimension");
        assert!(self.num_heads > 0, "zkGPT-like inference requires at least one head");
        assert_eq!(
            self.hidden_size % self.num_heads,
            0,
            "hidden size must be divisible by the head count"
        );
        assert_eq!(
            self.linear_size,
            3 * self.hidden_size,
            "the zkGPT prototype uses one 3 * hidden width for QKV and the MLP"
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EventCounts {
    mul: u128,
    add_sub: u128,
    unary: u128,
    div: u128,
}

impl EventCounts {
    fn for_shape(shape: Shape) -> Self {
        shape.validate();
        let layers = shape.layers as u128;
        let sequence_length = shape.sequence_length as u128;
        let hidden_size = shape.hidden_size as u128;
        let num_heads = shape.num_heads as u128;
        let linear_size = shape.linear_size as u128;
        let head_dimension = hidden_size / num_heads;
        let causal_scores = num_heads * sequence_length * (sequence_length + 1) / 2;

        // Two row-wise layer normalizations.
        let layer_norm_mul = sequence_length * 4 * hidden_size;
        let layer_norm_add_sub = sequence_length * (8 * hidden_size - 2);
        let layer_norm_unary = sequence_length * (2 * hidden_size + 2);
        let layer_norm_div = sequence_length * 4;

        // QKV, attention projection, MLP expansion, and MLP projection, all without bias.
        let linear_mul =
            sequence_length * (3 * hidden_size * linear_size + hidden_size * hidden_size);
        let linear_add_sub = sequence_length
            * (3 * hidden_size * linear_size + hidden_size * hidden_size
                - 2 * linear_size
                - 2 * hidden_size);

        // Causal QK scores, max subtraction, softmax accumulation, and probability-weighted V.
        let attention_mul = causal_scores * (2 * head_dimension + 1);
        let attention_add_sub = causal_scores * head_dimension
            + (num_heads + hidden_size) * sequence_length * (sequence_length - 1) / 2;
        let attention_unary = causal_scores;
        let attention_div = causal_scores;

        let per_layer = Self {
            mul: layer_norm_mul + linear_mul + attention_mul,
            add_sub: layer_norm_add_sub + linear_add_sub + attention_add_sub,
            unary: layer_norm_unary + attention_unary + sequence_length * linear_size,
            div: layer_norm_div + attention_div,
        };
        Self {
            mul: per_layer.mul * layers,
            add_sub: per_layer.add_sub * layers,
            unary: per_layer.unary * layers,
            div: per_layer.div * layers,
        }
    }

    fn total(self) -> u128 {
        self.mul + self.add_sub + self.unary + self.div
    }

    fn standard_gpt2_for_shape(shape: Shape) -> Self {
        shape.validate();
        let layers = shape.layers as u128;
        let sequence_length = shape.sequence_length as u128;
        let hidden_size = shape.hidden_size as u128;
        let num_heads = shape.num_heads as u128;
        let inner_size = 4 * hidden_size;
        let head_dimension = hidden_size / num_heads;
        let causal_scores = num_heads * sequence_length * (sequence_length + 1) / 2;

        let layer_norm_mul = sequence_length * 4 * hidden_size;
        let layer_norm_add_sub = sequence_length * (8 * hidden_size - 2);
        let layer_norm_unary = sequence_length * (2 * hidden_size + 2);
        let layer_norm_div = sequence_length * 4;
        let linear_mul =
            sequence_length * (4 * hidden_size * hidden_size + 2 * hidden_size * inner_size);
        // Standard GPT-2 adds a bias after every dot product and has two residual additions.
        let linear_and_residual_add_sub = linear_mul + 2 * sequence_length * hidden_size;
        let attention_mul = causal_scores * (2 * head_dimension + 1);
        let attention_add_sub = causal_scores * head_dimension
            + (num_heads + hidden_size) * sequence_length * (sequence_length - 1) / 2;

        let per_layer = Self {
            mul: layer_norm_mul + linear_mul + attention_mul,
            add_sub: layer_norm_add_sub + linear_and_residual_add_sub + attention_add_sub,
            unary: layer_norm_unary + causal_scores + sequence_length * inner_size,
            div: layer_norm_div + causal_scores,
        };
        Self {
            mul: per_layer.mul * layers,
            add_sub: per_layer.add_sub * layers,
            unary: per_layer.unary * layers,
            div: per_layer.div * layers,
        }
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    shape: Shape,
    allow_large_build: bool,
    output_dir: PathBuf,
    data_dir: PathBuf,
    synthetic: bool,
    block: Option<usize>,
    previous_block_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct RealLayerData {
    layer_norm_1_weight: Vec<u16>,
    layer_norm_1_bias: Vec<u16>,
    attention_qkv_weight: Vec<u16>,
    attention_projection_weight: Vec<u16>,
    layer_norm_2_weight: Vec<u16>,
    layer_norm_2_bias: Vec<u16>,
    mlp_expansion_weight: Vec<u16>,
    mlp_projection_weight: Vec<u16>,
    attention_max_hints: Vec<u16>,
}

#[derive(Debug)]
struct RealData {
    hidden_states: Vec<u16>,
    layers: Vec<RealLayerData>,
}

#[derive(Clone, Copy, Debug)]
struct UpstreamCommitments {
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Copy, Debug)]
struct BlockCommitments {
    upstream: Option<Digest>,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Debug)]
struct ProofArtifacts {
    proof_file: String,
    proof_bytes: u64,
    verifying_key_file: String,
    verifying_key_bytes: u64,
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

fn parse_arguments() -> Arguments {
    let mut mode = Mode::Estimate;
    let mut shape = Shape::default();
    let mut allow_large_build = false;
    let mut output_dir = std::env::var_os("SP1_ZKGPT_LIKE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-like-output"));
    let mut data_dir = std::env::var_os("SP1_ZKGPT_LIKE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    let mut synthetic = false;
    let mut block = None;
    let mut previous_block_dir = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--estimate") => mode = Mode::Estimate,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--check-data") => mode = Mode::CheckData,
            Some("--small") => {
                shape = Shape {
                    layers: 2,
                    sequence_length: 2,
                    hidden_size: 8,
                    num_heads: 2,
                    linear_size: 24,
                };
                synthetic = true;
            }
            Some("--layers") => shape.layers = parse_usize(arguments.next(), "--layers"),
            Some("--seq-len") => shape.sequence_length = parse_usize(arguments.next(), "--seq-len"),
            Some("--hidden-size") => {
                shape.hidden_size = parse_usize(arguments.next(), "--hidden-size")
            }
            Some("--heads") => shape.num_heads = parse_usize(arguments.next(), "--heads"),
            Some("--linear-size") => {
                shape.linear_size = parse_usize(arguments.next(), "--linear-size")
            }
            Some("--block") => block = Some(parse_usize(arguments.next(), "--block")),
            Some("--previous-block-dir") => {
                previous_block_dir = Some(PathBuf::from(
                    arguments
                        .next()
                        .unwrap_or_else(|| panic!("--previous-block-dir requires a value")),
                ))
            }
            Some("--allow-large-build") => allow_large_build = true,
            Some("--synthetic") => synthetic = true,
            Some("--real") => synthetic = false,
            Some("--data-dir") => {
                data_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--data-dir requires a value")),
                );
                synthetic = false;
            }
            Some("--output-dir") => {
                output_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--output-dir requires a value")),
                )
            }
            Some(value) => panic!("unknown option: {value}"),
            None => panic!("command-line options must be valid UTF-8"),
        }
    }
    if let Some(block) = block {
        assert!(block < DEFAULT_LAYERS, "block must be in 0..{DEFAULT_LAYERS}");
        shape.layers = 1;
        if block == 0 {
            assert!(previous_block_dir.is_none(), "block 0 must not specify --previous-block-dir");
        } else {
            assert!(previous_block_dir.is_some(), "block {block} requires --previous-block-dir");
        }
    } else {
        assert!(
            previous_block_dir.is_none(),
            "--previous-block-dir is only valid together with --block"
        );
    }
    shape.validate();
    Arguments {
        mode,
        shape,
        allow_large_build,
        output_dir,
        data_dir,
        synthetic,
        block,
        previous_block_dir,
    }
}

fn read_bf16_binary(path: &Path) -> Vec<u16> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 2, 0, "{} has an odd byte length", path.display());
    bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect()
}

fn digest_hex(digest: &Digest) -> String {
    digest
        .iter()
        .map(|value| format!("{:08X}", value.as_canonical_u32()))
        .collect::<Vec<_>>()
        .join(":")
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

fn json_usize_field(path: &Path, key: &str) -> usize {
    json_string_field(path, key)
        .parse()
        .unwrap_or_else(|error| panic!("{} has invalid {key}: {error}", path.display()))
}

fn block_stem(block: usize) -> String {
    format!("zkgpt_block_l{block:02}")
}

fn verify_and_load_previous_block(arguments: &Arguments) -> (Vec<u16>, UpstreamCommitments) {
    type A = RecursionAir<SP1Field, 3, 2>;

    let block = arguments.block.expect("previous Block verification requires --block");
    assert!(block > 0, "Block 0 has no previous Block");
    let previous = block - 1;
    let directory = arguments
        .previous_block_dir
        .as_ref()
        .expect("Block 1 or later requires --previous-block-dir");
    let stem = block_stem(previous);
    let manifest_path = directory.join(format!("{stem}.manifest.json"));
    assert_eq!(
        json_string_field(&manifest_path, "stage"),
        "zkgpt_block",
        "previous Block manifest has the wrong stage"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "block"),
        previous,
        "previous Block manifest has the wrong Block index"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "sequence_length"),
        arguments.shape.sequence_length,
        "previous Block sequence length differs"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "hidden_size"),
        arguments.shape.hidden_size,
        "previous Block hidden size differs"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "shards"),
        1,
        "previous Block proof must contain exactly one shard"
    );
    let upstream = UpstreamCommitments {
        output: parse_digest(&json_string_field(&manifest_path, "output_commitment")),
        transcript: parse_digest(&json_string_field(&manifest_path, "transcript_commitment")),
    };

    let proof_path = directory.join(json_string_field(&manifest_path, "proof_file"));
    let vk_path = directory.join(json_string_field(&manifest_path, "verifying_key_file"));
    let proof: StoredProof = bincode::deserialize_from(
        File::open(&proof_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", proof_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", proof_path.display()));
    let vk: StoredVerifyingKey = bincode::deserialize_from(
        File::open(&vk_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", vk_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", vk_path.display()));
    assert_eq!(proof.shard_proofs.len(), 1, "previous Block proof must contain one shard");
    let public_values: &RecursionPublicValues<SP1Field> =
        proof.shard_proofs[0].public_values.as_slice().borrow();
    assert_eq!(
        public_values.digest, upstream.transcript,
        "previous Block proof public digest differs from its manifest"
    );
    let verifier = ShardVerifier::from_basefold_parameters(
        FriConfig::new(
            PROOF_LOG_BLOWUP,
            unique_decoding_queries(PROOF_LOG_BLOWUP),
            SP1_PROOF_OF_WORK_BITS,
        ),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        A::verillm_machine(),
    );
    simple_prover(verifier)
        .verify(&vk, &proof)
        .expect("previous Block proof must verify before it can be chained");

    let output_path = directory.join(json_string_field(&manifest_path, "private_output_file"));
    let hidden_states = read_bf16_binary(&output_path);
    assert_eq!(
        hidden_states.len(),
        arguments.shape.sequence_length * arguments.shape.hidden_size,
        "previous Block private output shape differs"
    );
    assert_eq!(
        host_commit_u16(DOMAIN_BLOCK_OUTPUT, &hidden_states),
        upstream.output,
        "previous Block private output differs from its proved commitment"
    );
    println!(
        "previous Block verified: block={previous} transcript={}",
        digest_hex(&upstream.transcript)
    );
    (hidden_states, upstream)
}

fn load_real_data(
    data_dir: &Path,
    shape: Shape,
    first_layer: usize,
    hidden_states: Option<Vec<u16>>,
) -> RealData {
    let load_started = Instant::now();
    let hidden_states =
        hidden_states.unwrap_or_else(|| read_bf16_binary(&data_dir.join("hidden_state.bf16.bin")));
    assert_eq!(
        hidden_states.len(),
        shape.sequence_length * shape.hidden_size,
        "real BF16 hidden-state shape mismatch"
    );
    let mut layers = Vec::with_capacity(shape.layers);
    for layer in first_layer..first_layer + shape.layers {
        let layer_dir = data_dir.join(format!("layer-{layer:02}"));
        let read = |name: &str| read_bf16_binary(&layer_dir.join(format!("{name}.bf16.bin")));
        let data = RealLayerData {
            layer_norm_1_weight: read("ln_1_weight"),
            layer_norm_1_bias: read("ln_1_bias"),
            attention_qkv_weight: read("attention_qkv_weight"),
            attention_projection_weight: read("attention_projection_weight"),
            layer_norm_2_weight: read("ln_2_weight"),
            layer_norm_2_bias: read("ln_2_bias"),
            mlp_expansion_weight: read("mlp_expansion_weight"),
            mlp_projection_weight: read("mlp_projection_weight"),
            attention_max_hints: read("attention_max_hints"),
        };
        assert_eq!(data.layer_norm_1_weight.len(), shape.hidden_size);
        assert_eq!(data.layer_norm_1_bias.len(), shape.hidden_size);
        assert_eq!(data.attention_qkv_weight.len(), shape.hidden_size * 3 * shape.hidden_size);
        assert_eq!(data.attention_projection_weight.len(), shape.hidden_size * shape.hidden_size);
        assert_eq!(data.layer_norm_2_weight.len(), shape.hidden_size);
        assert_eq!(data.layer_norm_2_bias.len(), shape.hidden_size);
        assert_eq!(data.mlp_expansion_weight.len(), shape.hidden_size * shape.linear_size);
        assert_eq!(data.mlp_projection_weight.len(), shape.linear_size * shape.hidden_size);
        assert_eq!(data.attention_max_hints.len(), shape.sequence_length * shape.num_heads);
        layers.push(data);
    }
    println!(
        "loaded real BF16 fixture: data_dir={} layers={} hidden_values={} elapsed={:.3}s",
        data_dir.display(),
        layers.len(),
        hidden_states.len(),
        load_started.elapsed().as_secs_f64()
    );
    RealData { hidden_states, layers }
}

fn padded_rows(events: u128, lanes: usize) -> u128 {
    let lanes = lanes as u128;
    events.div_ceil(lanes).next_multiple_of(32)
}

fn next_power_of_two(value: usize) -> usize {
    value.next_power_of_two()
}

fn print_estimate(shape: Shape, events: EventCounts) {
    let standard_events = EventCounts::standard_gpt2_for_shape(shape);
    let parameter_values_per_layer = 4 * shape.hidden_size
        + shape.hidden_size * 3 * shape.hidden_size
        + shape.hidden_size * shape.hidden_size
        + shape.hidden_size * shape.linear_size
        + shape.linear_size * shape.hidden_size;
    let padded_sequence = next_power_of_two(shape.sequence_length);
    let padded_hidden = next_power_of_two(shape.hidden_size);
    let padded_linear = next_power_of_two(shape.linear_size);
    let padded_matrix_muls = (shape.layers as u128)
        * (padded_sequence as u128)
        * (3 * (padded_hidden as u128) * (padded_linear as u128)
            + (padded_hidden as u128) * (padded_hidden as u128));
    let max_rows = 1u128 << 23;
    let proof_max_rows = 1u128 << PROOF_MAX_LOG_ROW_COUNT;
    let event_record_bytes = events.mul * size_of_u128::<Bf16MulEvent<SP1Field>>()
        + events.add_sub * size_of_u128::<Bf16AddSubEvent<SP1Field>>()
        + events.unary * size_of_u128::<Bf16UnaryEvent<SP1Field>>()
        + events.div * size_of_u128::<Bf16DivEvent<SP1Field>>();

    println!("modeled architecture: zkGPT-like full-sequence transformer");
    println!(
        "shape: layers={} seq_len={} hidden={} heads={} head_dim={} linear={}",
        shape.layers,
        shape.sequence_length,
        shape.hidden_size,
        shape.num_heads,
        shape.hidden_size / shape.num_heads,
        shape.linear_size,
    );
    println!("semantics: bias=false residual=false ln_f=false lm_head=false arithmetic=BF16");
    println!(
        "parameters: per_layer={parameter_values_per_layer} total={}",
        parameter_values_per_layer * shape.layers
    );
    println!(
        "events: mul={} add_sub={} unary={} div={} total={}",
        events.mul,
        events.add_sub,
        events.unary,
        events.div,
        events.total(),
    );
    println!(
        "12-lane rows: mul={} add_sub={} max_2^23={max_rows}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW),
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW),
    );
    println!(
        "monolithic proof limit: max_2^{PROOF_MAX_LOG_ROW_COUNT}={proof_max_rows} \
         mul_fits={} add_sub_fits={}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW) <= proof_max_rows,
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW) <= proof_max_rows,
    );
    println!(
        "BF16 executor-record lower bound: bytes={event_record_bytes} gib={:.2} \
         (excludes IR, traces, PCS workspace, and proving keys)",
        event_record_bytes as f64 / 1024_f64.powi(3)
    );
    println!(
        "12-lane minimum shards at 2^23 rows: mul={} add_sub={}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW).div_ceil(max_rows),
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW).div_ceil(max_rows),
    );
    println!(
        "minimum lanes for one 2^23-row shard: mul={} add_sub={}",
        events.mul.div_ceil(max_rows),
        events.add_sub.div_ceil(max_rows),
    );
    println!(
        "zkGPT power-of-two padding: seq={padded_sequence} hidden={padded_hidden} linear={padded_linear}"
    );
    println!("zkGPT padded FC multiplication gates: {padded_matrix_muls}");
    println!(
        "standard GPT-2 at same shape (without ln_f/LM head): mul={} add_sub={} unary={} div={}",
        standard_events.mul, standard_events.add_sub, standard_events.unary, standard_events.div,
    );
    println!(
        "standard 12-lane rows: mul={} add_sub={}",
        padded_rows(standard_events.mul, NUM_BF16_MUL_EVENTS_PER_ROW),
        padded_rows(standard_events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW),
    );
    println!(
        "zkGPT-like event reduction: mul={:.2}% add_sub={:.2}%",
        100.0 * (1.0 - events.mul as f64 / standard_events.mul as f64),
        100.0 * (1.0 - events.add_sub as f64 / standard_events.add_sub as f64),
    );
}

fn size_of_u128<T>() -> u128 {
    std::mem::size_of::<T>() as u128
}

fn repeated_constants(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    raw: u16,
    count: usize,
) -> Vec<Felt<SP1Field>> {
    let value = SP1Field::from_canonical_u16(raw);
    (0..count).map(|_| builder.constant(value)).collect()
}

fn constants(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[u16],
) -> Vec<Felt<SP1Field>> {
    values.iter().map(|&value| builder.constant(SP1Field::from_canonical_u16(value))).collect()
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

fn circuit_commit_fields(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
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

fn block_transcript_fields_host(
    arguments: &Arguments,
    upstream: Option<Digest>,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
) -> Vec<SP1Field> {
    let block = arguments.block.expect("Block transcript requires --block");
    let mut fields = [
        BLOCK_PROTOCOL_VERSION,
        if block == 0 { BLOCK_STAGE_GENESIS } else { BLOCK_STAGE_CHAINED },
        block as u32,
        arguments.shape.sequence_length as u32,
        arguments.shape.hidden_size as u32,
        arguments.shape.num_heads as u32,
        arguments.shape.linear_size as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
        GPT2_ATTENTION_SCALE as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    if let Some(upstream) = upstream {
        fields.extend(upstream);
    }
    fields.extend(input);
    fields.extend(parameters);
    fields.extend(hints);
    fields.extend(output);
    fields
}

fn compute_host_block_commitments(
    arguments: &Arguments,
    upstream: Option<UpstreamCommitments>,
    input: &[u16],
    parameters: &[u16],
    hints: &[u16],
    output: &[u16],
) -> BlockCommitments {
    let block = arguments.block.expect("Block commitments require --block");
    let input_domain = if block == 0 { DOMAIN_BLOCK_INPUT } else { DOMAIN_BLOCK_OUTPUT };
    let input = host_commit_u16(input_domain, input);
    if let Some(upstream) = upstream {
        assert_eq!(input, upstream.output, "Block input differs from previous Block output");
    }
    let parameters = host_commit_u16(DOMAIN_BLOCK_PARAMETERS, parameters);
    let hints = host_commit_u16(DOMAIN_BLOCK_HINTS, hints);
    let output = host_commit_u16(DOMAIN_BLOCK_OUTPUT, output);
    let upstream_transcript = upstream.map(|value| value.transcript);
    let transcript = host_commit_fields(
        DOMAIN_BLOCK_TRANSCRIPT,
        &block_transcript_fields_host(
            arguments,
            upstream_transcript,
            input,
            parameters,
            hints,
            output,
        ),
    );
    BlockCommitments { upstream: upstream_transcript, input, parameters, hints, output, transcript }
}

fn commit_block_transcript(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    arguments: &Arguments,
    upstream: Option<UpstreamCommitments>,
    input: &[Felt<SP1Field>],
    parameters: &[Felt<SP1Field>],
    hints: &[Felt<SP1Field>],
    output: &[Felt<SP1Field>],
) {
    let block = arguments.block.expect("Block transcript requires --block");
    let input_domain = if block == 0 { DOMAIN_BLOCK_INPUT } else { DOMAIN_BLOCK_OUTPUT };
    let input_digest = circuit_commit_fields(builder, input_domain, input);
    if let Some(upstream) = upstream {
        for (&computed, expected) in input_digest.iter().zip(upstream.output) {
            builder.assert_felt_eq(computed, expected);
        }
    }
    let parameters_digest = circuit_commit_fields(builder, DOMAIN_BLOCK_PARAMETERS, parameters);
    let hints_digest = circuit_commit_fields(builder, DOMAIN_BLOCK_HINTS, hints);
    let output_digest = circuit_commit_fields(builder, DOMAIN_BLOCK_OUTPUT, output);

    let constants = [
        BLOCK_PROTOCOL_VERSION,
        if block == 0 { BLOCK_STAGE_GENESIS } else { BLOCK_STAGE_CHAINED },
        block as u32,
        arguments.shape.sequence_length as u32,
        arguments.shape.hidden_size as u32,
        arguments.shape.num_heads as u32,
        arguments.shape.linear_size as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
        GPT2_ATTENTION_SCALE as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut transcript_fields = constants.to_vec();
    if let Some(upstream) = upstream {
        transcript_fields.extend(
            upstream
                .transcript
                .into_iter()
                .map(|value| builder.constant::<Felt<SP1Field>>(value)),
        );
    }
    transcript_fields.extend(input_digest);
    transcript_fields.extend(parameters_digest);
    transcript_fields.extend(hints_digest);
    transcript_fields.extend(output_digest);
    let transcript = circuit_commit_fields(builder, DOMAIN_BLOCK_TRANSCRIPT, &transcript_fields);

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = transcript;
    builder.commit_public_values_v2(*public_values);
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

async fn prove_and_verify(
    program: Arc<RecursionProgram<SP1Field>>,
    record: ExecutionRecord<SP1Field>,
    output_dir: &Path,
    artifact_stem: &str,
) -> ProofArtifacts {
    type A = RecursionAir<SP1Field, 3, 2>;

    let end_to_end_started = Instant::now();
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
    // The development prover has one permit. Release the setup permit before proving.
    let pk = unsafe { pk.into_inner() };

    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    assert_eq!(proof.shard_proofs.len(), 1, "a Block proof must contain exactly one shard");
    println!("proof generated: elapsed={:.3}s", prove_started.elapsed().as_secs_f64());

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated zkGPT-like proof must verify");
    println!("proof verified: elapsed={:.3}s", verify_started.elapsed().as_secs_f64());

    fs::create_dir_all(output_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", output_dir.display()));
    let proof_file = format!("{artifact_stem}.proof.bin");
    let verifying_key_file = format!("{artifact_stem}.vk.bin");
    let proof_path = output_dir.join(&proof_file);
    let vk_path = output_dir.join(&verifying_key_file);
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    let proof_bytes = fs::metadata(&proof_path).unwrap().len();
    let verifying_key_bytes = fs::metadata(&vk_path).unwrap().len();
    println!(
        "proof artifacts: proof={} ({} bytes) vk={} ({} bytes)",
        proof_path.display(),
        proof_bytes,
        vk_path.display(),
        verifying_key_bytes,
    );
    println!(
        "proof end-to-end (setup+prove+verify+serialize): elapsed={:.3}s",
        end_to_end_started.elapsed().as_secs_f64()
    );
    ProofArtifacts { proof_file, proof_bytes, verifying_key_file, verifying_key_bytes }
}

fn raw_block_parameters(data: &RealLayerData) -> Vec<u16> {
    let mut values = Vec::with_capacity(
        data.layer_norm_1_weight.len()
            + data.layer_norm_1_bias.len()
            + data.attention_qkv_weight.len()
            + data.attention_projection_weight.len()
            + data.layer_norm_2_weight.len()
            + data.layer_norm_2_bias.len()
            + data.mlp_expansion_weight.len()
            + data.mlp_projection_weight.len(),
    );
    values.extend_from_slice(&data.layer_norm_1_weight);
    values.extend_from_slice(&data.layer_norm_1_bias);
    values.extend_from_slice(&data.attention_qkv_weight);
    values.extend_from_slice(&data.attention_projection_weight);
    values.extend_from_slice(&data.layer_norm_2_weight);
    values.extend_from_slice(&data.layer_norm_2_bias);
    values.extend_from_slice(&data.mlp_expansion_weight);
    values.extend_from_slice(&data.mlp_projection_weight);
    values
}

fn read_printed_bf16(path: &Path) -> Vec<u16> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(|line| {
            line.strip_prefix("PRINTF=")
                .unwrap_or_else(|| panic!("unexpected circuit output line: {line}"))
                .parse::<u16>()
                .unwrap_or_else(|error| panic!("invalid circuit BF16 output {line}: {error}"))
        })
        .collect()
}

fn write_private_block_output(output_dir: &Path, block: usize, output: &[u16]) -> String {
    fs::create_dir_all(output_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", output_dir.display()));
    let output_file = format!("{}.output.private.bf16.bin", block_stem(block));
    let output_path = output_dir.join(&output_file);
    let bytes = output.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    fs::write(&output_path, bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));
    output_file
}

fn write_block_manifest(
    arguments: &Arguments,
    commitments: BlockCommitments,
    private_output_file: &str,
    proof: &ProofArtifacts,
) {
    let block = arguments.block.expect("Block manifest requires --block");
    let stem = block_stem(block);
    let manifest_path = arguments.output_dir.join(format!("{stem}.manifest.json"));
    let upstream = commitments
        .upstream
        .map_or_else(|| "null".to_owned(), |digest| format!("\"{}\"", digest_hex(&digest)));
    let mut manifest = String::new();
    writeln!(manifest, "{{").unwrap();
    writeln!(manifest, "  \"version\": {BLOCK_PROTOCOL_VERSION},").unwrap();
    writeln!(manifest, "  \"stage\": \"zkgpt_block\",").unwrap();
    writeln!(manifest, "  \"block\": {block},").unwrap();
    writeln!(manifest, "  \"sequence_length\": {},", arguments.shape.sequence_length).unwrap();
    writeln!(manifest, "  \"hidden_size\": {},", arguments.shape.hidden_size).unwrap();
    writeln!(manifest, "  \"num_heads\": {},", arguments.shape.num_heads).unwrap();
    writeln!(manifest, "  \"linear_size\": {},", arguments.shape.linear_size).unwrap();
    writeln!(manifest, "  \"shards\": 1,").unwrap();
    writeln!(manifest, "  \"upstream_transcript\": {upstream},").unwrap();
    writeln!(manifest, "  \"input_commitment\": \"{}\",", digest_hex(&commitments.input)).unwrap();
    writeln!(manifest, "  \"parameters_commitment\": \"{}\",", digest_hex(&commitments.parameters))
        .unwrap();
    writeln!(manifest, "  \"hints_commitment\": \"{}\",", digest_hex(&commitments.hints)).unwrap();
    writeln!(manifest, "  \"output_commitment\": \"{}\",", digest_hex(&commitments.output))
        .unwrap();
    writeln!(manifest, "  \"transcript_commitment\": \"{}\",", digest_hex(&commitments.transcript))
        .unwrap();
    writeln!(manifest, "  \"private_output_file\": \"{private_output_file}\",").unwrap();
    writeln!(
        manifest,
        "  \"output_values\": {},",
        arguments.shape.sequence_length * arguments.shape.hidden_size
    )
    .unwrap();
    writeln!(manifest, "  \"proof_file\": \"{}\",", proof.proof_file).unwrap();
    writeln!(manifest, "  \"proof_bytes\": {},", proof.proof_bytes).unwrap();
    writeln!(manifest, "  \"verifying_key_file\": \"{}\",", proof.verifying_key_file).unwrap();
    writeln!(manifest, "  \"verifying_key_bytes\": {}", proof.verifying_key_bytes).unwrap();
    writeln!(manifest, "}}").unwrap();
    fs::write(&manifest_path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
    println!("Block manifest: {}", manifest_path.display());
}

async fn materialize(arguments: &Arguments, expected: EventCounts) {
    if expected.total() > SAFE_MATERIALIZED_EVENT_LIMIT && !arguments.allow_large_build {
        panic!(
            "refusing to materialize {} BF16 events; pass --allow-large-build only on a machine \
             provisioned for this experiment",
            expected.total()
        );
    }

    let shape = arguments.shape;
    let (block_input, upstream) = match arguments.block {
        Some(0) | None => (None, None),
        Some(_) => {
            let (hidden_states, upstream) = verify_and_load_previous_block(arguments);
            (Some(hidden_states), Some(upstream))
        }
    };
    let real_data = if arguments.synthetic {
        println!(
            "fixture: synthetic hidden=zeros linear_weights=ones \
             layer_norm_weight=one layer_norm_bias=zero"
        );
        None
    } else {
        println!("fixture: real pretrained GPT-2 BF16 weights adapted to zkGPT layout");
        Some(load_real_data(
            &arguments.data_dir,
            shape,
            arguments.block.unwrap_or(0),
            block_input.clone(),
        ))
    };
    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let raw_input = match (&real_data, block_input) {
        (Some(data), _) => data.hidden_states.clone(),
        (None, Some(hidden_states)) => hidden_states,
        (None, None) => vec![0x0000; shape.sequence_length * shape.hidden_size],
    };
    let mut hidden_states = constants(&mut builder, &raw_input);
    let initial_hidden_states = hidden_states.clone();
    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let attention_scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));
    let zero = builder.constant(SP1Field::zero());
    let mut committed_parameters = Vec::new();
    let mut committed_hints = Vec::new();

    for layer in 0..shape.layers {
        let layer_started = Instant::now();
        let (
            layer_norm_1_weight,
            layer_norm_1_bias,
            attention_qkv_weight,
            attention_projection_weight,
            layer_norm_2_weight,
            layer_norm_2_bias,
            mlp_expansion_weight,
            mlp_projection_weight,
            hints,
        ) = match &real_data {
            Some(data) => {
                let layer_data = &data.layers[layer];
                (
                    constants(&mut builder, &layer_data.layer_norm_1_weight),
                    constants(&mut builder, &layer_data.layer_norm_1_bias),
                    constants(&mut builder, &layer_data.attention_qkv_weight),
                    constants(&mut builder, &layer_data.attention_projection_weight),
                    constants(&mut builder, &layer_data.layer_norm_2_weight),
                    constants(&mut builder, &layer_data.layer_norm_2_bias),
                    constants(&mut builder, &layer_data.mlp_expansion_weight),
                    constants(&mut builder, &layer_data.mlp_projection_weight),
                    constants(&mut builder, &layer_data.attention_max_hints),
                )
            }
            None => (
                repeated_constants(&mut builder, 0x3f80, shape.hidden_size),
                repeated_constants(&mut builder, 0x0000, shape.hidden_size),
                repeated_constants(&mut builder, 0x3f80, shape.hidden_size * 3 * shape.hidden_size),
                repeated_constants(&mut builder, 0x3f80, shape.hidden_size * shape.hidden_size),
                repeated_constants(&mut builder, 0x3f80, shape.hidden_size),
                repeated_constants(&mut builder, 0x0000, shape.hidden_size),
                repeated_constants(&mut builder, 0x3f80, shape.hidden_size * shape.linear_size),
                repeated_constants(&mut builder, 0x3f80, shape.linear_size * shape.hidden_size),
                vec![zero; shape.sequence_length * shape.num_heads],
            ),
        };
        if arguments.block.is_some() {
            committed_parameters.extend_from_slice(&layer_norm_1_weight);
            committed_parameters.extend_from_slice(&layer_norm_1_bias);
            committed_parameters.extend_from_slice(&attention_qkv_weight);
            committed_parameters.extend_from_slice(&attention_projection_weight);
            committed_parameters.extend_from_slice(&layer_norm_2_weight);
            committed_parameters.extend_from_slice(&layer_norm_2_bias);
            committed_parameters.extend_from_slice(&mlp_expansion_weight);
            committed_parameters.extend_from_slice(&mlp_projection_weight);
            committed_hints.extend_from_slice(&hints);
        }
        let params = Bf16ZkGptBlockParams {
            layer_norm_1_weight: &layer_norm_1_weight,
            layer_norm_1_bias: &layer_norm_1_bias,
            attention_qkv_weight: &attention_qkv_weight,
            attention_projection_weight: &attention_projection_weight,
            layer_norm_2_weight: &layer_norm_2_weight,
            layer_norm_2_bias: &layer_norm_2_bias,
            mlp_expansion_weight: &mlp_expansion_weight,
            mlp_projection_weight: &mlp_projection_weight,
            layer_norm_epsilon: epsilon,
            attention_scale,
            num_heads: shape.num_heads,
        };
        hidden_states = builder.bf16_zkgpt_block(&hidden_states, &hints, &params);
        println!("built layer={layer} elapsed={:.3}s", layer_started.elapsed().as_secs_f64());
    }
    if arguments.synthetic {
        for &value in &hidden_states {
            builder.assert_felt_eq(value, SP1Field::zero());
        }
    }
    if arguments.block.is_some() {
        commit_block_transcript(
            &mut builder,
            arguments,
            upstream,
            &initial_hidden_states,
            &committed_parameters,
            &committed_hints,
            &hidden_states,
        );
        for &value in &hidden_states {
            builder.print_f(value);
        }
    } else {
        commit_bf16_output(&mut builder, &hidden_states);
    }
    let block = builder.into_root_block();
    println!(
        "built: ir_ops={} elapsed={:.3}s",
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
    let print_path = arguments
        .block
        .map(|block| arguments.output_dir.join(format!("{}.output.txt", block_stem(block))));
    let execute_started = Instant::now();
    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    if let Some(path) = &print_path {
        executor.debug_stdout = Box::new(
            File::create(path)
                .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
        );
    }
    executor.run().unwrap();
    println!("executed: elapsed={:.3}s", execute_started.elapsed().as_secs_f64());
    assert_eq!(executor.record.bf16_mul_events.len() as u128, expected.mul);
    assert_eq!(executor.record.bf16_add_sub_events.len() as u128, expected.add_sub);
    assert_eq!(executor.record.bf16_unary_events.len() as u128, expected.unary);
    assert_eq!(executor.record.bf16_div_events.len() as u128, expected.div);
    println!(
        "verified events: mul={} add_sub={} unary={} div={}",
        executor.record.bf16_mul_events.len(),
        executor.record.bf16_add_sub_events.len(),
        executor.record.bf16_unary_events.len(),
        executor.record.bf16_div_events.len(),
    );
    let public_digest = executor.record.public_values.digest;
    println!(
        "public {} digest: {}",
        if arguments.block.is_some() { "Block transcript" } else { "output" },
        digest_hex(&public_digest),
    );
    let proof_record =
        (arguments.mode == Mode::Prove).then(|| std::mem::take(&mut executor.record));
    drop(executor);

    let block_artifacts = arguments.block.map(|block| {
        let output = read_printed_bf16(print_path.as_ref().unwrap());
        assert_eq!(
            output.len(),
            shape.sequence_length * shape.hidden_size,
            "Block circuit output shape mismatch"
        );
        let (raw_parameters, raw_hints) = match &real_data {
            Some(data) => {
                (raw_block_parameters(&data.layers[0]), data.layers[0].attention_max_hints.clone())
            }
            None => {
                let mut parameters = Vec::new();
                parameters.extend(vec![0x3f80; shape.hidden_size]);
                parameters.extend(vec![0x0000; shape.hidden_size]);
                parameters.extend(vec![0x3f80; shape.hidden_size * 3 * shape.hidden_size]);
                parameters.extend(vec![0x3f80; shape.hidden_size * shape.hidden_size]);
                parameters.extend(vec![0x3f80; shape.hidden_size]);
                parameters.extend(vec![0x0000; shape.hidden_size]);
                parameters.extend(vec![0x3f80; shape.hidden_size * shape.linear_size]);
                parameters.extend(vec![0x3f80; shape.linear_size * shape.hidden_size]);
                (parameters, vec![0x0000; shape.sequence_length * shape.num_heads])
            }
        };
        let commitments = compute_host_block_commitments(
            arguments,
            upstream,
            &raw_input,
            &raw_parameters,
            &raw_hints,
            &output,
        );
        assert_eq!(
            public_digest, commitments.transcript,
            "circuit Block transcript differs from the independent host commitment"
        );
        let private_output_file = write_private_block_output(&arguments.output_dir, block, &output);
        println!("Block commitment chain verified: block={block}");
        (commitments, private_output_file)
    });

    if let Some(record) = proof_record {
        let artifact_stem =
            arguments.block.map(block_stem).unwrap_or_else(|| "zkgpt_like".to_owned());
        let proof = prove_and_verify(program, record, &arguments.output_dir, &artifact_stem).await;
        if let Some((commitments, private_output_file)) = block_artifacts {
            write_block_manifest(arguments, commitments, &private_output_file, &proof);
        }
    }
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let events = EventCounts::for_shape(arguments.shape);
    println!("mode={:?}", arguments.mode);
    if let Some(block) = arguments.block {
        println!("proof partition: one Block per proof, selected_block={block}");
    }
    print_estimate(arguments.shape, events);
    if arguments.mode == Mode::CheckData {
        assert!(!arguments.synthetic, "--check-data requires a real BF16 fixture");
        let data = load_real_data(
            &arguments.data_dir,
            arguments.shape,
            arguments.block.unwrap_or(0),
            None,
        );
        let parameter_values = data
            .layers
            .iter()
            .map(|layer| {
                layer.layer_norm_1_weight.len()
                    + layer.layer_norm_1_bias.len()
                    + layer.attention_qkv_weight.len()
                    + layer.attention_projection_weight.len()
                    + layer.layer_norm_2_weight.len()
                    + layer.layer_norm_2_bias.len()
                    + layer.mlp_expansion_weight.len()
                    + layer.mlp_projection_weight.len()
            })
            .sum::<usize>();
        println!("real BF16 data valid: parameters={parameter_values}");
    } else if arguments.mode != Mode::Estimate {
        materialize(&arguments, events).await;
    }
}
