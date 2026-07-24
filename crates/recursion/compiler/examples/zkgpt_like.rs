#![allow(clippy::print_stdout)]

use std::{
    collections::VecDeque,
    fmt::Write as _,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_basefold::FriConfig;
use slop_challenger::IopCtx;
use slop_symmetric::Permutation;
use sp1_hypercube::{
    air::MachineAir, inner_perm, prover::simple_prover, HashableKey, MachineProof,
    MachineVerifyingKey, SP1PcsProofInner, ShardProof, ShardVerifier,
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
    Bf16AddSubEvent, Bf16DivEvent, Bf16MulEvent, Bf16UnaryEvent, Block, EventRange,
    ExecutionRecord, Executor, RecursionAirEventCount, RecursionEventRanges, RecursionProgram,
    DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH,
};
use sp1_recursion_machine::{
    chips::bf16::{NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW},
    chips::global_memory_boundary::prepare_event_shard_boundary,
    sharding::{estimate_verillm_trace, plan_event_shards, ShardLimits},
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
const BLOCK_PROTOCOL_VERSION: u32 = 2;
const EVENT_BLOCK_RECURSION_PROTOCOL_VERSION: u32 = 2;
const EVENT_BLOCK_RECURSION_STAGE: &str = "zkgpt_block_shard_recursion";
const EVENT_SHARD_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_4557;
const EVENT_SHARD_PROTOCOL_VERSION: u32 = 3;

type StoredProof = MachineProof<SP1GlobalContext, SP1PcsProofInner>;
type StoredVerifyingKey = MachineVerifyingKey<SP1GlobalContext>;
type StoredShardProof = ShardProof<SP1GlobalContext, SP1PcsProofInner>;
type StoredDigest = <SP1GlobalContext as IopCtx>::Digest;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Mode {
    #[default]
    Estimate,
    Build,
    Compile,
    PlanShards,
    Execute,
    Prove,
    ProveShards,
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
        let layer_norm_mul = sequence_length * (4 * hidden_size + 4);
        let layer_norm_add_sub = sequence_length * (8 * hidden_size - 2);
        let layer_norm_unary = sequence_length * (2 * hidden_size + 2);

        // QKV, attention projection, MLP expansion, and MLP projection, all without bias.
        let linear_mul =
            sequence_length * (3 * hidden_size * linear_size + hidden_size * hidden_size);
        let linear_add_sub = sequence_length
            * (3 * hidden_size * linear_size + hidden_size * hidden_size
                - 2 * linear_size
                - 2 * hidden_size);

        // Causal QK scores, max subtraction, softmax accumulation, and probability-weighted V.
        let attention_mul = causal_scores * (2 * head_dimension + 2);
        let attention_add_sub = causal_scores * head_dimension
            + (num_heads + hidden_size) * sequence_length * (sequence_length - 1) / 2;
        let attention_unary = causal_scores;
        let attention_div = num_heads * sequence_length;

        let per_layer = Self {
            mul: layer_norm_mul + linear_mul + attention_mul,
            add_sub: layer_norm_add_sub + linear_add_sub + attention_add_sub,
            unary: layer_norm_unary + attention_unary + sequence_length * linear_size,
            div: attention_div,
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

        let layer_norm_mul = sequence_length * (4 * hidden_size + 4);
        let layer_norm_add_sub = sequence_length * (8 * hidden_size - 2);
        let layer_norm_unary = sequence_length * (2 * hidden_size + 2);
        let linear_mul =
            sequence_length * (4 * hidden_size * hidden_size + 2 * hidden_size * inner_size);
        // Standard GPT-2 adds a bias after every dot product and has two residual additions.
        let linear_and_residual_add_sub = linear_mul + 2 * sequence_length * hidden_size;
        let attention_mul = causal_scores * (2 * head_dimension + 2);
        let attention_add_sub = causal_scores * head_dimension
            + (num_heads + hidden_size) * sequence_length * (sequence_length - 1) / 2;

        let per_layer = Self {
            mul: layer_norm_mul + linear_mul + attention_mul,
            add_sub: layer_norm_add_sub + linear_and_residual_add_sub + attention_add_sub,
            unary: layer_norm_unary + causal_scores + sequence_length * inner_size,
            div: num_heads * sequence_length,
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
    shard_limits: ShardLimits,
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

#[derive(Debug)]
struct ProofArtifacts {
    proof_file: String,
    proof_bytes: u64,
    verifying_key_file: String,
    verifying_key_bytes: u64,
}

#[derive(Debug, Clone)]
struct EventShardBatchBinding {
    ranges: RecursionEventRanges,
    verifying_key: StoredVerifyingKey,
    main_commitment: StoredDigest,
    real_heights: Vec<(String, usize)>,
}

fn event_count_fields(counts: RecursionAirEventCount) -> [usize; 9] {
    [
        counts.mem_const_events,
        counts.mem_var_events,
        counts.base_alu_events,
        counts.poseidon2_wide_events,
        counts.bf16_mul_events,
        counts.bf16_unary_events,
        counts.bf16_div_events,
        counts.bf16_add_sub_events,
        counts.commit_pv_hash_events,
    ]
}

fn event_range_fields(ranges: RecursionEventRanges) -> [usize; 18] {
    [
        ranges.mem_const.start,
        ranges.mem_const.end,
        ranges.mem_var.start,
        ranges.mem_var.end,
        ranges.base_alu.start,
        ranges.base_alu.end,
        ranges.poseidon2_wide.start,
        ranges.poseidon2_wide.end,
        ranges.bf16_mul.start,
        ranges.bf16_mul.end,
        ranges.bf16_unary.start,
        ranges.bf16_unary.end,
        ranges.bf16_div.start,
        ranges.bf16_div.end,
        ranges.bf16_add_sub.start,
        ranges.bf16_add_sub.end,
        ranges.commit_pv_hash.start,
        ranges.commit_pv_hash.end,
    ]
}

const fn bisect_event_range(range: EventRange) -> [EventRange; 2] {
    let middle = range.start + range.len() / 2;
    [EventRange { start: range.start, end: middle }, EventRange { start: middle, end: range.end }]
}

fn bisect_event_ranges(ranges: RecursionEventRanges) -> [RecursionEventRanges; 2] {
    let mem_const = bisect_event_range(ranges.mem_const);
    let mem_var = bisect_event_range(ranges.mem_var);
    let base_alu = bisect_event_range(ranges.base_alu);
    let poseidon2_wide = bisect_event_range(ranges.poseidon2_wide);
    let bf16_mul = bisect_event_range(ranges.bf16_mul);
    let bf16_unary = bisect_event_range(ranges.bf16_unary);
    let bf16_div = bisect_event_range(ranges.bf16_div);
    let bf16_add_sub = bisect_event_range(ranges.bf16_add_sub);
    let commit_pv_hash = bisect_event_range(ranges.commit_pv_hash);
    [
        RecursionEventRanges {
            mem_const: mem_const[0],
            mem_var: mem_var[0],
            base_alu: base_alu[0],
            poseidon2_wide: poseidon2_wide[0],
            bf16_mul: bf16_mul[0],
            bf16_unary: bf16_unary[0],
            bf16_div: bf16_div[0],
            bf16_add_sub: bf16_add_sub[0],
            commit_pv_hash: commit_pv_hash[0],
        },
        RecursionEventRanges {
            mem_const: mem_const[1],
            mem_var: mem_var[1],
            base_alu: base_alu[1],
            poseidon2_wide: poseidon2_wide[1],
            bf16_mul: bf16_mul[1],
            bf16_unary: bf16_unary[1],
            bf16_div: bf16_div[1],
            bf16_add_sub: bf16_add_sub[1],
            commit_pv_hash: commit_pv_hash[1],
        },
    ]
}

fn commit_fields(domain: u32, values: &[SP1Field]) -> [SP1Field; DIGEST_SIZE] {
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

fn event_shard_transcript(
    shape: Shape,
    block: Option<usize>,
    global_counts: RecursionAirEventCount,
    bindings: &[EventShardBatchBinding],
) -> [SP1Field; DIGEST_SIZE] {
    let mut fields = [
        EVENT_SHARD_PROTOCOL_VERSION,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.linear_size as u32,
        block.unwrap_or_default() as u32,
        bindings.len() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(event_count_fields(global_counts).map(SP1Field::from_canonical_usize));
    for (index, binding) in bindings.iter().enumerate() {
        fields.push(SP1Field::from_canonical_usize(index));
        fields.extend(event_range_fields(binding.ranges).map(SP1Field::from_canonical_usize));
        fields.extend(binding.verifying_key.hash_koalabear());
        fields.extend(binding.main_commitment);
        fields.push(SP1Field::from_canonical_usize(binding.real_heights.len()));
        for (name, height) in &binding.real_heights {
            fields.push(SP1Field::from_canonical_usize(name.len()));
            fields.extend(name.bytes().map(SP1Field::from_canonical_u8));
            fields.push(SP1Field::from_canonical_usize(*height));
        }
    }
    commit_fields(EVENT_SHARD_TRANSCRIPT_DOMAIN, &fields)
}

fn write_event_shard_proof(
    output_dir: &Path,
    artifact_stem: &str,
    verifying_key: &StoredVerifyingKey,
    shard_proof: StoredShardProof,
) -> ProofArtifacts {
    fs::create_dir_all(output_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", output_dir.display()));
    let proof_file = format!("{artifact_stem}.proof.bin");
    let verifying_key_file = format!("{artifact_stem}.vk.bin");
    let proof_path = output_dir.join(&proof_file);
    let verifying_key_path = output_dir.join(&verifying_key_file);
    let proof = MachineProof::from(vec![shard_proof]);
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&verifying_key_path).unwrap(), verifying_key).unwrap();
    ProofArtifacts {
        proof_file,
        proof_bytes: fs::metadata(&proof_path).unwrap().len(),
        verifying_key_file,
        verifying_key_bytes: fs::metadata(&verifying_key_path).unwrap().len(),
    }
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

fn digest_hex<const N: usize>(digest: &[SP1Field; N]) -> String {
    digest
        .iter()
        .map(|value| format!("{:08X}", value.as_canonical_u32()))
        .collect::<Vec<_>>()
        .join(":")
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
    let mut shard_limits = ShardLimits::core_style();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--estimate") => mode = Mode::Estimate,
            Some("--build") => mode = Mode::Build,
            Some("--compile") => mode = Mode::Compile,
            Some("--plan-shards") => mode = Mode::PlanShards,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--prove-shards") => mode = Mode::ProveShards,
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
            Some("--max-shard-log-rows") => {
                shard_limits.max_log_rows = parse_usize(arguments.next(), "--max-shard-log-rows")
            }
            Some("--max-shard-area") => {
                shard_limits.max_trace_area = parse_usize(arguments.next(), "--max-shard-area")
            }
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
        shard_limits,
    }
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

fn json_usize_field(path: &Path, key: &str) -> usize {
    json_string_field(path, key)
        .parse()
        .unwrap_or_else(|error| panic!("{} has invalid {key}: {error}", path.display()))
}

fn block_stem(block: usize) -> String {
    format!("zkgpt_block_l{block:02}")
}

fn verify_and_load_previous_block(arguments: &Arguments) -> Vec<u16> {
    type A = RecursionAir<SP1Field, 3, 2>;

    let block = arguments.block.expect("previous Block verification requires --block");
    assert!(block > 0, "Block 0 has no previous Block");
    let previous = block - 1;
    let directory = arguments
        .previous_block_dir
        .as_ref()
        .expect("Block 1 or later requires --previous-block-dir");
    let stem = block_stem(previous);
    let recursion_manifest =
        directory.join(format!("zkgpt_block_{previous:02}_shard_recursion.manifest.json"));
    let legacy_manifest = directory.join(format!("{stem}.manifest.json"));
    let (manifest_path, machine) = if recursion_manifest.is_file() {
        assert_eq!(
            json_string_field(&recursion_manifest, "stage"),
            EVENT_BLOCK_RECURSION_STAGE,
            "previous Block shard-recursion manifest has the wrong stage"
        );
        assert_eq!(
            json_usize_field(&recursion_manifest, "version"),
            EVENT_BLOCK_RECURSION_PROTOCOL_VERSION as usize,
            "previous Block shard-recursion manifest has the wrong version"
        );
        (recursion_manifest, A::compress_machine())
    } else {
        assert_eq!(
            json_string_field(&legacy_manifest, "stage"),
            "zkgpt_block",
            "previous Block manifest has the wrong stage"
        );
        assert_eq!(
            json_usize_field(&legacy_manifest, "version"),
            BLOCK_PROTOCOL_VERSION as usize,
            "previous Block manifest uses the old explicit-commitment protocol"
        );
        (legacy_manifest, A::verillm_machine())
    };
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
    println!(
        "previous Block proof verified and private output loaded: block={previous}; \
         proof_stage={}; explicit output commitment disabled",
        json_string_field(&manifest_path, "stage")
    );
    hidden_states
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
        "privacy: public_input=MemoryConst private_parameters=Hint \
         attention_hints=Hint explicit_value_commitments=false"
    );
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

fn constants(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[u16],
) -> Vec<Felt<SP1Field>> {
    values.iter().map(|&value| builder.constant(SP1Field::from_canonical_u16(value))).collect()
}

fn private_bf16_hints(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    values: &[u16],
    witness_stream: &mut Vec<Block<SP1Field>>,
) -> Vec<Felt<SP1Field>> {
    let hinted = builder.hint_felts_v2(values.len());
    witness_stream
        .extend(values.iter().map(|&value| Block::from(SP1Field::from_canonical_u16(value))));
    hinted
}

fn repeated_private_bf16_hints(
    builder: &mut Builder<sp1_recursion_compiler::circuit::AsmConfig>,
    raw: u16,
    count: usize,
    witness_stream: &mut Vec<Block<SP1Field>>,
) -> Vec<Felt<SP1Field>> {
    let hinted = builder.hint_felts_v2(count);
    let value = SP1Field::from_canonical_u16(raw);
    witness_stream.extend((0..count).map(|_| Block::from(value)));
    hinted
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

fn write_block_manifest(arguments: &Arguments, private_output_file: &str, proof: &ProofArtifacts) {
    let block = arguments.block.expect("Block manifest requires --block");
    let stem = block_stem(block);
    let manifest_path = arguments.output_dir.join(format!("{stem}.manifest.json"));
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
    writeln!(manifest, "  \"explicit_value_commitments\": false,").unwrap();
    writeln!(manifest, "  \"public_input_storage\": \"memory_const\",").unwrap();
    writeln!(manifest, "  \"private_parameters_storage\": \"hint\",").unwrap();
    writeln!(manifest, "  \"private_auxiliary_hints_storage\": \"hint\",").unwrap();
    writeln!(manifest, "  \"private_values_binding\": \"execution_trace_commitment\",").unwrap();
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
    let block_input = match arguments.block {
        Some(0) | None => None,
        Some(_) => Some(verify_and_load_previous_block(arguments)),
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
    let mut witness_stream = Vec::<Block<SP1Field>>::new();
    let raw_input = match (&real_data, block_input) {
        (Some(data), _) => data.hidden_states.clone(),
        (None, Some(hidden_states)) => hidden_states,
        (None, None) => vec![0x0000; shape.sequence_length * shape.hidden_size],
    };
    let mut hidden_states = constants(&mut builder, &raw_input);
    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let attention_scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));

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
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.layer_norm_1_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.layer_norm_1_bias,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.attention_qkv_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.attention_projection_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.layer_norm_2_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.layer_norm_2_bias,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.mlp_expansion_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.mlp_projection_weight,
                        &mut witness_stream,
                    ),
                    private_bf16_hints(
                        &mut builder,
                        &layer_data.attention_max_hints,
                        &mut witness_stream,
                    ),
                )
            }
            None => (
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x0000,
                    shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.hidden_size * 3 * shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.hidden_size * shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x0000,
                    shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.hidden_size * shape.linear_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x3f80,
                    shape.linear_size * shape.hidden_size,
                    &mut witness_stream,
                ),
                repeated_private_bf16_hints(
                    &mut builder,
                    0x0000,
                    shape.sequence_length * shape.num_heads,
                    &mut witness_stream,
                ),
            ),
        };
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
        for &value in &hidden_states {
            builder.print_f(value);
        }
    }
    println!(
        "private witness stream: values={} (model parameters and attention hints)",
        witness_stream.len()
    );
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
    assert_eq!(program.event_counts.bf16_mul_events as u128, expected.mul);
    assert_eq!(program.event_counts.bf16_add_sub_events as u128, expected.add_sub);
    assert_eq!(program.event_counts.bf16_unary_events as u128, expected.unary);
    assert_eq!(program.event_counts.bf16_div_events as u128, expected.div);
    if arguments.mode == Mode::PlanShards {
        let plan = plan_event_shards(program.event_counts, arguments.shard_limits);
        println!(
            "global event shard plan: shards={} max_log_rows={} max_rows={} max_trace_area={}",
            plan.shards.len(),
            plan.limits.max_log_rows,
            plan.limits.max_rows(),
            plan.limits.max_trace_area
        );
        for shard in &plan.shards {
            let counts = shard.ranges.event_counts();
            println!(
                "shard {}/{}: events={{mem_const:{},mem_var:{},base_alu:{},poseidon2:{},\
                 mul:{},unary:{},div:{},add_sub:{}}} rows={{mem_const:{},mem_var:{},\
                 base_alu:{},poseidon2:{},lookup:{},mul:{},unary:{},div:{},add_sub:{}}} \
                 area={{preprocessed:{},main:{},total:{}}}",
                shard.index + 1,
                plan.shards.len(),
                counts.mem_const_events,
                counts.mem_var_events,
                counts.base_alu_events,
                counts.poseidon2_wide_events,
                counts.bf16_mul_events,
                counts.bf16_unary_events,
                counts.bf16_div_events,
                counts.bf16_add_sub_events,
                shard.estimate.memory_const_rows,
                shard.estimate.memory_var_rows,
                shard.estimate.base_alu_rows,
                shard.estimate.poseidon2_rows,
                shard.estimate.bf16_lookup_rows,
                shard.estimate.bf16_mul_rows,
                shard.estimate.bf16_unary_rows,
                shard.estimate.bf16_div_rows,
                shard.estimate.bf16_add_sub_rows,
                shard.estimate.preprocessed_area,
                shard.estimate.main_area,
                shard.estimate.total_area,
            );
        }
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }
    if arguments.mode == Mode::Compile {
        println!(
            "compiled program: instructions={} total_memory={}",
            program.inner.iter().count(),
            program.total_memory
        );
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }
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
    executor.witness_stream = witness_stream.into();
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
    println!("explicit input/output/parameter/hint commitments: disabled");
    let proof_record = matches!(arguments.mode, Mode::Prove | Mode::ProveShards)
        .then(|| std::mem::take(&mut executor.record));
    drop(executor);

    let block_artifacts = arguments.block.map(|block| {
        let output = read_printed_bf16(print_path.as_ref().unwrap());
        assert_eq!(
            output.len(),
            shape.sequence_length * shape.hidden_size,
            "Block circuit output shape mismatch"
        );
        let private_output_file = write_private_block_output(&arguments.output_dir, block, &output);
        println!("Block execution output materialized without an explicit output commitment");
        private_output_file
    });

    if arguments.mode == Mode::ProveShards {
        let record = proof_record.expect("sharded proving requires an execution record");
        let plan = plan_event_shards(program.event_counts, arguments.shard_limits);
        let shard_programs = plan
            .shards
            .iter()
            .map(|shard| Arc::new(program.with_event_ranges(shard.ranges)))
            .collect::<Vec<_>>();
        let shard_records = record.into_event_shards(shard_programs);

        type A = RecursionAir<SP1Field, 3, 2>;
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
        let prover = simple_prover(verifier);

        let initial_shard_count = plan.shards.len();
        let mut pending = plan
            .shards
            .iter()
            .zip(shard_records)
            .map(|(shard, record)| (shard.ranges, shard.estimate, record))
            .collect::<VecDeque<_>>();
        let mut artifacts = Vec::with_capacity(initial_shard_count);
        let mut bindings = Vec::with_capacity(initial_shard_count);
        let mut global_memory_accumulator = [SP1Field::zero(); 14];
        let mut total_boundary_events = 0usize;
        while let Some((ranges, estimate, mut record)) = pending.pop_front() {
            let boundary_started = Instant::now();
            let shard_accumulator = prepare_event_shard_boundary(&mut record);
            let boundary_events = record.global_memory_boundary_events.len();
            let boundary_rows = boundary_events.next_multiple_of(32).max(16);
            let exact_shape =
                prover.shape_from_record(&record).expect("VeriLLM machine has no shard shape");
            let exact_trace_area = exact_shape.preprocessed_area + exact_shape.main_area;
            let exact_max_rows = estimate.max_rows().max(boundary_rows);
            if exact_max_rows > plan.limits.max_rows()
                || exact_trace_area > plan.limits.max_trace_area
            {
                let largest_event_range =
                    event_count_fields(ranges.event_counts()).into_iter().max().unwrap_or_default();
                assert!(
                    largest_event_range > 1,
                    "one scalar event cannot fit after adding global memory boundaries: \
                     max_rows={exact_max_rows} trace_area={exact_trace_area}"
                );
                println!(
                    "refining oversized event shard before proving: ranges={:?} \
                     boundary_events={} max_rows={}/{} trace_area={}/{}",
                    ranges,
                    boundary_events,
                    exact_max_rows,
                    plan.limits.max_rows(),
                    exact_trace_area,
                    plan.limits.max_trace_area
                );
                record.global_memory_boundary_events.clear();
                let child_ranges = bisect_event_ranges(ranges);
                let child_programs =
                    child_ranges.map(|child| Arc::new(program.with_event_ranges(child))).to_vec();
                let child_records = record.into_event_shards(child_programs);
                for (child_ranges, child_record) in
                    child_ranges.into_iter().zip(child_records).rev()
                {
                    pending.push_front((
                        child_ranges,
                        estimate_verillm_trace(child_ranges.event_counts()),
                        child_record,
                    ));
                }
                continue;
            }

            let shard_index = artifacts.len();
            record.index =
                u32::try_from(shard_index).expect("too many independently provable event shards");
            global_memory_accumulator
                .iter_mut()
                .zip(shard_accumulator)
                .for_each(|(total, value)| *total += value);
            total_boundary_events += boundary_events;
            let artifact_stem = arguments
                .block
                .map(|block| format!("{}_s{:03}", block_stem(block), shard_index))
                .unwrap_or_else(|| format!("zkgpt_like_s{:03}", shard_index));
            println!(
                "proving independent event shard {}: pending={} initial_plan={} \
                 planned_max_rows={} planned_trace_area={} boundary_events={} \
                 boundary_rows={} exact_trace_area={}",
                shard_index + 1,
                pending.len(),
                initial_shard_count,
                estimate.max_rows(),
                estimate.total_area,
                boundary_events,
                boundary_rows,
                exact_trace_area
            );
            println!(
                "prepared shard memory boundary: events={} elapsed={:.3}s",
                boundary_events,
                boundary_started.elapsed().as_secs_f64()
            );
            let setup_started = Instant::now();
            let (pk, verifying_key) = prover.setup(record.program.clone()).await;
            let pk = unsafe { pk.into_inner() };
            println!("shard setup: elapsed={:.3}s", setup_started.elapsed().as_secs_f64());

            let prove_started = Instant::now();
            let shard_proof = prover.prove_shard(pk, record).await;
            let real_heights = shard_proof
                .opened_values
                .chips
                .iter()
                .map(|(name, values)| {
                    (
                        name.clone(),
                        values.degree.bit_string_evaluation().as_canonical_u32() as usize,
                    )
                })
                .collect::<Vec<_>>();
            println!(
                "shard proof generated: elapsed={:.3}s",
                prove_started.elapsed().as_secs_f64()
            );

            let verify_started = Instant::now();
            prover
                .verify(&verifying_key, &MachineProof::from(vec![shard_proof.clone()]))
                .expect("independent event shard proof must verify");
            println!(
                "shard proof verified: elapsed={:.3}s",
                verify_started.elapsed().as_secs_f64()
            );
            bindings.push(EventShardBatchBinding {
                ranges,
                verifying_key: verifying_key.clone(),
                main_commitment: shard_proof.main_commitment,
                real_heights,
            });
            let proof = write_event_shard_proof(
                &arguments.output_dir,
                &artifact_stem,
                &verifying_key,
                shard_proof,
            );
            let mut artifact = String::new();
            writeln!(artifact, "    {{").unwrap();
            writeln!(artifact, "      \"index\": {shard_index},").unwrap();
            writeln!(artifact, "      \"ranges\": {{").unwrap();
            writeln!(
                artifact,
                "        \"mem_const\": [{}, {}],",
                ranges.mem_const.start, ranges.mem_const.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"mem_var\": [{}, {}],",
                ranges.mem_var.start, ranges.mem_var.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"base_alu\": [{}, {}],",
                ranges.base_alu.start, ranges.base_alu.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"poseidon2_wide\": [{}, {}],",
                ranges.poseidon2_wide.start, ranges.poseidon2_wide.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"bf16_mul\": [{}, {}],",
                ranges.bf16_mul.start, ranges.bf16_mul.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"bf16_unary\": [{}, {}],",
                ranges.bf16_unary.start, ranges.bf16_unary.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"bf16_div\": [{}, {}],",
                ranges.bf16_div.start, ranges.bf16_div.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"bf16_add_sub\": [{}, {}],",
                ranges.bf16_add_sub.start, ranges.bf16_add_sub.end
            )
            .unwrap();
            writeln!(
                artifact,
                "        \"commit_pv_hash\": [{}, {}]",
                ranges.commit_pv_hash.start, ranges.commit_pv_hash.end
            )
            .unwrap();
            writeln!(artifact, "      }},").unwrap();
            writeln!(artifact, "      \"max_rows\": {},", estimate.max_rows()).unwrap();
            writeln!(artifact, "      \"preprocessed_area\": {},", estimate.preprocessed_area)
                .unwrap();
            writeln!(artifact, "      \"main_area\": {},", estimate.main_area).unwrap();
            writeln!(artifact, "      \"trace_area\": {},", estimate.total_area).unwrap();
            writeln!(artifact, "      \"global_memory_boundary_events\": {boundary_events},")
                .unwrap();
            writeln!(artifact, "      \"global_memory_boundary_rows\": {boundary_rows},").unwrap();
            writeln!(
                artifact,
                "      \"exact_preprocessed_area\": {},",
                exact_shape.preprocessed_area
            )
            .unwrap();
            writeln!(artifact, "      \"exact_main_area\": {},", exact_shape.main_area).unwrap();
            writeln!(artifact, "      \"exact_trace_area\": {exact_trace_area},").unwrap();
            writeln!(artifact, "      \"proof_file\": \"{}\",", proof.proof_file).unwrap();
            writeln!(artifact, "      \"proof_bytes\": {},", proof.proof_bytes).unwrap();
            writeln!(artifact, "      \"verifying_key_file\": \"{}\",", proof.verifying_key_file)
                .unwrap();
            writeln!(artifact, "      \"verifying_key_bytes\": {}", proof.verifying_key_bytes)
                .unwrap();
            write!(artifact, "    }}").unwrap();
            artifacts.push(artifact);
        }
        assert!(
            global_memory_accumulator.iter().all(Field::is_zero),
            "event-shard global memory accumulators do not close"
        );
        println!(
            "independent event shard proofs closed locally; global memory accumulator closed: \
             boundary_events={total_boundary_events}"
        );
        let block_transcript =
            event_shard_transcript(shape, arguments.block, plan.global_counts, &bindings);
        let manifest_path = arguments.output_dir.join("zkgpt_event_shards.manifest.json");
        let mut manifest = String::new();
        writeln!(manifest, "{{").unwrap();
        writeln!(manifest, "  \"version\": {EVENT_SHARD_PROTOCOL_VERSION},").unwrap();
        writeln!(manifest, "  \"stage\": \"zkgpt_event_shards\",").unwrap();
        writeln!(manifest, "  \"lookup_protocol\": \"independent_global_memory_accumulator\",")
            .unwrap();
        writeln!(manifest, "  \"lookup_batch_closed\": true,").unwrap();
        writeln!(
            manifest,
            "  \"transcript_binding\": \
             \"global_event_counts_ordered_ranges_verifying_keys_main_trace_commitments_real_heights\","
        )
        .unwrap();
        writeln!(manifest, "  \"global_memory_accumulator\": \"poseidon_vector_14\",").unwrap();
        writeln!(manifest, "  \"sequence_length\": {},", shape.sequence_length).unwrap();
        writeln!(manifest, "  \"hidden_size\": {},", shape.hidden_size).unwrap();
        writeln!(manifest, "  \"num_heads\": {},", shape.num_heads).unwrap();
        writeln!(manifest, "  \"linear_size\": {},", shape.linear_size).unwrap();
        writeln!(
            manifest,
            "  \"block_transcript_commitment\": \"{}\",",
            digest_hex(&block_transcript)
        )
        .unwrap();
        writeln!(manifest, "  \"global_event_counts\": {{").unwrap();
        writeln!(manifest, "    \"mem_const\": {},", plan.global_counts.mem_const_events).unwrap();
        writeln!(manifest, "    \"mem_var\": {},", plan.global_counts.mem_var_events).unwrap();
        writeln!(manifest, "    \"base_alu\": {},", plan.global_counts.base_alu_events).unwrap();
        writeln!(manifest, "    \"poseidon2_wide\": {},", plan.global_counts.poseidon2_wide_events)
            .unwrap();
        writeln!(manifest, "    \"bf16_mul\": {},", plan.global_counts.bf16_mul_events).unwrap();
        writeln!(manifest, "    \"bf16_unary\": {},", plan.global_counts.bf16_unary_events)
            .unwrap();
        writeln!(manifest, "    \"bf16_div\": {},", plan.global_counts.bf16_div_events).unwrap();
        writeln!(manifest, "    \"bf16_add_sub\": {},", plan.global_counts.bf16_add_sub_events)
            .unwrap();
        writeln!(manifest, "    \"commit_pv_hash\": {}", plan.global_counts.commit_pv_hash_events)
            .unwrap();
        writeln!(manifest, "  }},").unwrap();
        match arguments.block {
            Some(block) => writeln!(manifest, "  \"block\": {block},").unwrap(),
            None => writeln!(manifest, "  \"block\": null,").unwrap(),
        }
        match &block_artifacts {
            Some(file) => writeln!(manifest, "  \"private_output_file\": \"{file}\",").unwrap(),
            None => writeln!(manifest, "  \"private_output_file\": null,").unwrap(),
        }
        writeln!(manifest, "  \"shards\": {},", artifacts.len()).unwrap();
        writeln!(manifest, "  \"max_shard_log_rows\": {},", plan.limits.max_log_rows).unwrap();
        writeln!(manifest, "  \"max_shard_area\": {},", plan.limits.max_trace_area).unwrap();
        writeln!(manifest, "  \"artifacts\": [").unwrap();
        writeln!(manifest, "{}", artifacts.join(",\n")).unwrap();
        writeln!(manifest, "  ]").unwrap();
        writeln!(manifest, "}}").unwrap();
        fs::write(&manifest_path, manifest)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
        println!("event shard manifest: {}", manifest_path.display());
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    if let Some(record) = proof_record {
        let artifact_stem =
            arguments.block.map(block_stem).unwrap_or_else(|| "zkgpt_like".to_owned());
        let proof = prove_and_verify(program, record, &arguments.output_dir, &artifact_stem).await;
        if let Some(private_output_file) = block_artifacts {
            write_block_manifest(arguments, &private_output_file, &proof);
        }
    }
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    sp1_core_machine::utils::setup_logger();
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
