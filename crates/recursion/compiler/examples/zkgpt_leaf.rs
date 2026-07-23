#![allow(clippy::print_stdout)]

use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Write as _,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(not(target_os = "linux"))]
use std::process::Command;

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
    prelude::{Builder, Felt, IrIter},
};
use sp1_recursion_executor::{
    Bf16AddSubEvent, Bf16AddSubOpcode, Bf16AddSubWitness, Bf16DivEvent, Bf16DivWitness,
    Bf16MulEvent, Bf16MulWitness, Bf16UnaryEvent, Bf16UnaryOpcode, Bf16UnaryWitness,
    ExecutionRecord, Executor, RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE,
    PERMUTATION_WIDTH, RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::{
    chips::bf16::{NUM_BF16_ADD_SUB_EVENTS_PER_ROW, NUM_BF16_MUL_EVENTS_PER_ROW},
    sharding::{
        plan_uniform_natural_shards, Bf16EventsPerUnit, ShardLimits, DEFAULT_MAX_TRACE_AREA,
    },
    RecursionAir,
};

const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_NUM_HEADS: usize = 12;
const DEFAULT_LAYER: usize = 0;
const DEFAULT_HEAD: usize = 0;

const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;
const GPT2_ATTENTION_SCALE: u16 = 0x3e00;

const LEAF_VERSION: u32 = 1;
const LEAF_STAGE_QKV_ATTENTION: u32 = 1;
const LEAF_STAGE_CHAINED_QKV_ATTENTION: u32 = 10;
const DOMAIN_INPUT: u32 = 0x1001;
const DOMAIN_PARAMETERS: u32 = 0x1002;
const DOMAIN_HINTS: u32 = 0x1003;
const DOMAIN_OUTPUT: u32 = 0x1004;
const DOMAIN_TRANSCRIPT: u32 = 0x1005;
const DOMAIN_ATTENTION_GROUP_PARAMETERS: u32 = 0x1101;
const DOMAIN_ATTENTION_GROUP_HINTS: u32 = 0x1102;
const DOMAIN_ATTENTION_GROUP_OUTPUT: u32 = 0x1103;
const DOMAIN_ATTENTION_GROUP_TRANSCRIPT: u32 = 0x1104;
const ATTENTION_GROUP_STAGE: u32 = 2;
const ATTENTION_CHAINED_GROUP_STAGE: u32 = 11;

// These values must remain identical to `zkgpt_mlp_projection_join`.
const DOMAIN_MLP_PROJECTION_GROUP_OUTPUT: u32 = 0x1512;
const DOMAIN_SYNTHETIC_UPSTREAM_TRANSCRIPT: u32 = 0x15ff;
const PREVIOUS_BLOCK_JOIN_MAX_LOG_ROWS: usize = 16;

const PROOF_LOG_BLOWUP: usize = 1;
const FULL_LEAF_MAX_LOG_ROWS: usize = 19;
const SMALL_LEAF_MAX_LOG_ROWS: usize = 16;

type LeafBuilder = Builder<AsmConfig>;
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
    num_heads: usize,
}

impl Shape {
    fn full() -> Self {
        Self {
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            num_heads: DEFAULT_NUM_HEADS,
        }
    }

    fn small() -> Self {
        Self { sequence_length: 2, hidden_size: 8, num_heads: 2 }
    }

    fn validate(self) {
        assert!(self.sequence_length > 0, "leaf sequence length must be nonzero");
        assert!(self.hidden_size > 0, "leaf hidden size must be nonzero");
        assert!(self.num_heads > 0, "leaf head count must be nonzero");
        assert_eq!(
            self.hidden_size % self.num_heads,
            0,
            "leaf hidden size must be divisible by its head count"
        );
    }

    fn head_dimension(self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn qkv_head_width(self) -> usize {
        3 * self.head_dimension()
    }

    fn max_log_rows(self) -> usize {
        if self.hidden_size == DEFAULT_HIDDEN_SIZE
            && self.sequence_length == DEFAULT_SEQUENCE_LENGTH
            && self.num_heads == DEFAULT_NUM_HEADS
        {
            FULL_LEAF_MAX_LOG_ROWS
        } else {
            SMALL_LEAF_MAX_LOG_ROWS
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
        let sequence_length = shape.sequence_length;
        let hidden_size = shape.hidden_size;
        let head_dimension = shape.head_dimension();
        let qkv_head_width = shape.qkv_head_width();
        let causal_scores = sequence_length * (sequence_length + 1) / 2;
        let strict_causal_scores = sequence_length * (sequence_length - 1) / 2;

        let layer_norm_mul = sequence_length * 2 * hidden_size;
        let layer_norm_add_sub = sequence_length * (4 * hidden_size - 1);
        let layer_norm_unary = sequence_length * (hidden_size + 1);
        let layer_norm_div = sequence_length * 2;

        let qkv_mul = sequence_length * hidden_size * qkv_head_width;
        let qkv_add_sub = sequence_length * qkv_head_width * (hidden_size - 1);

        let attention_mul = causal_scores * (2 * head_dimension + 1);
        let attention_add_sub =
            causal_scores * head_dimension + (head_dimension + 1) * strict_causal_scores;

        Self {
            mul: layer_norm_mul + qkv_mul + attention_mul,
            add_sub: layer_norm_add_sub + qkv_add_sub + attention_add_sub,
            unary: layer_norm_unary + causal_scores,
            div: layer_norm_div + causal_scores,
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
    head: usize,
    all_heads: bool,
    data_dir: PathBuf,
    output_dir: PathBuf,
    previous_block_dir: Option<PathBuf>,
    previous_block_join_dir: Option<PathBuf>,
    expected_input_digest: Option<Digest>,
    synthetic: bool,
}

impl Arguments {
    fn chained(&self) -> bool {
        self.layer > 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UpstreamCommitments {
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Debug)]
struct LeafInput {
    hidden_states: Vec<u16>,
    upstream: Option<UpstreamCommitments>,
}

#[derive(Clone, Debug)]
struct LeafData {
    hidden_states: Vec<u16>,
    layer_norm_weight: Vec<u16>,
    layer_norm_bias: Vec<u16>,
    qkv_head_weight: Vec<u16>,
    attention_max_hints: Vec<u16>,
    upstream: Option<UpstreamCommitments>,
}

#[derive(Clone, Copy, Debug)]
struct LeafCommitments {
    upstream: Option<Digest>,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Copy, Debug)]
struct LeafTranscriptInputs {
    upstream: Option<Digest>,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
}

#[derive(Clone, Copy, Debug)]
struct AttentionGroupCommitments {
    upstream: Option<Digest>,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
    transcript: Digest,
}

struct HeadExecution {
    head: usize,
    output: Vec<u16>,
    commitments: LeafCommitments,
    record: ExecutionRecord<SP1Field>,
    load_seconds: f64,
    reference_seconds: f64,
    execute_seconds: f64,
}

#[derive(Debug)]
struct HeadSummary {
    head: usize,
    commitments: LeafCommitments,
    load_seconds: f64,
    reference_seconds: f64,
    execute_seconds: f64,
    prove_seconds: Option<f64>,
    verify_seconds: Option<f64>,
    sampled_peak_rss_kib: Option<u64>,
    proof_file: Option<String>,
    proof_bytes: Option<u64>,
}

#[derive(Debug)]
struct CompletedHead {
    output: Vec<u16>,
    summary: HeadSummary,
}

#[derive(Clone, Debug, Default)]
struct BatchMetrics {
    build_seconds: f64,
    compile_seconds: f64,
    setup_seconds: Option<f64>,
    setup_sampled_peak_rss_kib: Option<u64>,
    batch_seconds: f64,
    verifying_key_file: Option<String>,
    verifying_key_bytes: Option<u64>,
}

struct RssMonitor {
    stop: mpsc::Sender<()>,
    sampled_peak_kib: Arc<AtomicU64>,
    handle: JoinHandle<()>,
}

impl RssMonitor {
    fn start() -> Self {
        let (stop, receiver) = mpsc::channel();
        let sampled_peak_kib = Arc::new(AtomicU64::new(0));
        let thread_peak = sampled_peak_kib.clone();
        let handle = thread::spawn(move || loop {
            if let Some(rss_kib) = resident_set_size_kib() {
                thread_peak.fetch_max(rss_kib, Ordering::Relaxed);
            }
            if receiver.recv_timeout(Duration::from_millis(250)).is_ok() {
                break;
            }
        });
        Self { stop, sampled_peak_kib, handle }
    }

    fn finish(self) -> Option<u64> {
        let _ = self.stop.send(());
        self.handle.join().expect("RSS monitor thread must not panic");
        match self.sampled_peak_kib.load(Ordering::Relaxed) {
            0 => None,
            value => Some(value),
        }
    }
}

fn resident_set_size_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
        return line.split_whitespace().nth(1)?.parse().ok();
    }

    #[cfg(not(target_os = "linux"))]
    {
        let pid = std::process::id().to_string();
        let output = Command::new("ps").args(["-o", "rss=", "-p", &pid]).output().ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout).ok()?.trim().parse().ok()
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
    let mut shape = Shape::full();
    let mut layer = DEFAULT_LAYER;
    let mut head = DEFAULT_HEAD;
    let mut head_was_set = false;
    let mut all_heads = false;
    let mut data_dir = std::env::var_os("SP1_ZKGPT_LIKE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    let mut output_dir = std::env::var_os("SP1_ZKGPT_LEAF_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-leaf-output"));
    let mut previous_block_dir =
        std::env::var_os("SP1_ZKGPT_PREVIOUS_BLOCK_DIR").map(PathBuf::from);
    let mut previous_block_join_dir =
        std::env::var_os("SP1_ZKGPT_PREVIOUS_BLOCK_JOIN_DIR").map(PathBuf::from);
    let mut expected_input_digest = None;
    let mut synthetic = false;
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
            Some("--head") => {
                head = parse_usize(arguments.next(), "--head");
                head_was_set = true;
            }
            Some("--all-heads") => all_heads = true,
            Some("--data-dir") => {
                data_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--data-dir requires a value")),
                );
                synthetic = false;
            }
            Some("--output-dir") => {
                output_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--output-dir requires a value")),
                );
            }
            Some("--previous-block-dir") => {
                previous_block_dir = Some(PathBuf::from(
                    arguments
                        .next()
                        .unwrap_or_else(|| panic!("--previous-block-dir requires a value")),
                ));
            }
            Some("--previous-block-join-dir") => {
                previous_block_join_dir = Some(PathBuf::from(
                    arguments
                        .next()
                        .unwrap_or_else(|| panic!("--previous-block-join-dir requires a value")),
                ));
            }
            Some("--expected-input-digest") => {
                let value = arguments
                    .next()
                    .unwrap_or_else(|| panic!("--expected-input-digest requires a value"));
                expected_input_digest = Some(parse_digest(
                    value.to_str().expect("--expected-input-digest must be valid UTF-8"),
                ));
            }
            Some(value) => panic!("unknown option: {value}"),
            None => panic!("command-line options must be valid UTF-8"),
        }
    }

    shape.validate();
    assert!(!(all_heads && head_was_set), "--all-heads cannot be combined with --head");
    assert!(head < shape.num_heads, "head index {head} is out of range");
    if !synthetic {
        assert_eq!(
            (shape.sequence_length, shape.hidden_size, shape.num_heads),
            (DEFAULT_SEQUENCE_LENGTH, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_HEADS),
            "real leaf data currently uses the fixed 30 x 768 x 12 fixture"
        );
    }
    assert!(
        layer > 0 || (previous_block_dir.is_none() && previous_block_join_dir.is_none()),
        "previous-block inputs are only valid for layer 1 or later"
    );
    if previous_block_join_dir.is_none() {
        previous_block_join_dir = previous_block_dir.clone();
    }
    if layer > 0 && !synthetic && matches!(mode, Mode::Execute | Mode::Prove) {
        assert!(previous_block_dir.is_some(), "layer 1 or later requires --previous-block-dir");
        assert!(
            previous_block_join_dir.is_some(),
            "layer 1 or later requires --previous-block-join-dir"
        );
    }

    Arguments {
        mode,
        shape,
        layer,
        head,
        all_heads,
        data_dir,
        output_dir,
        previous_block_dir,
        previous_block_join_dir,
        expected_input_digest,
        synthetic,
    }
}

fn selected_heads(arguments: &Arguments) -> Vec<usize> {
    if arguments.all_heads {
        (0..arguments.shape.num_heads).collect()
    } else {
        vec![arguments.head]
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

fn verify_previous_block_join(arguments: &Arguments) -> UpstreamCommitments {
    type A = RecursionAir<SP1Field, 3, 2>;

    assert!(arguments.chained(), "a genesis leaf has no previous block");
    let previous_layer = arguments.layer - 1;
    let join_dir = arguments
        .previous_block_join_dir
        .as_ref()
        .expect("layer 1 or later requires --previous-block-join-dir");
    let stem = format!("zkgpt_mlp_projection_join_l{previous_layer:02}");
    let manifest_path = join_dir.join(format!("{stem}.manifest.json"));
    assert_eq!(
        json_string_field(&manifest_path, "stage"),
        "mlp_projection_block_join",
        "previous-block manifest has the wrong stage"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "layer"),
        previous_layer,
        "previous-block manifest has the wrong layer"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "sequence_length"),
        arguments.shape.sequence_length,
        "previous-block sequence length differs from the attention leaf"
    );
    assert_eq!(
        json_usize_field(&manifest_path, "hidden_size"),
        arguments.shape.hidden_size,
        "previous-block hidden size differs from the attention leaf"
    );
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
        PREVIOUS_BLOCK_JOIN_MAX_LOG_ROWS as u32,
        PREVIOUS_BLOCK_JOIN_MAX_LOG_ROWS,
        A::verillm_machine(),
    );
    let prover = simple_prover(verifier);
    let vk_path = join_dir.join(format!("{stem}.vk.bin"));
    let proof_path = join_dir.join(format!("{stem}.proof.bin"));
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
    assert_eq!(proof.shard_proofs.len(), 1, "previous-block join proof must contain one shard");
    let public_values: &RecursionPublicValues<SP1Field> =
        proof.shard_proofs[0].public_values.as_slice().borrow();
    assert_eq!(
        public_values.digest, upstream.transcript,
        "previous-block join proof public digest differs from its manifest"
    );
    prover.verify(&vk, &proof).expect("previous-block MLP projection join proof must verify");
    println!(
        "previous-block join proof verified: layer={previous_layer} transcript={}",
        digest_hex(&upstream.transcript)
    );
    upstream
}

fn load_leaf_input(arguments: &Arguments) -> LeafInput {
    let expected_values = arguments.shape.sequence_length * arguments.shape.hidden_size;
    if arguments.synthetic {
        let hidden_states = vec![0; expected_values];
        let upstream = arguments.chained().then(|| UpstreamCommitments {
            output: host_commit_u16(DOMAIN_MLP_PROJECTION_GROUP_OUTPUT, &hidden_states),
            transcript: host_commit_fields(
                DOMAIN_SYNTHETIC_UPSTREAM_TRANSCRIPT,
                &[SP1Field::from_canonical_usize(arguments.layer - 1)],
            ),
        });
        return LeafInput { hidden_states, upstream };
    }

    if !arguments.chained() {
        let hidden_states = read_bf16_binary(&arguments.data_dir.join("hidden_state.bf16.bin"));
        assert_eq!(hidden_states.len(), expected_values);
        return LeafInput { hidden_states, upstream: None };
    }

    let upstream = verify_previous_block_join(arguments);
    let previous_layer = arguments.layer - 1;
    let private_dir = arguments
        .previous_block_dir
        .as_ref()
        .expect("layer 1 or later requires --previous-block-dir");
    let input_path = private_dir
        .join(format!("zkgpt_mlp_projection_l{previous_layer:02}.output.private.bf16.bin"));
    let hidden_states = read_bf16_binary(&input_path);
    assert_eq!(hidden_states.len(), expected_values);
    let input_digest = host_commit_u16(DOMAIN_MLP_PROJECTION_GROUP_OUTPUT, &hidden_states);
    assert_eq!(
        input_digest, upstream.output,
        "private previous-block output does not match the verified join commitment"
    );
    println!(
        "previous-block private output bound: layer={previous_layer} values={} commitment={}",
        hidden_states.len(),
        digest_hex(&input_digest)
    );
    LeafInput { hidden_states, upstream: Some(upstream) }
}

fn extract_qkv_head_weight(
    full_weight: &[u16],
    hidden_size: usize,
    num_heads: usize,
    head: usize,
) -> Vec<u16> {
    let head_dimension = hidden_size / num_heads;
    let full_output_width = 3 * hidden_size;
    assert_eq!(full_weight.len(), hidden_size * full_output_width);
    assert!(head < num_heads);

    let mut tile = Vec::with_capacity(hidden_size * 3 * head_dimension);
    for row in full_weight.chunks_exact(full_output_width) {
        for section in 0..3 {
            let start = section * hidden_size + head * head_dimension;
            tile.extend_from_slice(&row[start..start + head_dimension]);
        }
    }
    tile
}

fn load_leaf_data(arguments: &Arguments, input: &LeafInput, head: usize) -> LeafData {
    let shape = arguments.shape;
    assert!(head < shape.num_heads, "head index {head} is out of range");
    if arguments.synthetic {
        return LeafData {
            hidden_states: input.hidden_states.clone(),
            layer_norm_weight: vec![0x3f80; shape.hidden_size],
            layer_norm_bias: vec![0; shape.hidden_size],
            qkv_head_weight: vec![0x3f80; shape.hidden_size * shape.qkv_head_width()],
            attention_max_hints: vec![0; shape.sequence_length],
            upstream: input.upstream,
        };
    }

    let hidden_states = input.hidden_states.clone();
    let layer_dir = arguments.data_dir.join(format!("layer-{:02}", arguments.layer));
    let layer_norm_weight = read_bf16_binary(&layer_dir.join("ln_1_weight.bf16.bin"));
    let layer_norm_bias = read_bf16_binary(&layer_dir.join("ln_1_bias.bf16.bin"));
    let full_qkv_weight = read_bf16_binary(&layer_dir.join("attention_qkv_weight.bf16.bin"));
    let all_max_hints = read_bf16_binary(&layer_dir.join("attention_max_hints.bf16.bin"));

    assert_eq!(hidden_states.len(), shape.sequence_length * shape.hidden_size);
    assert_eq!(layer_norm_weight.len(), shape.hidden_size);
    assert_eq!(layer_norm_bias.len(), shape.hidden_size);
    assert_eq!(all_max_hints.len(), shape.sequence_length * shape.num_heads);

    let qkv_head_weight =
        extract_qkv_head_weight(&full_qkv_weight, shape.hidden_size, shape.num_heads, head);
    let attention_max_hints = (0..shape.sequence_length)
        .map(|token| all_max_hints[token * shape.num_heads + head])
        .collect();

    LeafData {
        hidden_states,
        layer_norm_weight,
        layer_norm_bias,
        qkv_head_weight,
        attention_max_hints,
        upstream: input.upstream,
    }
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
                sum = reference_add(sum, product);
            }
            output.push(sum);
        }
    }
    output
}

fn reference_dot(lhs: &[u16], rhs: &[u16]) -> u16 {
    let mut sum = Bf16MulWitness::new(lhs[0], rhs[0]).output.raw;
    for (&lhs, &rhs) in lhs[1..].iter().zip(&rhs[1..]) {
        let product = Bf16MulWitness::new(lhs, rhs).output.raw;
        sum = reference_add(sum, product);
    }
    sum
}

fn reference_attention_head(
    query: &[u16],
    keys: &[u16],
    values: &[u16],
    scale: u16,
    max_hint: u16,
) -> Vec<u16> {
    let scores = keys
        .chunks_exact(query.len())
        .map(|key| {
            let dot = reference_dot(query, key);
            Bf16MulWitness::new(dot, scale).output.raw
        })
        .collect::<Vec<_>>();
    let exponential_values = scores
        .iter()
        .map(|&score| {
            let shifted = reference_sub(score, max_hint);
            Bf16UnaryWitness::new(Bf16UnaryOpcode::Exponential, shifted).output
        })
        .collect::<Vec<_>>();
    let sum = exponential_values[1..]
        .iter()
        .fold(exponential_values[0], |sum, &value| reference_add(sum, value));
    let probabilities = exponential_values
        .into_iter()
        .map(|value| Bf16DivWitness::new(value, sum).output.raw)
        .collect::<Vec<_>>();

    (0..query.len())
        .map(|feature| {
            let value_column =
                values.chunks_exact(query.len()).map(|row| row[feature]).collect::<Vec<_>>();
            reference_dot(&probabilities, &value_column)
        })
        .collect()
}

fn reference_leaf(shape: Shape, data: &LeafData) -> Vec<u16> {
    let mut normalized = Vec::with_capacity(data.hidden_states.len());
    for row in data.hidden_states.chunks_exact(shape.hidden_size) {
        normalized.extend(reference_layer_norm_row(
            row,
            &data.layer_norm_weight,
            &data.layer_norm_bias,
            GPT2_LAYER_NORM_EPSILON,
        ));
    }
    let qkv = reference_linear_rows_no_bias(&normalized, shape.hidden_size, &data.qkv_head_weight);

    let head_dimension = shape.head_dimension();
    let qkv_width = shape.qkv_head_width();
    let mut output = Vec::with_capacity(shape.sequence_length * head_dimension);
    for query_token in 0..shape.sequence_length {
        let query_row = &qkv[query_token * qkv_width..(query_token + 1) * qkv_width];
        let query = &query_row[..head_dimension];
        let mut causal_keys = Vec::with_capacity((query_token + 1) * head_dimension);
        let mut causal_values = Vec::with_capacity((query_token + 1) * head_dimension);
        for key_token in 0..=query_token {
            let row = &qkv[key_token * qkv_width..(key_token + 1) * qkv_width];
            causal_keys.extend_from_slice(&row[head_dimension..2 * head_dimension]);
            causal_values.extend_from_slice(&row[2 * head_dimension..3 * head_dimension]);
        }
        output.extend(reference_attention_head(
            query,
            &causal_keys,
            &causal_values,
            GPT2_ATTENTION_SCALE,
            data.attention_max_hints[query_token],
        ));
    }
    output
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
    builder: &mut LeafBuilder,
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

fn transcript_fields_host(
    shape: Shape,
    layer: usize,
    head: usize,
    commitments: LeafTranscriptInputs,
) -> Vec<SP1Field> {
    let mut fields = [
        LEAF_VERSION,
        if commitments.upstream.is_some() {
            LEAF_STAGE_CHAINED_QKV_ATTENTION
        } else {
            LEAF_STAGE_QKV_ATTENTION
        },
        layer as u32,
        head as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.head_dimension() as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
        GPT2_ATTENTION_SCALE as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    if let Some(upstream) = commitments.upstream {
        fields.extend(upstream);
    }
    fields.extend(commitments.input);
    fields.extend(commitments.parameters);
    fields.extend(commitments.hints);
    fields.extend(commitments.output);
    fields
}

fn compute_host_commitments(
    shape: Shape,
    layer: usize,
    head: usize,
    data: &LeafData,
    output: &[u16],
) -> LeafCommitments {
    let input_domain =
        if data.upstream.is_some() { DOMAIN_MLP_PROJECTION_GROUP_OUTPUT } else { DOMAIN_INPUT };
    let input = host_commit_u16(input_domain, &data.hidden_states);
    if let Some(upstream) = data.upstream {
        assert_eq!(input, upstream.output, "leaf input differs from previous-block output");
    }
    let mut parameter_values = Vec::with_capacity(
        data.layer_norm_weight.len() + data.layer_norm_bias.len() + data.qkv_head_weight.len(),
    );
    parameter_values.extend_from_slice(&data.layer_norm_weight);
    parameter_values.extend_from_slice(&data.layer_norm_bias);
    parameter_values.extend_from_slice(&data.qkv_head_weight);
    let parameters = host_commit_u16(DOMAIN_PARAMETERS, &parameter_values);
    let hints = host_commit_u16(DOMAIN_HINTS, &data.attention_max_hints);
    let output = host_commit_u16(DOMAIN_OUTPUT, output);
    let transcript = host_commit_fields(
        DOMAIN_TRANSCRIPT,
        &transcript_fields_host(
            shape,
            layer,
            head,
            LeafTranscriptInputs {
                upstream: data.upstream.map(|upstream| upstream.transcript),
                input,
                parameters,
                hints,
                output,
            },
        ),
    );
    LeafCommitments {
        upstream: data.upstream.map(|upstream| upstream.transcript),
        input,
        parameters,
        hints,
        output,
        transcript,
    }
}

fn build_leaf(builder: &mut LeafBuilder, shape: Shape, chained: bool) {
    let metadata = builder.hint_felts_v2(2);
    let (upstream, expected_input) = if chained {
        let upstream: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let expected_input: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        (Some(upstream), Some(expected_input))
    } else {
        (None, None)
    };
    let hidden_states = builder.hint_felts_v2(shape.sequence_length * shape.hidden_size);
    let layer_norm_weight = builder.hint_felts_v2(shape.hidden_size);
    let layer_norm_bias = builder.hint_felts_v2(shape.hidden_size);
    let qkv_head_weight = builder.hint_felts_v2(shape.hidden_size * shape.qkv_head_width());
    let attention_max_hints = builder.hint_felts_v2(shape.sequence_length);

    let input_domain = if chained { DOMAIN_MLP_PROJECTION_GROUP_OUTPUT } else { DOMAIN_INPUT };
    let input_digest = circuit_commit_fields(builder, input_domain, &hidden_states);
    if let Some(expected_input) = expected_input {
        for (&computed, &expected) in input_digest.iter().zip(&expected_input) {
            builder.assert_felt_eq(computed, expected);
        }
    }
    let mut parameter_values =
        Vec::with_capacity(layer_norm_weight.len() + layer_norm_bias.len() + qkv_head_weight.len());
    parameter_values.extend_from_slice(&layer_norm_weight);
    parameter_values.extend_from_slice(&layer_norm_bias);
    parameter_values.extend_from_slice(&qkv_head_weight);
    let parameter_digest = circuit_commit_fields(builder, DOMAIN_PARAMETERS, &parameter_values);
    let hints_digest = circuit_commit_fields(builder, DOMAIN_HINTS, &attention_max_hints);

    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));
    let normalized = builder.bf16_layer_norm_rows(
        &hidden_states,
        shape.hidden_size,
        &layer_norm_weight,
        &layer_norm_bias,
        epsilon,
    );
    let qkv = builder.bf16_linear_rows_no_bias(&normalized, shape.hidden_size, &qkv_head_weight);

    let head_dimension = shape.head_dimension();
    let qkv_width = shape.qkv_head_width();
    let attention_rows: Vec<Vec<_>> =
        (0..shape.sequence_length).ir_par_map_collect(builder, |builder, query_token| {
            let query_row = &qkv[query_token * qkv_width..(query_token + 1) * qkv_width];
            let query = &query_row[..head_dimension];
            let mut causal_keys = Vec::with_capacity((query_token + 1) * head_dimension);
            let mut causal_values = Vec::with_capacity((query_token + 1) * head_dimension);
            for key_token in 0..=query_token {
                let row = &qkv[key_token * qkv_width..(key_token + 1) * qkv_width];
                causal_keys.extend_from_slice(&row[head_dimension..2 * head_dimension]);
                causal_values.extend_from_slice(&row[2 * head_dimension..3 * head_dimension]);
            }
            builder.bf16_attention_head(
                query,
                &causal_keys,
                &causal_values,
                scale,
                attention_max_hints[query_token],
            )
        });
    let output = attention_rows.into_iter().flatten().collect::<Vec<_>>();
    let output_digest = circuit_commit_fields(builder, DOMAIN_OUTPUT, &output);

    let constants = [
        LEAF_VERSION,
        if chained { LEAF_STAGE_CHAINED_QKV_ATTENTION } else { LEAF_STAGE_QKV_ATTENTION },
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.head_dimension() as u32,
        GPT2_LAYER_NORM_EPSILON as u32,
        GPT2_ATTENTION_SCALE as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut transcript_fields = vec![constants[0], constants[1], metadata[0], metadata[1]];
    transcript_fields.extend_from_slice(&constants[2..]);
    if let Some(upstream) = upstream {
        transcript_fields.extend(upstream);
    }
    transcript_fields.extend(input_digest);
    transcript_fields.extend(parameter_digest);
    transcript_fields.extend(hints_digest);
    transcript_fields.extend(output_digest);
    let transcript_digest = circuit_commit_fields(builder, DOMAIN_TRANSCRIPT, &transcript_fields);

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = transcript_digest;
    builder.commit_public_values_v2(*public_values);
}

fn witness_stream(
    arguments: &Arguments,
    head: usize,
    data: &LeafData,
) -> Vec<sp1_recursion_executor::Block<SP1Field>> {
    let upstream_values = usize::from(data.upstream.is_some()) * 2 * DIGEST_SIZE;
    let mut values = Vec::with_capacity(
        2 + upstream_values
            + data.hidden_states.len()
            + data.layer_norm_weight.len()
            + data.layer_norm_bias.len()
            + data.qkv_head_weight.len()
            + data.attention_max_hints.len(),
    );
    values.push(SP1Field::from_canonical_usize(arguments.layer));
    values.push(SP1Field::from_canonical_usize(head));
    if let Some(upstream) = data.upstream {
        values.extend(upstream.transcript);
        values.extend(upstream.output);
    }
    values.extend(data.hidden_states.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.layer_norm_weight.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.layer_norm_bias.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.extend(data.qkv_head_weight.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values
        .extend(data.attention_max_hints.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    values.into_iter().map(Into::into).collect()
}

fn padded_rows(events: usize, lanes: usize) -> usize {
    events.div_ceil(lanes).next_multiple_of(32)
}

fn print_estimate(shape: Shape, events: EventCounts) {
    let max_rows = 1usize << shape.max_log_rows();
    let plan = plan_uniform_natural_shards(
        shape.num_heads,
        Bf16EventsPerUnit {
            mul: events.mul,
            add_sub: events.add_sub,
            unary: events.unary,
            div: events.div,
        },
        ShardLimits { max_log_rows: shape.max_log_rows(), ..ShardLimits::full() },
    );
    let event_record_bytes = events.mul * std::mem::size_of::<Bf16MulEvent<SP1Field>>()
        + events.add_sub * std::mem::size_of::<Bf16AddSubEvent<SP1Field>>()
        + events.unary * std::mem::size_of::<Bf16UnaryEvent<SP1Field>>()
        + events.div * std::mem::size_of::<Bf16DivEvent<SP1Field>>();
    println!(
        "leaf shape: seq_len={} hidden={} heads={} head_dim={} qkv_head_width={}",
        shape.sequence_length,
        shape.hidden_size,
        shape.num_heads,
        shape.head_dimension(),
        shape.qkv_head_width(),
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
        "12-lane rows: mul={} add_sub={} max_2^{}={max_rows}",
        padded_rows(events.mul, NUM_BF16_MUL_EVENTS_PER_ROW),
        padded_rows(events.add_sub, NUM_BF16_ADD_SUB_EVENTS_PER_ROW),
        shape.max_log_rows(),
    );
    println!(
        "executor-record lower bound: bytes={event_record_bytes} gib={:.2}",
        event_record_bytes as f64 / 1024_f64.powi(3)
    );
    println!(
        "shape-aware capacity: natural_unit=attention_head heads_per_leaf={} leaves={} estimated_rows={} estimated_trace_area={} limit={}",
        plan.units_per_shard,
        plan.shard_count,
        plan.estimate.max_rows(),
        plan.estimate.estimated_trace_area,
        plan.limits.max_trace_area,
    );
    println!(
        "selected attention plan: heads_per_leaf=1 leaves={} (one complete head is the leaf protocol unit)",
        shape.num_heads,
    );
    if shape.sequence_length == DEFAULT_SEQUENCE_LENGTH
        && shape.hidden_size == DEFAULT_HIDDEN_SIZE
        && shape.num_heads == DEFAULT_NUM_HEADS
    {
        assert_eq!(
            plan.units_per_shard, 1,
            "the production attention shape unexpectedly fits multiple complete heads per leaf"
        );
    }
}

fn digest_hex(digest: &Digest) -> String {
    digest
        .iter()
        .map(|value| format!("{:08X}", value.as_canonical_u32()))
        .collect::<Vec<_>>()
        .join(":")
}

fn write_commitment_manifest(
    output_dir: &Path,
    arguments: &Arguments,
    head: usize,
    commitments: LeafCommitments,
) {
    fs::create_dir_all(output_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", output_dir.display()));
    let stage =
        if commitments.upstream.is_some() { "qkv_attention_chained" } else { "qkv_attention" };
    let upstream = commitments
        .upstream
        .map(|digest| format!("upstream={}\n", digest_hex(&digest)))
        .unwrap_or_default();
    let manifest = format!(
        "version={LEAF_VERSION}\nstage={stage}\nlayer={}\nhead={}\n{upstream}input={}\nparameters={}\nhints={}\noutput={}\ntranscript={}\n",
        arguments.layer,
        head,
        digest_hex(&commitments.input),
        digest_hex(&commitments.parameters),
        digest_hex(&commitments.hints),
        digest_hex(&commitments.output),
        digest_hex(&commitments.transcript),
    );
    let path =
        output_dir.join(format!("zkgpt_leaf_l{:02}_h{:02}.commitments.txt", arguments.layer, head));
    fs::write(&path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    println!("commitment manifest: {}", path.display());
}

fn execute_head(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    input: &LeafInput,
    events: EventCounts,
    head: usize,
) -> HeadExecution {
    let load_started = Instant::now();
    let data = load_leaf_data(arguments, input, head);
    let load_seconds = load_started.elapsed().as_secs_f64();
    println!(
        "loaded leaf data: synthetic={} layer={} head={} hidden={} qkv_weight={} elapsed={load_seconds:.3}s",
        arguments.synthetic,
        arguments.layer,
        head,
        data.hidden_states.len(),
        data.qkv_head_weight.len(),
    );

    let reference_started = Instant::now();
    let output = reference_leaf(arguments.shape, &data);
    let commitments =
        compute_host_commitments(arguments.shape, arguments.layer, head, &data, &output);
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    if let Some(expected) = arguments.expected_input_digest {
        assert_eq!(
            commitments.input, expected,
            "leaf input commitment does not match the previous host-chain output"
        );
        println!("host chain input: matched {}", digest_hex(&expected));
    }
    println!("host chain output: {}", digest_hex(&commitments.output));
    println!(
        "host reference and commitments: output_values={} elapsed={reference_seconds:.3}s",
        output.len(),
    );
    write_commitment_manifest(&arguments.output_dir, arguments, head, commitments);

    let mut executor =
        Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
    executor.witness_stream = witness_stream(arguments, head, &data).into();
    let execute_started = Instant::now();
    executor.run().unwrap();
    let execute_seconds = execute_started.elapsed().as_secs_f64();
    println!("executed leaf: head={head} elapsed={execute_seconds:.3}s");

    assert_eq!(executor.record.bf16_mul_events.len(), events.mul);
    assert_eq!(executor.record.bf16_add_sub_events.len(), events.add_sub);
    assert_eq!(executor.record.bf16_unary_events.len(), events.unary);
    assert_eq!(executor.record.bf16_div_events.len(), events.div);
    assert_eq!(
        executor.record.public_values.digest, commitments.transcript,
        "circuit transcript commitment differs from the independent host commitment chain"
    );
    println!("host commitment chain verified: head={head}");
    println!("public transcript: {}", digest_hex(&commitments.transcript));

    let record = std::mem::take(&mut executor.record);
    drop(executor);
    HeadExecution {
        head,
        output,
        commitments,
        record,
        load_seconds,
        reference_seconds,
        execute_seconds,
    }
}

fn check_common_input(common_input: &mut Option<Digest>, execution: &HeadExecution) {
    if let Some(expected) = common_input {
        assert_eq!(
            execution.commitments.input, *expected,
            "head {} does not share the attention group's input commitment",
            execution.head
        );
    } else {
        *common_input = Some(execution.commitments.input);
        let source =
            if execution.commitments.upstream.is_some() { "previous block" } else { "genesis" };
        println!("host chain input: {source} {}", digest_hex(&execution.commitments.input));
    }
}

fn execute_heads(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    input: &LeafInput,
    events: EventCounts,
    heads: &[usize],
) -> (Vec<CompletedHead>, f64) {
    let batch_started = Instant::now();
    let mut common_input = None;
    let mut completed = Vec::with_capacity(heads.len());
    for &head in heads {
        let monitor = RssMonitor::start();
        let execution = execute_head(program.clone(), arguments, input, events, head);
        check_common_input(&mut common_input, &execution);
        let sampled_peak_rss_kib = monitor.finish();
        let HeadExecution {
            head,
            output,
            commitments,
            record,
            load_seconds,
            reference_seconds,
            execute_seconds,
        } = execution;
        drop(record);
        completed.push(CompletedHead {
            output,
            summary: HeadSummary {
                head,
                commitments,
                load_seconds,
                reference_seconds,
                execute_seconds,
                prove_seconds: None,
                verify_seconds: None,
                sampled_peak_rss_kib,
                proof_file: None,
                proof_bytes: None,
            },
        });
    }
    (completed, batch_started.elapsed().as_secs_f64())
}

async fn prove_heads(
    program: Arc<RecursionProgram<SP1Field>>,
    arguments: &Arguments,
    input: &LeafInput,
    events: EventCounts,
    heads: &[usize],
) -> (Vec<CompletedHead>, BatchMetrics) {
    type A = RecursionAir<SP1Field, 3, 2>;

    let max_log_rows = arguments.shape.max_log_rows();
    let max_rows = 1usize << max_log_rows;
    let machine = A::verillm_machine();
    println!(
        "proof config: log_blowup={PROOF_LOG_BLOWUP} log_stacking_height={max_log_rows} \
         max_log_row_count={max_log_rows}"
    );
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

    let setup_monitor = RssMonitor::start();
    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program.clone()).await;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let setup_sampled_peak_rss_kib = setup_monitor.finish();
    println!("proof setup: elapsed={setup_seconds:.3}s");
    let pk = unsafe { pk.into_inner() };

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let vk_stem = if arguments.all_heads {
        format!("zkgpt_leaf_l{:02}.shared", arguments.layer)
    } else {
        format!("zkgpt_leaf_l{:02}_h{:02}", arguments.layer, heads[0])
    };
    let vk_file = format!("{vk_stem}.vk.bin");
    let vk_path = arguments.output_dir.join(&vk_file);
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    let vk_bytes = fs::metadata(&vk_path).unwrap().len();
    println!("shared verifying key: {} ({vk_bytes} bytes)", vk_path.display());

    let batch_started = Instant::now();
    let mut common_input = None;
    let mut completed = Vec::with_capacity(heads.len());
    for (head_position, &head) in heads.iter().enumerate() {
        println!("attention head {}/{}: head={head}", head_position + 1, heads.len());
        let monitor = RssMonitor::start();
        let execution = execute_head(program.clone(), arguments, input, events, head);
        check_common_input(&mut common_input, &execution);

        for chip in prover.machine().chips() {
            if chip.included(&execution.record) {
                let rows = chip.num_rows(&execution.record).unwrap_or_default();
                if head_position == 0 {
                    println!("trace: {:<22} rows={rows}", chip.name());
                }
                assert!(
                    rows <= max_rows,
                    "{} needs {rows} rows, exceeding the configured maximum {max_rows}",
                    chip.name()
                );
            }
        }
        let proof_shape =
            prover.shape_from_record(&execution.record).expect("leaf machine has no proof shape");
        let trace_area = proof_shape.preprocessed_area + proof_shape.main_area;
        assert!(
            trace_area <= DEFAULT_MAX_TRACE_AREA,
            "attention leaf trace area {trace_area} exceeds shape-aware limit {DEFAULT_MAX_TRACE_AREA}"
        );
        println!("proof shape: head={head} {proof_shape:?}");

        let HeadExecution {
            head,
            output,
            commitments,
            record,
            load_seconds,
            reference_seconds,
            execute_seconds,
        } = execution;
        let prove_started = Instant::now();
        let shard_proof = prover.prove_shard(pk.clone(), record).await;
        let proof = MachineProof::from(vec![shard_proof]);
        let prove_seconds = prove_started.elapsed().as_secs_f64();
        println!("proof generated: head={head} elapsed={prove_seconds:.3}s");

        let verify_started = Instant::now();
        prover.verify(&vk, &proof).expect("generated zkGPT leaf proof must verify");
        let verify_seconds = verify_started.elapsed().as_secs_f64();
        println!("proof verified: head={head} elapsed={verify_seconds:.3}s");

        let proof_file = format!("zkgpt_leaf_l{:02}_h{head:02}.proof.bin", arguments.layer);
        let proof_path = arguments.output_dir.join(&proof_file);
        bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
        let proof_bytes = fs::metadata(&proof_path).unwrap().len();
        println!("proof artifact: {} ({proof_bytes} bytes)", proof_path.display());
        drop(proof);
        let sampled_peak_rss_kib = monitor.finish();
        if let Some(rss_kib) = sampled_peak_rss_kib {
            println!(
                "sampled peak RSS: head={head} kib={rss_kib} gib={:.2}",
                rss_kib as f64 / 1024_f64.powi(2)
            );
        }

        completed.push(CompletedHead {
            output,
            summary: HeadSummary {
                head,
                commitments,
                load_seconds,
                reference_seconds,
                execute_seconds,
                prove_seconds: Some(prove_seconds),
                verify_seconds: Some(verify_seconds),
                sampled_peak_rss_kib,
                proof_file: Some(proof_file),
                proof_bytes: Some(proof_bytes),
            },
        });
    }

    (
        completed,
        BatchMetrics {
            setup_seconds: Some(setup_seconds),
            setup_sampled_peak_rss_kib,
            batch_seconds: batch_started.elapsed().as_secs_f64(),
            verifying_key_file: Some(vk_file),
            verifying_key_bytes: Some(vk_bytes),
            ..BatchMetrics::default()
        },
    )
}

fn compute_attention_group(
    shape: Shape,
    layer: usize,
    completed: &[CompletedHead],
) -> (Vec<u16>, AttentionGroupCommitments) {
    assert_eq!(completed.len(), shape.num_heads, "attention group requires every head");
    let upstream = completed[0].summary.commitments.upstream;
    let input = completed[0].summary.commitments.input;
    let mut parameter_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    let mut hint_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    for (expected_head, completed_head) in completed.iter().enumerate() {
        assert_eq!(completed_head.summary.head, expected_head, "attention heads must be ordered");
        assert_eq!(
            completed_head.summary.commitments.upstream, upstream,
            "attention upstream transcripts differ"
        );
        assert_eq!(completed_head.summary.commitments.input, input, "attention inputs differ");
        let head_field = SP1Field::from_canonical_usize(expected_head);
        parameter_fields.push(head_field);
        parameter_fields.extend(completed_head.summary.commitments.parameters);
        hint_fields.push(head_field);
        hint_fields.extend(completed_head.summary.commitments.hints);
    }
    let parameters = host_commit_fields(DOMAIN_ATTENTION_GROUP_PARAMETERS, &parameter_fields);
    let hints = host_commit_fields(DOMAIN_ATTENTION_GROUP_HINTS, &hint_fields);

    let head_dimension = shape.head_dimension();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for completed_head in completed {
            assert_eq!(
                completed_head.output.len(),
                shape.sequence_length * head_dimension,
                "attention head output has the wrong shape"
            );
            let start = token * head_dimension;
            output.extend_from_slice(&completed_head.output[start..start + head_dimension]);
        }
    }
    let output_digest = host_commit_u16(DOMAIN_ATTENTION_GROUP_OUTPUT, &output);

    let mut transcript_fields = [
        LEAF_VERSION,
        if upstream.is_some() { ATTENTION_CHAINED_GROUP_STAGE } else { ATTENTION_GROUP_STAGE },
        layer as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.head_dimension() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    if let Some(upstream) = upstream {
        transcript_fields.extend(upstream);
    }
    transcript_fields.extend(input);
    transcript_fields.extend(parameters);
    transcript_fields.extend(hints);
    transcript_fields.extend(output_digest);
    for completed_head in completed {
        transcript_fields.push(SP1Field::from_canonical_usize(completed_head.summary.head));
        transcript_fields.extend(completed_head.summary.commitments.transcript);
    }
    let transcript = host_commit_fields(DOMAIN_ATTENTION_GROUP_TRANSCRIPT, &transcript_fields);
    (
        output,
        AttentionGroupCommitments {
            upstream,
            input,
            parameters,
            hints,
            output: output_digest,
            transcript,
        },
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

fn write_attention_group_artifacts(
    arguments: &Arguments,
    completed: &[CompletedHead],
    commitments: AttentionGroupCommitments,
    output: &[u16],
    metrics: &BatchMetrics,
) {
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_attention_l{:02}", arguments.layer);
    let output_file = format!("{stem}.output.private.bf16.bin");
    let output_path = arguments.output_dir.join(&output_file);
    let output_bytes = output.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    fs::write(&output_path, output_bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));

    let mut manifest = String::new();
    writeln!(manifest, "{{").unwrap();
    writeln!(manifest, "  \"version\": {LEAF_VERSION},").unwrap();
    writeln!(manifest, "  \"stage\": \"qkv_attention_group\",").unwrap();
    writeln!(manifest, "  \"layer\": {},", arguments.layer).unwrap();
    writeln!(manifest, "  \"sequence_length\": {},", arguments.shape.sequence_length).unwrap();
    writeln!(manifest, "  \"hidden_size\": {},", arguments.shape.hidden_size).unwrap();
    writeln!(manifest, "  \"num_heads\": {},", arguments.shape.num_heads).unwrap();
    writeln!(manifest, "  \"sharding_strategy\": \"shape_aware_attention_heads\",").unwrap();
    writeln!(manifest, "  \"natural_unit\": \"attention_head\",").unwrap();
    writeln!(manifest, "  \"units_per_leaf\": 1,").unwrap();
    writeln!(manifest, "  \"max_log_rows\": {},", arguments.shape.max_log_rows()).unwrap();
    writeln!(manifest, "  \"max_trace_area\": {DEFAULT_MAX_TRACE_AREA},").unwrap();
    writeln!(
        manifest,
        "  \"upstream_transcript\": {},",
        commitments
            .upstream
            .map_or_else(|| "null".to_owned(), |digest| format!("\"{}\"", digest_hex(&digest)))
    )
    .unwrap();
    writeln!(manifest, "  \"input_commitment\": \"{}\",", digest_hex(&commitments.input)).unwrap();
    writeln!(manifest, "  \"parameters_commitment\": \"{}\",", digest_hex(&commitments.parameters))
        .unwrap();
    writeln!(manifest, "  \"hints_commitment\": \"{}\",", digest_hex(&commitments.hints)).unwrap();
    writeln!(manifest, "  \"output_commitment\": \"{}\",", digest_hex(&commitments.output))
        .unwrap();
    writeln!(manifest, "  \"transcript_commitment\": \"{}\",", digest_hex(&commitments.transcript))
        .unwrap();
    writeln!(manifest, "  \"private_output_file\": \"{output_file}\",").unwrap();
    writeln!(manifest, "  \"output_values\": {},", output.len()).unwrap();
    writeln!(manifest, "  \"build_seconds\": {:.6},", metrics.build_seconds).unwrap();
    writeln!(manifest, "  \"compile_seconds\": {:.6},", metrics.compile_seconds).unwrap();
    writeln!(manifest, "  \"setup_seconds\": {},", optional_f64(metrics.setup_seconds)).unwrap();
    writeln!(
        manifest,
        "  \"setup_sampled_peak_rss_kib\": {},",
        optional_u64(metrics.setup_sampled_peak_rss_kib)
    )
    .unwrap();
    writeln!(manifest, "  \"batch_seconds\": {:.6},", metrics.batch_seconds).unwrap();
    writeln!(
        manifest,
        "  \"verifying_key_file\": {},",
        optional_json_string(metrics.verifying_key_file.as_deref())
    )
    .unwrap();
    writeln!(manifest, "  \"verifying_key_bytes\": {},", optional_u64(metrics.verifying_key_bytes))
        .unwrap();
    writeln!(manifest, "  \"heads\": [").unwrap();
    for (index, completed_head) in completed.iter().enumerate() {
        let summary = &completed_head.summary;
        writeln!(manifest, "    {{").unwrap();
        writeln!(manifest, "      \"head\": {},", summary.head).unwrap();
        writeln!(
            manifest,
            "      \"parameters_commitment\": \"{}\",",
            digest_hex(&summary.commitments.parameters)
        )
        .unwrap();
        writeln!(
            manifest,
            "      \"hints_commitment\": \"{}\",",
            digest_hex(&summary.commitments.hints)
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
        writeln!(manifest, "      \"load_seconds\": {:.6},", summary.load_seconds).unwrap();
        writeln!(manifest, "      \"reference_seconds\": {:.6},", summary.reference_seconds)
            .unwrap();
        writeln!(manifest, "      \"execute_seconds\": {:.6},", summary.execute_seconds).unwrap();
        writeln!(manifest, "      \"prove_seconds\": {},", optional_f64(summary.prove_seconds))
            .unwrap();
        writeln!(manifest, "      \"verify_seconds\": {},", optional_f64(summary.verify_seconds))
            .unwrap();
        writeln!(
            manifest,
            "      \"sampled_peak_rss_kib\": {},",
            optional_u64(summary.sampled_peak_rss_kib)
        )
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
    println!("attention group private output: {}", output_path.display());
    println!("attention group manifest: {}", manifest_path.display());
    println!("attention group output commitment: {}", digest_hex(&commitments.output));
    println!("attention group transcript: {}", digest_hex(&commitments.transcript));
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let heads = selected_heads(&arguments);
    let events = EventCounts::for_shape(arguments.shape);
    println!(
        "mode={:?} layer={} heads={heads:?} reuse_setup={}",
        arguments.mode, arguments.layer, arguments.all_heads
    );
    print_estimate(arguments.shape, events);
    if arguments.mode == Mode::Estimate {
        return;
    }

    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: LeafBuilder = AsmBuilder::default();
    build_leaf(&mut builder, arguments.shape, arguments.chained());
    let block = builder.into_root_block();
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("built leaf: ir_ops={} elapsed={build_seconds:.3}s", block.ops.len());
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    println!("compiled leaf: elapsed={compile_seconds:.3}s");
    let input_started = Instant::now();
    let input = load_leaf_input(&arguments);
    println!(
        "prepared leaf input: source={} values={} elapsed={:.3}s",
        if input.upstream.is_some() { "previous-block" } else { "genesis" },
        input.hidden_states.len(),
        input_started.elapsed().as_secs_f64()
    );

    let (completed, mut metrics) = match arguments.mode {
        Mode::Execute => {
            let (completed, batch_seconds) =
                execute_heads(program, &arguments, &input, events, &heads);
            (completed, BatchMetrics { batch_seconds, ..BatchMetrics::default() })
        }
        Mode::Prove => prove_heads(program, &arguments, &input, events, &heads).await,
        Mode::Estimate | Mode::Build => unreachable!(),
    };
    metrics.build_seconds = build_seconds;
    metrics.compile_seconds = compile_seconds;

    if arguments.all_heads {
        let (attention_output, commitments) =
            compute_attention_group(arguments.shape, arguments.layer, &completed);
        write_attention_group_artifacts(
            &arguments,
            &completed,
            commitments,
            &attention_output,
            &metrics,
        );
    }
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_arguments(shape: Shape, layer: usize) -> Arguments {
        Arguments {
            mode: Mode::Execute,
            shape,
            layer,
            head: 0,
            all_heads: false,
            data_dir: PathBuf::new(),
            output_dir: PathBuf::new(),
            previous_block_dir: None,
            previous_block_join_dir: None,
            expected_input_digest: None,
            synthetic: true,
        }
    }

    fn test_data(shape: Shape, upstream: Option<UpstreamCommitments>) -> LeafData {
        LeafData {
            hidden_states: vec![0; shape.sequence_length * shape.hidden_size],
            layer_norm_weight: vec![0x3f80; shape.hidden_size],
            layer_norm_bias: vec![0; shape.hidden_size],
            qkv_head_weight: vec![0x3f80; shape.hidden_size * shape.qkv_head_width()],
            attention_max_hints: vec![0; shape.sequence_length],
            upstream,
        }
    }

    fn compile_test_leaf(shape: Shape, chained: bool) -> Arc<RecursionProgram<SP1Field>> {
        let mut builder: LeafBuilder = AsmBuilder::default();
        build_leaf(&mut builder, shape, chained);
        let mut compiler = AsmCompiler::default();
        Arc::new(
            compiler.compile_inner(builder.into_root_block()).validate().expect("valid leaf IR"),
        )
    }

    fn execute_test_leaf(
        program: Arc<RecursionProgram<SP1Field>>,
        arguments: &Arguments,
        data: &LeafData,
    ) -> (bool, Digest) {
        let mut executor =
            Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
        executor.witness_stream = witness_stream(arguments, arguments.head, data).into();
        let result = executor.run();
        (result.is_ok(), executor.record.public_values.digest)
    }

    #[test]
    fn chained_leaf_binds_previous_block_output_and_transcript() {
        let shape = Shape::small();
        let arguments = test_arguments(shape, 1);
        let hidden_states = vec![0; shape.sequence_length * shape.hidden_size];
        let upstream = UpstreamCommitments {
            output: host_commit_u16(DOMAIN_MLP_PROJECTION_GROUP_OUTPUT, &hidden_states),
            transcript: host_commit_fields(
                DOMAIN_SYNTHETIC_UPSTREAM_TRANSCRIPT,
                &[SP1Field::one()],
            ),
        };
        let data = test_data(shape, Some(upstream));
        let output = reference_leaf(shape, &data);
        let expected =
            compute_host_commitments(shape, arguments.layer, arguments.head, &data, &output);
        let program = compile_test_leaf(shape, true);

        let (succeeded, digest) = execute_test_leaf(program.clone(), &arguments, &data);
        assert!(succeeded);
        assert_eq!(digest, expected.transcript);

        let mut modified_input = data.clone();
        modified_input.hidden_states[0] ^= 1;
        assert!(
            !execute_test_leaf(program.clone(), &arguments, &modified_input).0,
            "input differing from the previous block commitment must fail"
        );

        let mut modified_output_commitment = data.clone();
        modified_output_commitment.upstream.as_mut().unwrap().output[0] += SP1Field::one();
        assert!(
            !execute_test_leaf(program.clone(), &arguments, &modified_output_commitment).0,
            "incorrect previous block output commitment must fail"
        );

        let mut modified_transcript = data;
        modified_transcript.upstream.as_mut().unwrap().transcript[0] += SP1Field::one();
        let (succeeded, modified_digest) =
            execute_test_leaf(program, &arguments, &modified_transcript);
        assert!(succeeded);
        assert_ne!(modified_digest, expected.transcript);
    }

    #[test]
    fn genesis_leaf_keeps_the_original_transcript_layout() {
        let shape = Shape::small();
        let arguments = test_arguments(shape, 0);
        let data = test_data(shape, None);
        let output = reference_leaf(shape, &data);
        let expected =
            compute_host_commitments(shape, arguments.layer, arguments.head, &data, &output);
        let program = compile_test_leaf(shape, false);
        let (succeeded, digest) = execute_test_leaf(program, &arguments, &data);
        assert!(succeeded);
        assert_eq!(digest, expected.transcript);
    }
}
