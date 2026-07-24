#![allow(clippy::print_stdout)]

use std::{
    borrow::{Borrow, BorrowMut},
    fs::{self, File},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_basefold::{BasefoldVerifier, FriConfig};
use slop_symmetric::Permutation;
use sp1_hypercube::{
    air::MachineAir, inner_perm, prover::simple_prover, HashableKey, MachineProof,
    MachineVerifyingKey, SP1PcsProofInner, ShardProof, ShardVerifier, NUM_SP1_COMMITMENTS,
};
use sp1_primitives::{
    fri_params::{unique_decoding_queries, SP1_PROOF_OF_WORK_BITS},
    SP1DiffusionMatrix, SP1ExtensionField, SP1Field, SP1GlobalContext,
};
use sp1_recursion_circuit::{
    basefold::{
        stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs,
        RecursiveBasefoldVerifier,
    },
    challenger::CanObserveVariable,
    jagged::{RecursiveJaggedEvalSumcheckConfig, RecursiveJaggedPcsVerifier},
    shard::{MachineVerifyingKeyVariable, RecursiveShardVerifier, ShardProofVariable},
    witness::Witnessable,
    SP1FieldConfigVariable,
};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler, AsmConfig, CircuitV2Builder},
    prelude::{Builder, Felt, SymbolicFelt},
};
use sp1_recursion_executor::{
    event_shard_descriptor_digest, Block, EventRange, Executor, RecursionAirEventCount,
    RecursionEventRanges, RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE,
    PERMUTATION_WIDTH, RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::RecursionAir;

const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_NUM_HEADS: usize = 12;
const DEFAULT_LINEAR_SIZE: usize = 2304;
const PROOF_LOG_BLOWUP: usize = 1;
const PROOF_LOG_STACKING_HEIGHT: u32 = 22;
const PROOF_MAX_LOG_ROW_COUNT: usize = 28;

const BLOCK_PROTOCOL_VERSION: u32 = 2;

const RECURSION_PROTOCOL_VERSION: u32 = 3;
const RECURSION_STAGE: u32 = 0x1620;
const DOMAIN_RECURSION_TRANSCRIPT: u32 = 0x1621;
const NODE_KIND_BLOCK: u32 = 0x1622;
const NODE_KIND_RECURSION: u32 = 0x1623;
const EVENT_SHARD_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_4557;
const EVENT_BLOCK_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_4558;
const EVENT_SHARD_PROTOCOL_VERSION: u32 = 3;
const EVENT_BLOCK_RECURSION_PROTOCOL_VERSION: u32 = 2;
const EVENT_BLOCK_RECURSION_STAGE: &str = "zkgpt_block_shard_recursion";

type Digest = [SP1Field; DIGEST_SIZE];
type Air = RecursionAir<SP1Field, 3, 2>;
type StoredProof = MachineProof<SP1GlobalContext, SP1PcsProofInner>;
type StoredShardProof = ShardProof<SP1GlobalContext, SP1PcsProofInner>;
type StoredVerifyingKey = MachineVerifyingKey<SP1GlobalContext>;
type RecursiveVerifier = RecursiveShardVerifier<SP1GlobalContext, Air, AsmConfig>;
type VerifyingKeyVariable = MachineVerifyingKeyVariable<AsmConfig, SP1GlobalContext>;
type ProofVariable = ShardProofVariable<AsmConfig, SP1GlobalContext>;
type ChallengerVariable =
    <SP1GlobalContext as SP1FieldConfigVariable<AsmConfig>>::FriChallengerVariable;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Check,
    Build,
    Execute,
    Prove,
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    left_manifest: Option<PathBuf>,
    right_manifest: Option<PathBuf>,
    verify_manifest: Option<PathBuf>,
    event_shard_manifest: Option<PathBuf>,
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NodeKind {
    Block,
    Recursion,
}

impl NodeKind {
    const fn tag(self) -> u32 {
        match self {
            Self::Block => NODE_KIND_BLOCK,
            Self::Recursion => NODE_KIND_RECURSION,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Shape {
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            num_heads: DEFAULT_NUM_HEADS,
            linear_size: DEFAULT_LINEAR_SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NodeDescriptor {
    kind: NodeKind,
    start_block: usize,
    end_block: usize,
    transcript: Digest,
    verifying_key: Digest,
}

struct ChildArtifact {
    descriptor: NodeDescriptor,
    shape: Shape,
    manifest_path: PathBuf,
    verifying_key: StoredVerifyingKey,
    proof: StoredShardProof,
}

struct EventShardArtifact {
    index: usize,
    ranges: RecursionEventRanges,
    verifying_key: StoredVerifyingKey,
    proof: StoredShardProof,
    real_heights: Vec<(String, usize)>,
}

struct EventShardBatch {
    manifest_path: PathBuf,
    block: usize,
    shape: Shape,
    trace_transcript: Digest,
    transcript: Digest,
    private_output_file: String,
    global_counts: RecursionAirEventCount,
    shards: Vec<EventShardArtifact>,
}

#[derive(Debug, Deserialize)]
struct BlockManifest {
    version: u32,
    stage: String,
    block: usize,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
    shards: usize,
    explicit_value_commitments: bool,
    public_input_storage: String,
    private_parameters_storage: String,
    private_auxiliary_hints_storage: String,
    private_values_binding: String,
    proof_file: String,
    verifying_key_file: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestNodeDescriptor {
    kind: NodeKind,
    start_block: usize,
    end_block: usize,
    transcript_commitment: String,
    verifying_key_commitment: String,
}

#[derive(Debug, Deserialize)]
struct RecursionManifest {
    version: u32,
    stage: String,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
    shards: usize,
    start_block: usize,
    end_block: usize,
    transcript_commitment: String,
    verifying_key_commitment: String,
    left: ManifestNodeDescriptor,
    right: ManifestNodeDescriptor,
    proof_file: String,
    verifying_key_file: String,
}

#[derive(Debug, Deserialize)]
struct EventShardManifest {
    version: u32,
    stage: String,
    lookup_protocol: String,
    lookup_batch_closed: bool,
    transcript_binding: String,
    global_memory_accumulator: String,
    block: Option<usize>,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
    block_transcript_commitment: String,
    private_output_file: Option<String>,
    global_event_counts: EventCountManifest,
    shards: usize,
    artifacts: Vec<EventShardManifestArtifact>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct EventCountManifest {
    mem_const: usize,
    mem_var: usize,
    base_alu: usize,
    poseidon2_wide: usize,
    bf16_mul: usize,
    bf16_unary: usize,
    bf16_div: usize,
    bf16_add_sub: usize,
    commit_pv_hash: usize,
}

impl EventCountManifest {
    const fn into_event_counts(self) -> RecursionAirEventCount {
        RecursionAirEventCount {
            mem_const_events: self.mem_const,
            mem_var_events: self.mem_var,
            base_alu_events: self.base_alu,
            ext_alu_events: 0,
            ext_felt_conversion_events: 0,
            poseidon2_wide_events: self.poseidon2_wide,
            poseidon2_linear_layer_events: 0,
            poseidon2_sbox_events: 0,
            select_events: 0,
            bf16_mul_events: self.bf16_mul,
            bf16_unary_events: self.bf16_unary,
            bf16_div_events: self.bf16_div,
            bf16_add_sub_events: self.bf16_add_sub,
            prefix_sum_checks_events: 0,
            commit_pv_hash_events: self.commit_pv_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct EventRangeManifest {
    mem_const: [usize; 2],
    mem_var: [usize; 2],
    base_alu: [usize; 2],
    poseidon2_wide: [usize; 2],
    bf16_mul: [usize; 2],
    bf16_unary: [usize; 2],
    bf16_div: [usize; 2],
    bf16_add_sub: [usize; 2],
    commit_pv_hash: [usize; 2],
}

impl EventRangeManifest {
    const fn into_event_ranges(self) -> RecursionEventRanges {
        const fn range(value: [usize; 2]) -> EventRange {
            EventRange { start: value[0], end: value[1] }
        }
        RecursionEventRanges {
            mem_const: range(self.mem_const),
            mem_var: range(self.mem_var),
            base_alu: range(self.base_alu),
            poseidon2_wide: range(self.poseidon2_wide),
            bf16_mul: range(self.bf16_mul),
            bf16_unary: range(self.bf16_unary),
            bf16_div: range(self.bf16_div),
            bf16_add_sub: range(self.bf16_add_sub),
            commit_pv_hash: range(self.commit_pv_hash),
        }
    }
}

#[derive(Debug, Deserialize)]
struct EventShardManifestArtifact {
    index: usize,
    ranges: EventRangeManifest,
    proof_file: String,
    verifying_key_file: String,
}

#[derive(Debug, Deserialize)]
struct EventBlockRecursionManifest {
    version: u32,
    stage: String,
    block: usize,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
    linear_size: usize,
    shards: usize,
    trace_transcript_commitment: String,
    transcript_commitment: String,
    verifying_key_commitment: String,
    proof_file: String,
    verifying_key_file: String,
}

fn parse_arguments() -> Arguments {
    let mut mode = Mode::Check;
    let mut left_manifest = None;
    let mut right_manifest = None;
    let mut verify_manifest = None;
    let mut event_shard_manifest = None;
    let mut output_dir = std::env::var_os("SP1_ZKGPT_RECURSION_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-block-recursion"));
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--check") => mode = Mode::Check,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--left-manifest") => {
                left_manifest = Some(PathBuf::from(
                    arguments.next().expect("--left-manifest requires a value"),
                ));
            }
            Some("--right-manifest") => {
                right_manifest = Some(PathBuf::from(
                    arguments.next().expect("--right-manifest requires a value"),
                ));
            }
            Some("--verify-manifest") => {
                verify_manifest = Some(PathBuf::from(
                    arguments.next().expect("--verify-manifest requires a value"),
                ));
            }
            Some("--event-shard-manifest") => {
                event_shard_manifest = Some(PathBuf::from(
                    arguments.next().expect("--event-shard-manifest requires a value"),
                ));
            }
            Some("--output-dir") => {
                output_dir =
                    PathBuf::from(arguments.next().expect("--output-dir requires a value"));
            }
            Some(value) => panic!("unknown option: {value}"),
            None => panic!("command-line options must be valid UTF-8"),
        }
    }
    if verify_manifest.is_some() {
        assert!(
            left_manifest.is_none() && right_manifest.is_none() && event_shard_manifest.is_none(),
            "--verify-manifest cannot be combined with child manifests"
        );
        assert_eq!(mode, Mode::Check, "--verify-manifest cannot generate a new proof");
    } else if event_shard_manifest.is_some() {
        assert!(
            left_manifest.is_none() && right_manifest.is_none(),
            "--event-shard-manifest cannot be combined with child manifests"
        );
    } else {
        assert!(
            left_manifest.is_some() && right_manifest.is_some(),
            "--left-manifest and --right-manifest are required"
        );
    }
    Arguments {
        mode,
        left_manifest,
        right_manifest,
        verify_manifest,
        event_shard_manifest,
        output_dir,
    }
}

fn proof_fri_config() -> FriConfig<SP1Field> {
    FriConfig::new(
        PROOF_LOG_BLOWUP,
        unique_decoding_queries(PROOF_LOG_BLOWUP),
        SP1_PROOF_OF_WORK_BITS,
    )
}

fn parse_digest(value: &str, label: &str) -> Digest {
    let limbs = value
        .split(':')
        .map(|limb| {
            let limb = limb.strip_prefix("0x").unwrap_or(limb);
            u32::from_str_radix(limb, 16)
                .unwrap_or_else(|error| panic!("invalid {label} limb {limb}: {error}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(limbs.len(), DIGEST_SIZE, "{label} must contain {DIGEST_SIZE} limbs");
    limbs.into_iter().map(SP1Field::from_canonical_u32).collect::<Vec<_>>().try_into().unwrap()
}

fn digest_hex(digest: &Digest) -> String {
    digest
        .iter()
        .map(|value| format!("{:08X}", value.as_canonical_u32()))
        .collect::<Vec<_>>()
        .join(":")
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

fn circuit_commit_fields(
    builder: &mut Builder<AsmConfig>,
    domain: u32,
    values: &[Felt<SP1Field>],
) -> [Felt<SP1Field>; DIGEST_SIZE] {
    let zero: Felt<SP1Field> = builder.constant(SP1Field::zero());
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

fn descriptor_fields(descriptor: &NodeDescriptor) -> Vec<SP1Field> {
    let mut fields = vec![
        SP1Field::from_canonical_u32(descriptor.kind.tag()),
        SP1Field::from_canonical_usize(descriptor.start_block),
        SP1Field::from_canonical_usize(descriptor.end_block),
    ];
    fields.extend(descriptor.transcript);
    fields.extend(descriptor.verifying_key);
    fields
}

fn aggregate_transcript(shape: Shape, left: &NodeDescriptor, right: &NodeDescriptor) -> Digest {
    let mut fields = [
        RECURSION_PROTOCOL_VERSION,
        RECURSION_STAGE,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.linear_size as u32,
        left.start_block as u32,
        right.end_block as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(descriptor_fields(left));
    fields.extend(descriptor_fields(right));
    host_commit_fields(DOMAIN_RECURSION_TRANSCRIPT, &fields)
}

fn descriptor_from_manifest(value: &ManifestNodeDescriptor) -> NodeDescriptor {
    NodeDescriptor {
        kind: value.kind,
        start_block: value.start_block,
        end_block: value.end_block,
        transcript: parse_digest(&value.transcript_commitment, "node transcript"),
        verifying_key: parse_digest(&value.verifying_key_commitment, "node verifying key"),
    }
}

fn manifest_descriptor(value: NodeDescriptor) -> ManifestNodeDescriptor {
    ManifestNodeDescriptor {
        kind: value.kind,
        start_block: value.start_block,
        end_block: value.end_block,
        transcript_commitment: digest_hex(&value.transcript),
        verifying_key_commitment: digest_hex(&value.verifying_key),
    }
}

fn validate_adjacent(left: &NodeDescriptor, right: &NodeDescriptor) {
    assert_eq!(left.end_block, right.start_block, "child Block ranges are not adjacent");
}

fn safe_artifact_path(manifest_path: &Path, file: &str, label: &str) -> PathBuf {
    let relative = Path::new(file);
    assert!(
        !relative.is_absolute()
            && relative.file_name().is_some_and(|name| name == relative.as_os_str()),
        "{label} must be a plain relative file name"
    );
    manifest_path.parent().expect("manifest must have a parent").join(relative)
}

fn read_proof_artifacts(
    manifest_path: &Path,
    proof_file: &str,
    verifying_key_file: &str,
) -> (StoredProof, StoredVerifyingKey) {
    let proof_path = safe_artifact_path(manifest_path, proof_file, "proof_file");
    let vk_path = safe_artifact_path(manifest_path, verifying_key_file, "verifying_key_file");
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
    (proof, vk)
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

fn validate_event_shard_coverage(
    global_counts: RecursionAirEventCount,
    shards: &[EventShardArtifact],
) {
    assert!(!shards.is_empty(), "event shard batch cannot be empty");
    let mut expected = RecursionEventRanges::default();
    for (index, shard) in shards.iter().enumerate() {
        assert_eq!(shard.index, index, "event shards must be ordered");
        let ranges = shard.ranges;
        assert_eq!(ranges.mem_const.start, expected.mem_const.end);
        assert_eq!(ranges.mem_var.start, expected.mem_var.end);
        assert_eq!(ranges.base_alu.start, expected.base_alu.end);
        assert_eq!(ranges.poseidon2_wide.start, expected.poseidon2_wide.end);
        assert_eq!(ranges.bf16_mul.start, expected.bf16_mul.end);
        assert_eq!(ranges.bf16_unary.start, expected.bf16_unary.end);
        assert_eq!(ranges.bf16_div.start, expected.bf16_div.end);
        assert_eq!(ranges.bf16_add_sub.start, expected.bf16_add_sub.end);
        assert_eq!(ranges.commit_pv_hash.start, expected.commit_pv_hash.end);
        expected = ranges;
    }
    assert_eq!(
        expected,
        RecursionEventRanges::full(global_counts),
        "event shard ranges do not cover every global event exactly once"
    );
}

fn event_shard_transcript(
    shape: Shape,
    block: usize,
    global_counts: RecursionAirEventCount,
    shards: &[EventShardArtifact],
) -> Digest {
    let mut fields = [
        EVENT_SHARD_PROTOCOL_VERSION,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.linear_size as u32,
        block as u32,
        shards.len() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(event_count_fields(global_counts).map(SP1Field::from_canonical_usize));
    for shard in shards {
        fields.push(SP1Field::from_canonical_usize(shard.index));
        fields.extend(event_range_fields(shard.ranges).map(SP1Field::from_canonical_usize));
        fields.extend(shard.verifying_key.hash_koalabear());
        fields.extend(shard.proof.main_commitment);
        fields.push(SP1Field::from_canonical_usize(shard.real_heights.len()));
        for (name, height) in &shard.real_heights {
            fields.push(SP1Field::from_canonical_usize(name.len()));
            fields.extend(name.bytes().map(SP1Field::from_canonical_u8));
            fields.push(SP1Field::from_canonical_usize(*height));
        }
    }
    host_commit_fields(EVENT_SHARD_TRANSCRIPT_DOMAIN, &fields)
}

fn event_block_transcript(shape: Shape, block: usize, trace_transcript: Digest) -> Digest {
    let mut fields = [
        EVENT_BLOCK_RECURSION_PROTOCOL_VERSION,
        block as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.linear_size as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    fields.extend(trace_transcript);
    host_commit_fields(EVENT_BLOCK_TRANSCRIPT_DOMAIN, &fields)
}

fn load_event_shard_batch(manifest_path: &Path) -> EventShardBatch {
    let manifest: EventShardManifest = serde_json::from_slice(
        &fs::read(manifest_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display())),
    )
    .unwrap_or_else(|error| panic!("invalid event shard manifest: {error}"));
    assert_eq!(manifest.version, EVENT_SHARD_PROTOCOL_VERSION);
    assert_eq!(manifest.stage, "zkgpt_event_shards");
    assert_eq!(manifest.lookup_protocol, "independent_global_memory_accumulator");
    assert!(manifest.lookup_batch_closed);
    assert_eq!(
        manifest.transcript_binding,
        "global_event_counts_ordered_ranges_verifying_keys_main_trace_commitments_real_heights"
    );
    assert_eq!(manifest.global_memory_accumulator, "poseidon_vector_14");
    assert_eq!(manifest.shards, manifest.artifacts.len());
    let global_counts = manifest.global_event_counts.into_event_counts();
    let block = manifest.block.expect("block event shard manifest must contain a block index");
    let private_output_file = manifest
        .private_output_file
        .clone()
        .expect("block event shard manifest must contain its private output file");
    let _ = safe_artifact_path(manifest_path, &private_output_file, "private_output_file");
    let shape = Shape {
        sequence_length: manifest.sequence_length,
        hidden_size: manifest.hidden_size,
        num_heads: manifest.num_heads,
        linear_size: manifest.linear_size,
    };
    let manifest_transcript =
        parse_digest(&manifest.block_transcript_commitment, "block transcript commitment");

    let mut shards = Vec::with_capacity(manifest.artifacts.len());
    for (expected_index, artifact) in manifest.artifacts.iter().enumerate() {
        assert_eq!(artifact.index, expected_index, "event shards must be ordered");
        let (proof, verifying_key) =
            read_proof_artifacts(manifest_path, &artifact.proof_file, &artifact.verifying_key_file);
        assert_eq!(proof.shard_proofs.len(), 1, "event shard file must contain one proof");
        let proof = proof.shard_proofs.into_iter().next().unwrap();
        let real_heights = proof
            .opened_values
            .chips
            .iter()
            .map(|(name, values)| {
                (name.clone(), values.degree.bit_string_evaluation().as_canonical_u32() as usize)
            })
            .collect();
        shards.push(EventShardArtifact {
            index: expected_index,
            ranges: artifact.ranges.into_event_ranges(),
            verifying_key,
            proof,
            real_heights,
        });
    }
    validate_event_shard_coverage(global_counts, &shards);
    let trace_transcript = event_shard_transcript(shape, block, global_counts, &shards);
    assert_eq!(
        trace_transcript, manifest_transcript,
        "event shard trace transcript differs from the manifest"
    );
    let transcript = event_block_transcript(shape, block, trace_transcript);

    let verifier = ShardVerifier::from_basefold_parameters(
        proof_fri_config(),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        Air::verillm_machine(),
    );
    let prover = simple_prover(verifier);
    let mut global_accumulator = [SP1Field::zero(); 14];
    for shard in &shards {
        prover
            .verify(&shard.verifying_key, &MachineProof::from(vec![shard.proof.clone()]))
            .unwrap_or_else(|error| {
                panic!("event shard {} failed host verification: {error:?}", shard.index)
            });
        let public_values: &RecursionPublicValues<SP1Field> =
            shard.proof.public_values.as_slice().borrow();
        let expected_descriptor = event_shard_descriptor_digest(global_counts, shard.ranges);
        assert_eq!(
            public_values.digest, expected_descriptor,
            "event shard {} descriptor digest does not match its manifest range",
            shard.index
        );
        for (total, value) in global_accumulator.iter_mut().zip(
            public_values
                .global_cumulative_sum
                .0
                .x
                .0
                .iter()
                .chain(public_values.global_cumulative_sum.0.y.0.iter()),
        ) {
            *total += *value;
        }
    }
    assert!(
        global_accumulator.iter().all(Field::is_zero),
        "event shard global memory accumulators do not close"
    );
    println!(
        "verified independent event shards: block={} shards={} global_memory_sum=0 trace={} \
         block_transcript={}",
        block,
        shards.len(),
        digest_hex(&trace_transcript),
        digest_hex(&transcript)
    );
    EventShardBatch {
        manifest_path: manifest_path.to_owned(),
        block,
        shape,
        trace_transcript,
        transcript,
        private_output_file,
        global_counts,
        shards,
    }
}

fn host_verify(kind: NodeKind, vk: &StoredVerifyingKey, proof: &StoredProof) {
    let machine = match kind {
        NodeKind::Block => Air::verillm_machine(),
        NodeKind::Recursion => Air::compress_machine(),
    };
    let verifier = ShardVerifier::from_basefold_parameters(
        proof_fri_config(),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        machine,
    );
    simple_prover(verifier)
        .verify(vk, proof)
        .unwrap_or_else(|error| panic!("{kind:?} child proof failed host verification: {error:?}"));
}

fn load_child(manifest_path: &Path) -> ChildArtifact {
    let bytes = fs::read(manifest_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display()));
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("invalid JSON {}: {error}", manifest_path.display()));
    let stage = value
        .get("stage")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("{} is missing stage", manifest_path.display()))
        .to_owned();

    let (descriptor_without_vk, shape, proof, vk, expected_vk) = match stage.as_str() {
        "zkgpt_block" => {
            let manifest: BlockManifest = serde_json::from_value(value)
                .unwrap_or_else(|error| panic!("invalid Block manifest: {error}"));
            assert_eq!(manifest.version, BLOCK_PROTOCOL_VERSION, "unsupported Block protocol");
            assert_eq!(manifest.stage, "zkgpt_block", "invalid Block stage");
            assert_eq!(manifest.shards, 1, "a Block proof must contain one shard");
            assert!(
                !manifest.explicit_value_commitments,
                "Block must not contain explicit value commitments"
            );
            assert_eq!(manifest.public_input_storage, "memory_const");
            assert_eq!(manifest.private_parameters_storage, "hint");
            assert_eq!(manifest.private_auxiliary_hints_storage, "hint");
            assert_eq!(manifest.private_values_binding, "execution_trace_commitment");
            let shape = Shape {
                sequence_length: manifest.sequence_length,
                hidden_size: manifest.hidden_size,
                num_heads: manifest.num_heads,
                linear_size: manifest.linear_size,
            };
            let (proof, vk) = read_proof_artifacts(
                manifest_path,
                &manifest.proof_file,
                &manifest.verifying_key_file,
            );
            assert_eq!(proof.shard_proofs.len(), 1, "Block proof must contain one shard");
            let public_values: &RecursionPublicValues<SP1Field> =
                proof.shard_proofs[0].public_values.as_slice().borrow();
            let descriptor = NodeDescriptor {
                kind: NodeKind::Block,
                start_block: manifest.block,
                end_block: manifest.block + 1,
                transcript: public_values.digest,
                verifying_key: [SP1Field::zero(); DIGEST_SIZE],
            };
            (descriptor, shape, proof, vk, None)
        }
        "zkgpt_block_recursion" => {
            let manifest: RecursionManifest = serde_json::from_value(value)
                .unwrap_or_else(|error| panic!("invalid recursion manifest: {error}"));
            assert_eq!(
                manifest.version, RECURSION_PROTOCOL_VERSION,
                "unsupported recursion protocol"
            );
            assert_eq!(manifest.stage, "zkgpt_block_recursion", "invalid recursion stage");
            assert_eq!(manifest.shards, 1, "a recursion proof must contain one shard");
            let shape = Shape {
                sequence_length: manifest.sequence_length,
                hidden_size: manifest.hidden_size,
                num_heads: manifest.num_heads,
                linear_size: manifest.linear_size,
            };
            let left = descriptor_from_manifest(&manifest.left);
            let right = descriptor_from_manifest(&manifest.right);
            validate_adjacent(&left, &right);
            assert_eq!(manifest.start_block, left.start_block);
            assert_eq!(manifest.end_block, right.end_block);
            let transcript = aggregate_transcript(shape, &left, &right);
            assert_eq!(
                transcript,
                parse_digest(&manifest.transcript_commitment, "recursion transcript"),
                "recursion manifest transcript is inconsistent"
            );
            let descriptor = NodeDescriptor {
                kind: NodeKind::Recursion,
                start_block: manifest.start_block,
                end_block: manifest.end_block,
                transcript,
                verifying_key: [SP1Field::zero(); DIGEST_SIZE],
            };
            let (proof, vk) = read_proof_artifacts(
                manifest_path,
                &manifest.proof_file,
                &manifest.verifying_key_file,
            );
            (
                descriptor,
                shape,
                proof,
                vk,
                Some(parse_digest(&manifest.verifying_key_commitment, "recursion verifying key")),
            )
        }
        EVENT_BLOCK_RECURSION_STAGE => {
            let manifest: EventBlockRecursionManifest = serde_json::from_value(value)
                .unwrap_or_else(|error| panic!("invalid event block recursion manifest: {error}"));
            assert_eq!(
                manifest.version, EVENT_BLOCK_RECURSION_PROTOCOL_VERSION,
                "unsupported event block recursion protocol"
            );
            assert_eq!(manifest.stage, EVENT_BLOCK_RECURSION_STAGE);
            assert_eq!(manifest.shards, 1, "event block recursion proof must contain one shard");
            let shape = Shape {
                sequence_length: manifest.sequence_length,
                hidden_size: manifest.hidden_size,
                num_heads: manifest.num_heads,
                linear_size: manifest.linear_size,
            };
            let trace_transcript =
                parse_digest(&manifest.trace_transcript_commitment, "event block trace transcript");
            let transcript = event_block_transcript(shape, manifest.block, trace_transcript);
            assert_eq!(
                transcript,
                parse_digest(&manifest.transcript_commitment, "event block transcript"),
                "event block number or shape differs from its transcript"
            );
            let descriptor = NodeDescriptor {
                kind: NodeKind::Recursion,
                start_block: manifest.block,
                end_block: manifest.block + 1,
                transcript,
                verifying_key: [SP1Field::zero(); DIGEST_SIZE],
            };
            let (proof, vk) = read_proof_artifacts(
                manifest_path,
                &manifest.proof_file,
                &manifest.verifying_key_file,
            );
            (
                descriptor,
                shape,
                proof,
                vk,
                Some(parse_digest(
                    &manifest.verifying_key_commitment,
                    "event block recursion verifying key",
                )),
            )
        }
        other => panic!("unsupported child stage {other:?} in {}", manifest_path.display()),
    };

    assert_eq!(proof.shard_proofs.len(), 1, "child proof must contain exactly one shard");
    host_verify(descriptor_without_vk.kind, &vk, &proof);
    let verifying_key = vk.hash_koalabear();
    if let Some(expected) = expected_vk {
        assert_eq!(verifying_key, expected, "recursion verifying key differs from its manifest");
    }
    let public_values: &RecursionPublicValues<SP1Field> =
        proof.shard_proofs[0].public_values.as_slice().borrow();
    assert_eq!(
        public_values.digest, descriptor_without_vk.transcript,
        "child proof public digest differs from its manifest"
    );
    let mut descriptor = descriptor_without_vk;
    descriptor.verifying_key = verifying_key;
    println!(
        "verified child: kind={:?} range={}..{} transcript={} vk={}",
        descriptor.kind,
        descriptor.start_block,
        descriptor.end_block,
        digest_hex(&descriptor.transcript),
        digest_hex(&descriptor.verifying_key)
    );
    ChildArtifact {
        descriptor,
        shape,
        manifest_path: manifest_path.to_owned(),
        verifying_key: vk,
        proof: proof.shard_proofs.into_iter().next().unwrap(),
    }
}

fn recursive_verifier(kind: NodeKind) -> RecursiveVerifier {
    let basefold =
        BasefoldVerifier::<SP1GlobalContext>::new(proof_fri_config(), NUM_SP1_COMMITMENTS);
    let recursive_basefold = RecursiveBasefoldVerifier::<AsmConfig, SP1GlobalContext> {
        fri_config: basefold.fri_config,
        tcs: RecursiveMerkleTreeTcs(PhantomData),
    };
    let stacked = RecursiveStackedPcsVerifier::new(recursive_basefold, PROOF_LOG_STACKING_HEIGHT);
    let pcs_verifier = RecursiveJaggedPcsVerifier {
        stacked_pcs_verifier: stacked,
        max_log_row_count: PROOF_MAX_LOG_ROW_COUNT,
        jagged_evaluator: RecursiveJaggedEvalSumcheckConfig(PhantomData),
    };
    let machine = match kind {
        NodeKind::Block => Air::verillm_machine(),
        NodeKind::Recursion => Air::compress_machine(),
    };
    RecursiveShardVerifier { machine, pcs_verifier, _phantom: PhantomData }
}

fn observe_vk_variable(
    builder: &mut Builder<AsmConfig>,
    challenger: &mut ChallengerVariable,
    vk: &VerifyingKeyVariable,
) {
    challenger.observe(builder, vk.preprocessed_commit);
    challenger.observe_slice(builder, vk.pc_start);
    challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.x.0);
    challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.y.0);
    challenger.observe(builder, vk.untrusted_config.enable_untrusted_programs);
    let zero: Felt<SP1Field> = builder.constant(SP1Field::zero());
    for _ in 0..6 {
        challenger.observe(builder, zero);
    }
}

fn proof_height_felts(
    builder: &mut Builder<AsmConfig>,
    proof: &ProofVariable,
) -> Vec<(String, Felt<SP1Field>)> {
    let two = SymbolicFelt::from_canonical_u32(2);
    proof
        .opened_values
        .chips
        .iter()
        .map(|(name, values)| {
            let mut height = SymbolicFelt::zero();
            values.degree.iter().for_each(|bit| {
                height = *bit + two * height;
            });
            (name.clone(), builder.eval(height))
        })
        .collect()
}

fn event_shard_transcript_in_circuit(
    builder: &mut Builder<AsmConfig>,
    batch: &EventShardBatch,
    vks: &[VerifyingKeyVariable],
    proofs: &[ProofVariable],
) -> [Felt<SP1Field>; DIGEST_SIZE] {
    assert_eq!(vks.len(), proofs.len());
    let mut fields = [
        EVENT_SHARD_PROTOCOL_VERSION,
        batch.shape.sequence_length as u32,
        batch.shape.hidden_size as u32,
        batch.shape.num_heads as u32,
        batch.shape.linear_size as u32,
        batch.block as u32,
        proofs.len() as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)))
    .to_vec();
    fields.extend(event_count_fields(batch.global_counts).map(|value| -> Felt<SP1Field> {
        builder.constant(SP1Field::from_canonical_usize(value))
    }));
    for (index, ((vk, proof), shard)) in vks.iter().zip(proofs).zip(&batch.shards).enumerate() {
        fields.push(builder.constant(SP1Field::from_canonical_usize(index)));
        fields.extend(event_range_fields(shard.ranges).map(|value| -> Felt<SP1Field> {
            builder.constant(SP1Field::from_canonical_usize(value))
        }));
        fields.extend(vk.hash(builder));
        fields.extend(proof.main_commitment);
        let heights = proof_height_felts(builder, proof);
        fields.push(builder.constant(SP1Field::from_canonical_usize(heights.len())));
        for (name, height) in heights {
            fields.push(builder.constant(SP1Field::from_canonical_usize(name.len())));
            for byte in name.bytes() {
                fields.push(builder.constant(SP1Field::from_canonical_u8(byte)));
            }
            fields.push(height);
        }
    }
    circuit_commit_fields(builder, EVENT_SHARD_TRANSCRIPT_DOMAIN, &fields)
}

fn event_block_transcript_in_circuit(
    builder: &mut Builder<AsmConfig>,
    batch: &EventShardBatch,
    trace_transcript: [Felt<SP1Field>; DIGEST_SIZE],
) -> [Felt<SP1Field>; DIGEST_SIZE] {
    let mut fields = [
        EVENT_BLOCK_RECURSION_PROTOCOL_VERSION,
        batch.block as u32,
        batch.shape.sequence_length as u32,
        batch.shape.hidden_size as u32,
        batch.shape.num_heads as u32,
        batch.shape.linear_size as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)))
    .to_vec();
    fields.extend(trace_transcript);
    circuit_commit_fields(builder, EVENT_BLOCK_TRANSCRIPT_DOMAIN, &fields)
}

fn verify_child_in_circuit(
    builder: &mut Builder<AsmConfig>,
    artifact: &ChildArtifact,
    vk: &VerifyingKeyVariable,
    proof: &ProofVariable,
) -> ([Felt<SP1Field>; DIGEST_SIZE], [Felt<SP1Field>; DIGEST_SIZE]) {
    let mut challenger = SP1GlobalContext::challenger_variable(builder);
    observe_vk_variable(builder, &mut challenger, vk);
    recursive_verifier(artifact.descriptor.kind).verify_shard(builder, vk, proof, &mut challenger);

    let vk_digest = vk.hash(builder);
    for (&actual, expected) in vk_digest.iter().zip(artifact.descriptor.verifying_key) {
        builder.assert_felt_eq(actual, expected);
    }
    let public_values: &RecursionPublicValues<Felt<SP1Field>> =
        proof.public_values.as_slice().borrow();
    for (&actual, expected) in public_values.digest.iter().zip(artifact.descriptor.transcript) {
        builder.assert_felt_eq(actual, expected);
    }
    (public_values.digest, vk_digest)
}

fn circuit_descriptor_fields(
    builder: &mut Builder<AsmConfig>,
    descriptor: &NodeDescriptor,
    transcript: [Felt<SP1Field>; DIGEST_SIZE],
    verifying_key: [Felt<SP1Field>; DIGEST_SIZE],
) -> Vec<Felt<SP1Field>> {
    let mut fields =
        [descriptor.kind.tag(), descriptor.start_block as u32, descriptor.end_block as u32]
            .map(|value| builder.constant(SP1Field::from_canonical_u32(value)))
            .to_vec();
    fields.extend(transcript);
    fields.extend(verifying_key);
    fields
}

fn build_program(
    left: &ChildArtifact,
    right: &ChildArtifact,
) -> (Arc<RecursionProgram<SP1Field>>, Vec<Block<SP1Field>>, Digest) {
    assert_eq!(left.shape, right.shape, "children use different GPT-2 shapes");
    validate_adjacent(&left.descriptor, &right.descriptor);
    let expected = aggregate_transcript(left.shape, &left.descriptor, &right.descriptor);

    let mut builder: Builder<AsmConfig> = AsmBuilder::default();
    let left_vk = left.verifying_key.read(&mut builder);
    let left_proof = left.proof.read(&mut builder);
    let right_vk = right.verifying_key.read(&mut builder);
    let right_proof = right.proof.read(&mut builder);

    let (left_transcript, left_vk_digest) =
        verify_child_in_circuit(&mut builder, left, &left_vk, &left_proof);
    let (right_transcript, right_vk_digest) =
        verify_child_in_circuit(&mut builder, right, &right_vk, &right_proof);

    let mut fields = [
        RECURSION_PROTOCOL_VERSION,
        RECURSION_STAGE,
        left.shape.sequence_length as u32,
        left.shape.hidden_size as u32,
        left.shape.num_heads as u32,
        left.shape.linear_size as u32,
        left.descriptor.start_block as u32,
        right.descriptor.end_block as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)))
    .to_vec();
    fields.extend(circuit_descriptor_fields(
        &mut builder,
        &left.descriptor,
        left_transcript,
        left_vk_digest,
    ));
    fields.extend(circuit_descriptor_fields(
        &mut builder,
        &right.descriptor,
        right_transcript,
        right_vk_digest,
    ));
    let transcript = circuit_commit_fields(&mut builder, DOMAIN_RECURSION_TRANSCRIPT, &fields);
    for (&actual, expected) in transcript.iter().zip(expected) {
        builder.assert_felt_eq(actual, expected);
    }

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = transcript;
    builder.commit_public_values_v2(*public_values);

    let root = builder.into_root_block();
    println!("built recursive join: ir_ops={}", root.ops.len());
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(root).validate().unwrap());

    let mut witness = Vec::new();
    Witnessable::<AsmConfig>::write(&left.verifying_key, &mut witness);
    Witnessable::<AsmConfig>::write(&left.proof, &mut witness);
    Witnessable::<AsmConfig>::write(&right.verifying_key, &mut witness);
    Witnessable::<AsmConfig>::write(&right.proof, &mut witness);
    (program, witness, expected)
}

fn build_event_block_program(
    batch: &EventShardBatch,
) -> (Arc<RecursionProgram<SP1Field>>, Vec<Block<SP1Field>>) {
    let mut builder: Builder<AsmConfig> = AsmBuilder::default();
    let mut vks = Vec::with_capacity(batch.shards.len());
    let mut proofs = Vec::with_capacity(batch.shards.len());
    for shard in &batch.shards {
        vks.push(shard.verifying_key.read(&mut builder));
        proofs.push(shard.proof.read(&mut builder));
    }

    let verifier = recursive_verifier(NodeKind::Block);
    let mut global_accumulator = vec![SymbolicFelt::zero(); 14];
    for ((vk, proof), shard) in vks.iter().zip(&proofs).zip(&batch.shards) {
        let mut challenger = SP1GlobalContext::challenger_variable(&mut builder);
        observe_vk_variable(&mut builder, &mut challenger, vk);
        verifier.verify_shard(&mut builder, vk, proof, &mut challenger);

        let public_values: &RecursionPublicValues<Felt<SP1Field>> =
            proof.public_values.as_slice().borrow();
        let expected_descriptor = event_shard_descriptor_digest(batch.global_counts, shard.ranges);
        for (&actual, expected) in public_values.digest.iter().zip(expected_descriptor) {
            builder.assert_felt_eq(actual, expected);
        }
        for (total, value) in global_accumulator.iter_mut().zip(
            public_values
                .global_cumulative_sum
                .0
                .x
                .0
                .iter()
                .chain(public_values.global_cumulative_sum.0.y.0.iter()),
        ) {
            *total = *total + *value;
        }
    }
    for coordinate in global_accumulator {
        builder.assert_felt_eq(coordinate, SymbolicFelt::zero());
    }

    let trace_transcript = event_shard_transcript_in_circuit(&mut builder, batch, &vks, &proofs);
    for (&actual, expected) in trace_transcript.iter().zip(batch.trace_transcript) {
        builder.assert_felt_eq(actual, expected);
    }
    let transcript = event_block_transcript_in_circuit(&mut builder, batch, trace_transcript);
    for (&actual, expected) in transcript.iter().zip(batch.transcript) {
        builder.assert_felt_eq(actual, expected);
    }

    let zero = builder.constant(SP1Field::zero());
    let mut output_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let output: &mut RecursionPublicValues<Felt<SP1Field>> =
        output_elements.as_mut_slice().borrow_mut();
    output.digest = transcript;
    builder.commit_public_values_v2(*output);

    let root = builder.into_root_block();
    println!(
        "built event block recursion: shards={} ir_ops={}",
        batch.shards.len(),
        root.ops.len()
    );
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(root).validate().unwrap());
    let mut witness = Vec::new();
    for shard in &batch.shards {
        Witnessable::<AsmConfig>::write(&shard.verifying_key, &mut witness);
        Witnessable::<AsmConfig>::write(&shard.proof, &mut witness);
    }
    (program, witness)
}

fn recursion_stem(start_block: usize, end_block: usize) -> String {
    format!("zkgpt_recursion_b{start_block:02}_b{end_block:02}")
}

fn write_manifest(
    arguments: &Arguments,
    left: &ChildArtifact,
    right: &ChildArtifact,
    transcript: Digest,
    proof_path: &Path,
    vk_path: &Path,
    vk: &StoredVerifyingKey,
) {
    let start_block = left.descriptor.start_block;
    let end_block = right.descriptor.end_block;
    let manifest_path = arguments
        .output_dir
        .join(format!("{}.manifest.json", recursion_stem(start_block, end_block)));
    let manifest = serde_json::json!({
        "version": RECURSION_PROTOCOL_VERSION,
        "stage": "zkgpt_block_recursion",
        "sequence_length": left.shape.sequence_length,
        "hidden_size": left.shape.hidden_size,
        "num_heads": left.shape.num_heads,
        "linear_size": left.shape.linear_size,
        "shards": 1,
        "start_block": start_block,
        "end_block": end_block,
        "transcript_commitment": digest_hex(&transcript),
        "verifying_key_commitment": digest_hex(&vk.hash_koalabear()),
        "left_manifest": left.manifest_path.display().to_string(),
        "right_manifest": right.manifest_path.display().to_string(),
        "left": manifest_descriptor(left.descriptor),
        "right": manifest_descriptor(right.descriptor),
        "proof_file": proof_path.file_name().unwrap().to_string_lossy(),
        "proof_bytes": fs::metadata(proof_path).unwrap().len(),
        "verifying_key_file": vk_path.file_name().unwrap().to_string_lossy(),
        "verifying_key_bytes": fs::metadata(vk_path).unwrap().len(),
    });
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap())
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
    println!("recursion manifest: {}", manifest_path.display());
}

fn write_event_block_manifest(
    arguments: &Arguments,
    batch: &EventShardBatch,
    proof_path: &Path,
    vk_path: &Path,
    vk: &StoredVerifyingKey,
) {
    let stem = format!("zkgpt_block_{:02}_shard_recursion", batch.block);
    let manifest_path = arguments.output_dir.join(format!("{stem}.manifest.json"));
    let source_private_output =
        safe_artifact_path(&batch.manifest_path, &batch.private_output_file, "private_output_file");
    let persisted_private_output = arguments.output_dir.join(&batch.private_output_file);
    if source_private_output != persisted_private_output {
        fs::copy(&source_private_output, &persisted_private_output).unwrap_or_else(|error| {
            panic!(
                "failed to copy {} to {}: {error}",
                source_private_output.display(),
                persisted_private_output.display()
            )
        });
    }
    let manifest = serde_json::json!({
        "version": EVENT_BLOCK_RECURSION_PROTOCOL_VERSION,
        "stage": EVENT_BLOCK_RECURSION_STAGE,
        "block": batch.block,
        "sequence_length": batch.shape.sequence_length,
        "hidden_size": batch.shape.hidden_size,
        "num_heads": batch.shape.num_heads,
        "linear_size": batch.shape.linear_size,
        "shards": 1,
        "source_event_shards": batch.shards.len(),
        "source_manifest": batch.manifest_path.display().to_string(),
        "private_output_file": batch.private_output_file,
        "trace_transcript_commitment": digest_hex(&batch.trace_transcript),
        "transcript_commitment": digest_hex(&batch.transcript),
        "verifying_key_commitment": digest_hex(&vk.hash_koalabear()),
        "proof_file": proof_path.file_name().unwrap().to_string_lossy(),
        "proof_bytes": fs::metadata(proof_path).unwrap().len(),
        "verifying_key_file": vk_path.file_name().unwrap().to_string_lossy(),
        "verifying_key_bytes": fs::metadata(vk_path).unwrap().len(),
    });
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap())
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", manifest_path.display()));
    println!("event block recursion manifest: {}", manifest_path.display());
}

async fn run_event_shard_batch(arguments: &Arguments, manifest_path: &Path) {
    let started = Instant::now();
    let batch = load_event_shard_batch(manifest_path);
    if arguments.mode == Mode::Check {
        println!("event shard batch host verification completed");
        return;
    }

    let build_started = Instant::now();
    let (program, witness) = build_event_block_program(&batch);
    println!(
        "compiled event block recursion: elapsed={:.3}s",
        build_started.elapsed().as_secs_f64()
    );
    if arguments.mode == Mode::Build {
        return;
    }

    let execute_started = Instant::now();
    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.witness_stream = witness.into();
    executor.run().expect("valid event block recursion witness must execute");
    assert_eq!(executor.record.public_values.digest, batch.transcript);
    println!(
        "executed event block recursion: elapsed={:.3}s transcript={}",
        execute_started.elapsed().as_secs_f64(),
        digest_hex(&batch.transcript)
    );
    if arguments.mode == Mode::Execute {
        return;
    }

    let record = std::mem::take(&mut executor.record);
    drop(executor);
    let machine = Air::compress_machine();
    let max_rows = 1usize << PROOF_MAX_LOG_ROW_COUNT;
    for chip in machine.chips() {
        if chip.included(&record) {
            let rows = chip.num_rows(&record).unwrap_or_default();
            println!("trace: {:<24} rows={rows}", chip.name());
            assert!(rows <= max_rows, "{} exceeds the proof row limit", chip.name());
        }
    }
    let verifier = ShardVerifier::from_basefold_parameters(
        proof_fri_config(),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        machine,
    );
    let prover = simple_prover(verifier);
    let shape = prover.shape_from_record(&record).expect("event block recursion has no shape");
    println!("event block recursion proof shape: {shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    println!("recursion setup: elapsed={:.3}s", setup_started.elapsed().as_secs_f64());
    let pk = unsafe { pk.into_inner() };
    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    println!("recursion proof generated: elapsed={:.3}s", prove_started.elapsed().as_secs_f64());
    prover.verify(&vk, &proof).expect("event block recursion proof must verify");

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_block_{:02}_shard_recursion", batch.block);
    let proof_path = arguments.output_dir.join(format!("{stem}.proof.bin"));
    let vk_path = arguments.output_dir.join(format!("{stem}.vk.bin"));
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    write_event_block_manifest(arguments, &batch, &proof_path, &vk_path, &vk);
    println!(
        "event block recursion completed: elapsed={:.3}s proof={} bytes",
        started.elapsed().as_secs_f64(),
        fs::metadata(proof_path).unwrap().len()
    );
}

async fn run(arguments: Arguments) {
    let started = Instant::now();
    if let Some(manifest) = &arguments.verify_manifest {
        let artifact = load_child(manifest);
        assert_eq!(
            artifact.descriptor.kind,
            NodeKind::Recursion,
            "--verify-manifest requires a recursion proof"
        );
        println!(
            "verified final recursion proof: range={}..{} transcript={} vk={}",
            artifact.descriptor.start_block,
            artifact.descriptor.end_block,
            digest_hex(&artifact.descriptor.transcript),
            digest_hex(&artifact.descriptor.verifying_key)
        );
        return;
    }
    if let Some(manifest) = &arguments.event_shard_manifest {
        run_event_shard_batch(&arguments, manifest).await;
        return;
    }
    let left = load_child(
        arguments
            .left_manifest
            .as_deref()
            .expect("left manifest was validated during argument parsing"),
    );
    let right = load_child(
        arguments
            .right_manifest
            .as_deref()
            .expect("right manifest was validated during argument parsing"),
    );
    assert_eq!(left.shape, right.shape, "children use different GPT-2 shapes");
    validate_adjacent(&left.descriptor, &right.descriptor);
    println!(
        "recursive join: left={}..{} right={}..{} output={}..{}",
        left.descriptor.start_block,
        left.descriptor.end_block,
        right.descriptor.start_block,
        right.descriptor.end_block,
        left.descriptor.start_block,
        right.descriptor.end_block
    );
    if arguments.mode == Mode::Check {
        println!("child proofs and adjacent Block ranges verified");
        return;
    }

    let build_started = Instant::now();
    let (program, witness, expected_transcript) = build_program(&left, &right);
    println!("compiled recursive join: elapsed={:.3}s", build_started.elapsed().as_secs_f64());
    if arguments.mode == Mode::Build {
        return;
    }

    let execute_started = Instant::now();
    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.witness_stream = witness.into();
    executor.run().expect("valid recursive join witness must execute");
    assert_eq!(
        executor.record.public_values.digest, expected_transcript,
        "recursive circuit transcript differs from host transcript"
    );
    println!(
        "executed recursive join: elapsed={:.3}s transcript={}",
        execute_started.elapsed().as_secs_f64(),
        digest_hex(&expected_transcript)
    );
    if arguments.mode == Mode::Execute {
        return;
    }

    let record = std::mem::take(&mut executor.record);
    drop(executor);
    let machine = Air::compress_machine();
    let max_rows = 1usize << PROOF_MAX_LOG_ROW_COUNT;
    for chip in machine.chips() {
        if chip.included(&record) {
            let rows = chip.num_rows(&record).unwrap_or_default();
            println!("trace: {:<24} rows={rows}", chip.name());
            assert!(rows <= max_rows, "{} exceeds the proof row limit", chip.name());
        }
    }
    let verifier = ShardVerifier::from_basefold_parameters(
        proof_fri_config(),
        PROOF_LOG_STACKING_HEIGHT,
        PROOF_MAX_LOG_ROW_COUNT,
        machine,
    );
    let prover = simple_prover(verifier);
    let shape = prover.shape_from_record(&record).expect("recursion proof has no shape");
    println!("recursion proof shape: {shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    println!("recursion setup: elapsed={:.3}s", setup_started.elapsed().as_secs_f64());
    let pk = unsafe { pk.into_inner() };
    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    println!("recursion proof generated: elapsed={:.3}s", prove_started.elapsed().as_secs_f64());
    prover.verify(&vk, &proof).expect("generated recursion proof must verify");

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = recursion_stem(left.descriptor.start_block, right.descriptor.end_block);
    let proof_path = arguments.output_dir.join(format!("{stem}.proof.bin"));
    let vk_path = arguments.output_dir.join(format!("{stem}.vk.bin"));
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    write_manifest(&arguments, &left, &right, expected_transcript, &proof_path, &vk_path, &vk);
    println!(
        "recursive join completed: elapsed={:.3}s proof={} bytes",
        started.elapsed().as_secs_f64(),
        fs::metadata(proof_path).unwrap().len()
    );
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    run(parse_arguments()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u32) -> Digest {
        std::array::from_fn(|index| SP1Field::from_canonical_u32(seed + index as u32))
    }

    fn node(kind: NodeKind, start: usize, end: usize, seed: u32) -> NodeDescriptor {
        NodeDescriptor {
            kind,
            start_block: start,
            end_block: end,
            transcript: digest(seed),
            verifying_key: digest(seed + 10),
        }
    }

    #[test]
    fn aggregate_transcript_binds_child_kind_and_range() {
        let shape = Shape::default();
        let left = node(NodeKind::Block, 0, 1, 100);
        let mut right = node(NodeKind::Block, 1, 2, 200);
        let expected = aggregate_transcript(shape, &left, &right);
        right.kind = NodeKind::Recursion;
        assert_ne!(expected, aggregate_transcript(shape, &left, &right));
    }

    #[test]
    fn event_block_transcript_binds_block_and_shape() {
        let shape = Shape::default();
        let trace = digest(300);
        let expected = event_block_transcript(shape, 0, trace);
        assert_ne!(expected, event_block_transcript(shape, 1, trace));
        assert_ne!(
            expected,
            event_block_transcript(Shape { sequence_length: 1, ..shape }, 0, trace)
        );
    }

    #[test]
    #[should_panic(expected = "ranges are not adjacent")]
    fn adjacent_ranges_reject_a_gap() {
        let left = node(NodeKind::Block, 0, 1, 100);
        let right = node(NodeKind::Block, 2, 3, 200);
        validate_adjacent(&left, &right);
    }
}
