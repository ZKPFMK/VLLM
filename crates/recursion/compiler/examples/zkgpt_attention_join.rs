#![allow(clippy::print_stdout)]

use std::{
    borrow::{Borrow, BorrowMut},
    collections::BTreeMap,
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
    Executor, RecursionProgram, RecursionPublicValues, DIGEST_SIZE, HASH_RATE, PERMUTATION_WIDTH,
    RECURSIVE_PROOF_NUM_PV_ELTS,
};
use sp1_recursion_machine::RecursionAir;

const DEFAULT_SEQUENCE_LENGTH: usize = 30;
const DEFAULT_HIDDEN_SIZE: usize = 768;
const DEFAULT_NUM_HEADS: usize = 12;
const DEFAULT_LAYER: usize = 0;

// These constants must remain identical to `zkgpt_leaf`.
const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;
const GPT2_ATTENTION_SCALE: u16 = 0x3e00;
const LEAF_VERSION: u32 = 1;
const LEAF_STAGE_QKV_ATTENTION: u32 = 1;
const ATTENTION_GROUP_STAGE: u32 = 2;
const DOMAIN_OUTPUT: u32 = 0x1004;
const DOMAIN_TRANSCRIPT: u32 = 0x1005;
const DOMAIN_ATTENTION_GROUP_PARAMETERS: u32 = 0x1101;
const DOMAIN_ATTENTION_GROUP_HINTS: u32 = 0x1102;
const DOMAIN_ATTENTION_GROUP_OUTPUT: u32 = 0x1103;
const DOMAIN_ATTENTION_GROUP_TRANSCRIPT: u32 = 0x1104;

const PROOF_LOG_BLOWUP: usize = 1;
const JOIN_MAX_LOG_ROWS: usize = 16;
const FULL_LEAF_MAX_LOG_ROWS: usize = 19;
const SMALL_LEAF_MAX_LOG_ROWS: usize = 16;

type JoinBuilder = Builder<AsmConfig>;
type Digest = [SP1Field; DIGEST_SIZE];
type StoredLeafProof = MachineProof<SP1GlobalContext, SP1PcsProofInner>;
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
        assert!(self.sequence_length > 0, "join sequence length must be nonzero");
        assert!(self.hidden_size > 0, "join hidden size must be nonzero");
        assert!(self.num_heads > 0, "join head count must be nonzero");
        assert_eq!(
            self.hidden_size % self.num_heads,
            0,
            "join hidden size must be divisible by its head count"
        );
    }

    fn head_dimension(self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn leaf_max_log_rows(self) -> usize {
        if self.sequence_length == DEFAULT_SEQUENCE_LENGTH
            && self.hidden_size == DEFAULT_HIDDEN_SIZE
            && self.num_heads == DEFAULT_NUM_HEADS
        {
            FULL_LEAF_MAX_LOG_ROWS
        } else {
            SMALL_LEAF_MAX_LOG_ROWS
        }
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    shape: Shape,
    layer: usize,
    leaf_dir: PathBuf,
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct LeafManifest {
    layer: usize,
    head: usize,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Debug)]
struct JoinHeadData {
    head: usize,
    parameters: Digest,
    hints: Digest,
    transcript: Digest,
    output: Vec<u16>,
}

#[derive(Clone, Debug)]
struct JoinData {
    input: Digest,
    heads: Vec<JoinHeadData>,
}

#[derive(Clone, Copy, Debug)]
struct GroupCommitments {
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Debug, Default)]
struct JoinReport {
    proof_path: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    build_seconds: f64,
    compile_seconds: f64,
    execute_seconds: f64,
    setup_seconds: Option<f64>,
    prove_seconds: Option<f64>,
    verify_seconds: Option<f64>,
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
    let mut leaf_dir = std::env::var_os("SP1_ZKGPT_LEAF_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-leaf-output"));
    let mut output_dir = None;
    let mut arguments = std::env::args_os().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--estimate") => mode = Mode::Estimate,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--small") => shape = Shape::small(),
            Some("--layer") => layer = parse_usize(arguments.next(), "--layer"),
            Some("--leaf-dir") => {
                leaf_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--leaf-dir requires a value")),
                );
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
    let output_dir = output_dir.unwrap_or_else(|| leaf_dir.clone());
    Arguments { mode, shape, layer, leaf_dir, output_dir }
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

fn required_field<'a>(fields: &'a BTreeMap<String, String>, name: &str, path: &Path) -> &'a str {
    fields.get(name).unwrap_or_else(|| panic!("{} is missing {name}", path.display())).as_str()
}

fn parse_leaf_manifest(path: &Path) -> LeafManifest {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let fields = contents
        .lines()
        .map(|line| {
            line.split_once('=')
                .unwrap_or_else(|| panic!("invalid manifest line in {}: {line}", path.display()))
        })
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(required_field(&fields, "version", path), LEAF_VERSION.to_string());
    assert_eq!(required_field(&fields, "stage", path), "qkv_attention");
    let parse_index = |name: &str| {
        required_field(&fields, name, path)
            .parse()
            .unwrap_or_else(|error| panic!("invalid {name} in {}: {error}", path.display()))
    };
    LeafManifest {
        layer: parse_index("layer"),
        head: parse_index("head"),
        input: parse_digest(required_field(&fields, "input", path)),
        parameters: parse_digest(required_field(&fields, "parameters", path)),
        hints: parse_digest(required_field(&fields, "hints", path)),
        output: parse_digest(required_field(&fields, "output", path)),
        transcript: parse_digest(required_field(&fields, "transcript", path)),
    }
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

fn leaf_transcript_fields_host(
    shape: Shape,
    layer: usize,
    head: usize,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
) -> Vec<SP1Field> {
    let mut fields = [
        LEAF_VERSION,
        LEAF_STAGE_QKV_ATTENTION,
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
    fields.extend(input);
    fields.extend(parameters);
    fields.extend(hints);
    fields.extend(output);
    fields
}

fn compute_leaf_transcript(
    shape: Shape,
    layer: usize,
    head: usize,
    input: Digest,
    parameters: Digest,
    hints: Digest,
    output: Digest,
) -> Digest {
    host_commit_fields(
        DOMAIN_TRANSCRIPT,
        &leaf_transcript_fields_host(shape, layer, head, input, parameters, hints, output),
    )
}

fn split_attention_output(shape: Shape, combined: &[u16]) -> Vec<Vec<u16>> {
    assert_eq!(combined.len(), shape.sequence_length * shape.hidden_size);
    let head_dimension = shape.head_dimension();
    let mut heads =
        vec![Vec::with_capacity(shape.sequence_length * head_dimension); shape.num_heads];
    for token_row in combined.chunks_exact(shape.hidden_size) {
        for (head, output) in heads.iter_mut().enumerate() {
            let start = head * head_dimension;
            output.extend_from_slice(&token_row[start..start + head_dimension]);
        }
    }
    heads
}

fn concatenate_attention_output(shape: Shape, heads: &[JoinHeadData]) -> Vec<u16> {
    assert_eq!(heads.len(), shape.num_heads);
    let head_dimension = shape.head_dimension();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for head in heads {
            assert_eq!(head.output.len(), shape.sequence_length * head_dimension);
            let start = token * head_dimension;
            output.extend_from_slice(&head.output[start..start + head_dimension]);
        }
    }
    output
}

fn validate_join_data(shape: Shape, layer: usize, data: &JoinData) -> GroupCommitments {
    assert_eq!(data.heads.len(), shape.num_heads, "join requires every attention head");
    let mut parameter_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    let mut hint_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    for (expected_head, head) in data.heads.iter().enumerate() {
        assert_eq!(head.head, expected_head, "join heads must be ordered from zero");
        let output = host_commit_u16(DOMAIN_OUTPUT, &head.output);
        let transcript = compute_leaf_transcript(
            shape,
            layer,
            expected_head,
            data.input,
            head.parameters,
            head.hints,
            output,
        );
        assert_eq!(transcript, head.transcript, "head {expected_head} transcript mismatch");
        let head_field = SP1Field::from_canonical_usize(expected_head);
        parameter_fields.push(head_field);
        parameter_fields.extend(head.parameters);
        hint_fields.push(head_field);
        hint_fields.extend(head.hints);
    }

    let parameters = host_commit_fields(DOMAIN_ATTENTION_GROUP_PARAMETERS, &parameter_fields);
    let hints = host_commit_fields(DOMAIN_ATTENTION_GROUP_HINTS, &hint_fields);
    let concatenated = concatenate_attention_output(shape, &data.heads);
    let output = host_commit_u16(DOMAIN_ATTENTION_GROUP_OUTPUT, &concatenated);
    let mut transcript_fields = [
        LEAF_VERSION,
        ATTENTION_GROUP_STAGE,
        layer as u32,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.head_dimension() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    transcript_fields.extend(data.input);
    transcript_fields.extend(parameters);
    transcript_fields.extend(hints);
    transcript_fields.extend(output);
    for head in &data.heads {
        transcript_fields.push(SP1Field::from_canonical_usize(head.head));
        transcript_fields.extend(head.transcript);
    }
    let transcript = host_commit_fields(DOMAIN_ATTENTION_GROUP_TRANSCRIPT, &transcript_fields);
    GroupCommitments { input: data.input, parameters, hints, output, transcript }
}

fn load_join_data(arguments: &Arguments) -> JoinData {
    let output_path = arguments
        .leaf_dir
        .join(format!("zkgpt_attention_l{:02}.output.private.bf16.bin", arguments.layer));
    let combined = read_bf16_binary(&output_path);
    let head_outputs = split_attention_output(arguments.shape, &combined);

    let mut common_input = None;
    let mut heads = Vec::with_capacity(arguments.shape.num_heads);
    for (head, output) in head_outputs.into_iter().enumerate() {
        let manifest_path = arguments
            .leaf_dir
            .join(format!("zkgpt_leaf_l{:02}_h{head:02}.commitments.txt", arguments.layer));
        let manifest = parse_leaf_manifest(&manifest_path);
        assert_eq!(manifest.layer, arguments.layer, "leaf layer mismatch");
        assert_eq!(manifest.head, head, "leaf head mismatch");
        if let Some(expected) = common_input {
            assert_eq!(manifest.input, expected, "leaf input commitments differ");
        } else {
            common_input = Some(manifest.input);
        }
        let output_digest = host_commit_u16(DOMAIN_OUTPUT, &output);
        assert_eq!(output_digest, manifest.output, "head {head} output commitment mismatch");
        let transcript = compute_leaf_transcript(
            arguments.shape,
            arguments.layer,
            head,
            manifest.input,
            manifest.parameters,
            manifest.hints,
            output_digest,
        );
        assert_eq!(transcript, manifest.transcript, "head {head} manifest transcript mismatch");
        heads.push(JoinHeadData {
            head,
            parameters: manifest.parameters,
            hints: manifest.hints,
            transcript: manifest.transcript,
            output,
        });
    }
    let data = JoinData { input: common_input.expect("join requires an input"), heads };
    validate_join_data(arguments.shape, arguments.layer, &data);
    data
}

fn circuit_commit_fields(
    builder: &mut JoinBuilder,
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

fn build_join(builder: &mut JoinBuilder, shape: Shape) {
    let layer = builder.hint_felts_v2(1)[0];
    let input: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let mut parameter_digests = Vec::with_capacity(shape.num_heads);
    let mut hint_digests = Vec::with_capacity(shape.num_heads);
    let mut child_transcripts = Vec::with_capacity(shape.num_heads);
    let mut head_outputs = Vec::with_capacity(shape.num_heads);

    for head in 0..shape.num_heads {
        let parameters: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let hints: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let expected_transcript: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let output = builder.hint_felts_v2(shape.sequence_length * shape.head_dimension());
        let output_digest = circuit_commit_fields(builder, DOMAIN_OUTPUT, &output);

        let constants = [
            LEAF_VERSION,
            LEAF_STAGE_QKV_ATTENTION,
            head as u32,
            shape.sequence_length as u32,
            shape.hidden_size as u32,
            shape.num_heads as u32,
            shape.head_dimension() as u32,
            GPT2_LAYER_NORM_EPSILON as u32,
            GPT2_ATTENTION_SCALE as u32,
        ]
        .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
        let mut transcript_fields = vec![constants[0], constants[1], layer, constants[2]];
        transcript_fields.extend_from_slice(&constants[3..]);
        transcript_fields.extend(input);
        transcript_fields.extend(parameters);
        transcript_fields.extend(hints);
        transcript_fields.extend(output_digest);
        let computed_transcript =
            circuit_commit_fields(builder, DOMAIN_TRANSCRIPT, &transcript_fields);
        for (&computed, &expected) in computed_transcript.iter().zip(&expected_transcript) {
            builder.assert_felt_eq(computed, expected);
        }

        parameter_digests.push(parameters);
        hint_digests.push(hints);
        child_transcripts.push(expected_transcript);
        head_outputs.push(output);
    }

    let mut parameter_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    let mut hint_fields = Vec::with_capacity(shape.num_heads * (DIGEST_SIZE + 1));
    for head in 0..shape.num_heads {
        let head_constant = builder.constant(SP1Field::from_canonical_usize(head));
        parameter_fields.push(head_constant);
        parameter_fields.extend(parameter_digests[head]);
        hint_fields.push(head_constant);
        hint_fields.extend(hint_digests[head]);
    }
    let parameters =
        circuit_commit_fields(builder, DOMAIN_ATTENTION_GROUP_PARAMETERS, &parameter_fields);
    let hints = circuit_commit_fields(builder, DOMAIN_ATTENTION_GROUP_HINTS, &hint_fields);

    let head_dimension = shape.head_dimension();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for head_output in &head_outputs {
            let start = token * head_dimension;
            output.extend_from_slice(&head_output[start..start + head_dimension]);
        }
    }
    let output_digest = circuit_commit_fields(builder, DOMAIN_ATTENTION_GROUP_OUTPUT, &output);

    let constants = [
        LEAF_VERSION,
        ATTENTION_GROUP_STAGE,
        shape.sequence_length as u32,
        shape.hidden_size as u32,
        shape.num_heads as u32,
        shape.head_dimension() as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut group_fields = vec![constants[0], constants[1], layer];
    group_fields.extend_from_slice(&constants[2..]);
    group_fields.extend(input);
    group_fields.extend(parameters);
    group_fields.extend(hints);
    group_fields.extend(output_digest);
    for (head, transcript) in child_transcripts.into_iter().enumerate() {
        group_fields.push(builder.constant(SP1Field::from_canonical_usize(head)));
        group_fields.extend(transcript);
    }
    let group_transcript =
        circuit_commit_fields(builder, DOMAIN_ATTENTION_GROUP_TRANSCRIPT, &group_fields);

    let zero = builder.constant(SP1Field::zero());
    let mut public_value_elements = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
    let public_values: &mut RecursionPublicValues<Felt<SP1Field>> =
        public_value_elements.as_mut_slice().borrow_mut();
    public_values.digest = group_transcript;
    builder.commit_public_values_v2(*public_values);
}

fn witness_stream(
    shape: Shape,
    layer: usize,
    data: &JoinData,
) -> Vec<sp1_recursion_executor::Block<SP1Field>> {
    let values_per_head = 3 * DIGEST_SIZE + shape.sequence_length * shape.head_dimension();
    let mut values = Vec::with_capacity(1 + DIGEST_SIZE + shape.num_heads * values_per_head);
    values.push(SP1Field::from_canonical_usize(layer));
    values.extend(data.input);
    for (expected_head, head) in data.heads.iter().enumerate() {
        assert_eq!(head.head, expected_head);
        values.extend(head.parameters);
        values.extend(head.hints);
        values.extend(head.transcript);
        values.extend(head.output.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    }
    values.into_iter().map(Into::into).collect()
}

fn verify_child_proofs(arguments: &Arguments, data: &JoinData) {
    type A = RecursionAir<SP1Field, 3, 2>;
    let max_log_rows = arguments.shape.leaf_max_log_rows();
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
    let vk_path =
        arguments.leaf_dir.join(format!("zkgpt_leaf_l{:02}.shared.vk.bin", arguments.layer));
    let vk: StoredVerifyingKey = bincode::deserialize_from(
        File::open(&vk_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", vk_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", vk_path.display()));

    let started = Instant::now();
    for head in &data.heads {
        let proof_path = arguments
            .leaf_dir
            .join(format!("zkgpt_leaf_l{:02}_h{:02}.proof.bin", arguments.layer, head.head));
        let proof: StoredLeafProof =
            bincode::deserialize_from(File::open(&proof_path).unwrap_or_else(|error| {
                panic!("failed to open {}: {error}", proof_path.display())
            }))
            .unwrap_or_else(|error| panic!("failed to decode {}: {error}", proof_path.display()));
        assert_eq!(proof.shard_proofs.len(), 1, "leaf proof must contain one shard");
        let public_values: &RecursionPublicValues<SP1Field> =
            proof.shard_proofs[0].public_values.as_slice().borrow();
        assert_eq!(
            public_values.digest, head.transcript,
            "head {} proof public digest differs from its manifest",
            head.head
        );
        prover.verify(&vk, &proof).unwrap_or_else(|error| {
            panic!("head {} proof failed verification: {error}", head.head)
        });
        println!("child proof verified: head={}", head.head);
    }
    println!(
        "verified {} child proofs: elapsed={:.3}s",
        data.heads.len(),
        started.elapsed().as_secs_f64()
    );
}

fn write_join_manifest(
    arguments: &Arguments,
    data: &JoinData,
    commitments: GroupCommitments,
    report: &JoinReport,
) {
    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let optional_seconds =
        |value: Option<f64>| value.map_or_else(|| "null".to_owned(), |value| format!("{value:.6}"));
    let optional_path = |path: Option<&Path>| {
        path.map_or_else(
            || "null".to_owned(),
            |path| format!("\"{}\"", path.file_name().unwrap().to_string_lossy()),
        )
    };
    let optional_bytes = |path: Option<&Path>| {
        path.map_or_else(|| "null".to_owned(), |path| fs::metadata(path).unwrap().len().to_string())
    };
    let child_transcripts = data
        .heads
        .iter()
        .map(|head| format!("\"{}\"", digest_hex(&head.transcript)))
        .collect::<Vec<_>>()
        .join(", ");
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"version\": {leaf_version},\n",
            "  \"stage\": \"attention_join\",\n",
            "  \"layer\": {},\n",
            "  \"sequence_length\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"num_heads\": {},\n",
            "  \"input_commitment\": \"{}\",\n",
            "  \"parameters_commitment\": \"{}\",\n",
            "  \"hints_commitment\": \"{}\",\n",
            "  \"output_commitment\": \"{}\",\n",
            "  \"transcript_commitment\": \"{}\",\n",
            "  \"child_transcripts\": [{}],\n",
            "  \"build_seconds\": {build_seconds:.6},\n",
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
        arguments.shape.num_heads,
        digest_hex(&commitments.input),
        digest_hex(&commitments.parameters),
        digest_hex(&commitments.hints),
        digest_hex(&commitments.output),
        digest_hex(&commitments.transcript),
        child_transcripts,
        optional_seconds(report.setup_seconds),
        optional_seconds(report.prove_seconds),
        optional_seconds(report.verify_seconds),
        optional_path(report.proof_path.as_deref()),
        optional_bytes(report.proof_path.as_deref()),
        optional_path(report.vk_path.as_deref()),
        optional_bytes(report.vk_path.as_deref()),
        leaf_version = LEAF_VERSION,
        build_seconds = report.build_seconds,
        compile_seconds = report.compile_seconds,
        execute_seconds = report.execute_seconds,
    );
    let path = arguments
        .output_dir
        .join(format!("zkgpt_attention_join_l{:02}.manifest.json", arguments.layer));
    fs::write(&path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    println!("join manifest: {}", path.display());
}

async fn prove_join(
    program: Arc<RecursionProgram<SP1Field>>,
    record: sp1_recursion_executor::ExecutionRecord<SP1Field>,
    arguments: &Arguments,
) -> (PathBuf, PathBuf, f64, f64, f64) {
    type A = RecursionAir<SP1Field, 3, 2>;
    let max_rows = 1usize << JOIN_MAX_LOG_ROWS;
    let machine = A::verillm_machine();
    for chip in machine.chips() {
        if chip.included(&record) {
            let rows = chip.num_rows(&record).unwrap_or_default();
            println!("trace: {:<22} rows={rows}", chip.name());
            assert!(
                rows <= max_rows,
                "{} needs {rows} rows, exceeding join maximum {max_rows}",
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
        JOIN_MAX_LOG_ROWS as u32,
        JOIN_MAX_LOG_ROWS,
        machine,
    );
    let prover = simple_prover(verifier);
    let proof_shape = prover.shape_from_record(&record).expect("join machine has no proof shape");
    println!("proof shape: {proof_shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    println!("join setup: elapsed={setup_seconds:.3}s");
    let pk = unsafe { pk.into_inner() };

    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    let prove_seconds = prove_started.elapsed().as_secs_f64();
    println!("join proof generated: elapsed={prove_seconds:.3}s");

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated attention join proof must verify");
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    println!("join proof verified: elapsed={verify_seconds:.3}s");

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_attention_join_l{:02}", arguments.layer);
    let proof_path = arguments.output_dir.join(format!("{stem}.proof.bin"));
    let vk_path = arguments.output_dir.join(format!("{stem}.vk.bin"));
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    println!(
        "join proof artifacts: proof={} ({} bytes) vk={} ({} bytes)",
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
    println!(
        "mode={:?} layer={} seq_len={} hidden={} heads={} leaf_dir={}",
        arguments.mode,
        arguments.layer,
        arguments.shape.sequence_length,
        arguments.shape.hidden_size,
        arguments.shape.num_heads,
        arguments.leaf_dir.display()
    );
    println!(
        "join witness: child_outputs={} BF16 values; target_rows=2^{JOIN_MAX_LOG_ROWS}",
        arguments.shape.sequence_length * arguments.shape.hidden_size
    );
    if arguments.mode == Mode::Estimate {
        return;
    }

    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: JoinBuilder = AsmBuilder::default();
    build_join(&mut builder, arguments.shape);
    let block = builder.into_root_block();
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("built join: ir_ops={} elapsed={build_seconds:.3}s", block.ops.len());
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let load_started = Instant::now();
    let data = load_join_data(&arguments);
    let commitments = validate_join_data(arguments.shape, arguments.layer, &data);
    println!("loaded and checked join data: elapsed={:.3}s", load_started.elapsed().as_secs_f64());
    println!("attention output commitment: {}", digest_hex(&commitments.output));
    println!("attention group transcript: {}", digest_hex(&commitments.transcript));
    verify_child_proofs(&arguments, &data);

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    println!("compiled join: elapsed={compile_seconds:.3}s");

    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.witness_stream = witness_stream(arguments.shape, arguments.layer, &data).into();
    let execute_started = Instant::now();
    executor.run().expect("valid attention join witness must execute");
    let execute_seconds = execute_started.elapsed().as_secs_f64();
    println!("executed join: elapsed={execute_seconds:.3}s");
    assert_eq!(
        executor.record.public_values.digest, commitments.transcript,
        "join circuit transcript differs from host transcript"
    );
    println!("join circuit public digest matches all child transcripts");

    let mut report =
        JoinReport { build_seconds, compile_seconds, execute_seconds, ..JoinReport::default() };
    if arguments.mode == Mode::Prove {
        let record = std::mem::take(&mut executor.record);
        drop(executor);
        let result = prove_join(program, record, &arguments).await;
        report.proof_path = Some(result.0);
        report.vk_path = Some(result.1);
        report.setup_seconds = Some(result.2);
        report.prove_seconds = Some(result.3);
        report.verify_seconds = Some(result.4);
    }
    write_join_manifest(&arguments, &data, commitments, &report);
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(shape: Shape) -> JoinData {
        let input = host_commit_u16(0x2001, &[1, 2, 3, 4]);
        let heads = (0..shape.num_heads)
            .map(|head| {
                let parameters = host_commit_u16(0x2002, &[head as u16, 11]);
                let hints = host_commit_u16(0x2003, &[head as u16, 22]);
                let output = (0..shape.sequence_length * shape.head_dimension())
                    .map(|index| (head * 100 + index) as u16)
                    .collect::<Vec<_>>();
                let output_digest = host_commit_u16(DOMAIN_OUTPUT, &output);
                let transcript = compute_leaf_transcript(
                    shape,
                    DEFAULT_LAYER,
                    head,
                    input,
                    parameters,
                    hints,
                    output_digest,
                );
                JoinHeadData { head, parameters, hints, transcript, output }
            })
            .collect();
        JoinData { input, heads }
    }

    #[test]
    fn join_circuit_rejects_tampered_child_data() {
        let shape = Shape::small();
        let valid = synthetic_data(shape);
        let expected = validate_join_data(shape, DEFAULT_LAYER, &valid);

        let mut builder: JoinBuilder = AsmBuilder::default();
        build_join(&mut builder, shape);
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(
            compiler.compile_inner(builder.into_root_block()).validate().expect("valid join IR"),
        );
        let execute = |data: &JoinData| {
            let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
                program.clone(),
                inner_perm(),
            );
            executor.witness_stream = witness_stream(shape, DEFAULT_LAYER, data).into();
            let result = executor.run();
            (result.is_ok(), executor.record.public_values.digest)
        };

        let (succeeded, digest) = execute(&valid);
        assert!(succeeded);
        assert_eq!(digest, expected.transcript);

        let mut modified_output = valid.clone();
        modified_output.heads[0].output[0] ^= 1;
        assert!(!execute(&modified_output).0, "modified BF16 output must fail");

        let mut swapped_outputs = valid.clone();
        let head_zero = swapped_outputs.heads[0].output.clone();
        swapped_outputs.heads[0].output = swapped_outputs.heads[1].output.clone();
        swapped_outputs.heads[1].output = head_zero;
        assert!(!execute(&swapped_outputs).0, "swapped head outputs must fail");

        let mut modified_transcript = valid.clone();
        modified_transcript.heads[0].transcript[0] =
            modified_transcript.heads[0].transcript[0] + SP1Field::one();
        assert!(!execute(&modified_transcript).0, "modified child transcript must fail");

        let mut different_input = valid.clone();
        let alternate_input = host_commit_u16(0x2001, &[9, 8, 7, 6]);
        let head = &different_input.heads[1];
        let output_digest = host_commit_u16(DOMAIN_OUTPUT, &head.output);
        different_input.heads[1].transcript = compute_leaf_transcript(
            shape,
            DEFAULT_LAYER,
            1,
            alternate_input,
            head.parameters,
            head.hints,
            output_digest,
        );
        assert!(!execute(&different_input).0, "different child input commitment must fail");
    }
}
