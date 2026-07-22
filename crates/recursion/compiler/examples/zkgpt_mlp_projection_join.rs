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
const DEFAULT_EXPANSION_SIZE: usize = 2304;
const DEFAULT_NUM_TILES: usize = 12;
const DEFAULT_LAYER: usize = 0;

// These constants must remain identical to `zkgpt_mlp_projection_leaf`.
const PROTOCOL_VERSION: u32 = 1;
const MLP_PROJECTION_TILE_STAGE: u32 = 8;
const MLP_PROJECTION_GROUP_STAGE: u32 = 9;
const DOMAIN_MLP_PROJECTION_TILE_OUTPUT: u32 = 0x1502;
const DOMAIN_MLP_PROJECTION_TILE_TRANSCRIPT: u32 = 0x1503;
const DOMAIN_MLP_PROJECTION_GROUP_PARAMETERS: u32 = 0x1511;
const DOMAIN_MLP_PROJECTION_GROUP_OUTPUT: u32 = 0x1512;
const DOMAIN_MLP_PROJECTION_GROUP_TRANSCRIPT: u32 = 0x1513;

const PROOF_LOG_BLOWUP: usize = 1;
const JOIN_MAX_LOG_ROWS: usize = 16;
const FULL_TILE_MAX_LOG_ROWS: usize = 19;
const SMALL_TILE_MAX_LOG_ROWS: usize = 16;

type JoinBuilder = Builder<AsmConfig>;
type Digest = [SP1Field; DIGEST_SIZE];
type StoredTileProof = MachineProof<SP1GlobalContext, SP1PcsProofInner>;
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
    expansion_size: usize,
    num_tiles: usize,
}

impl Shape {
    fn full() -> Self {
        Self {
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            expansion_size: DEFAULT_EXPANSION_SIZE,
            num_tiles: DEFAULT_NUM_TILES,
        }
    }

    fn small() -> Self {
        Self { sequence_length: 2, hidden_size: 8, expansion_size: 24, num_tiles: 2 }
    }

    fn validate(self) {
        assert!(self.sequence_length > 0, "MLP projection join sequence length must be nonzero");
        assert!(self.hidden_size > 0, "MLP projection join output size must be nonzero");
        assert!(self.expansion_size > 0, "MLP projection join input size must be nonzero");
        assert!(self.num_tiles > 0, "MLP projection join tile count must be nonzero");
        assert_eq!(
            self.expansion_size,
            3 * self.hidden_size,
            "zkGPT MLP projection input width must be three times hidden size"
        );
        assert_eq!(
            self.hidden_size % self.num_tiles,
            0,
            "MLP projection output width must be divisible by tile count"
        );
    }

    fn tile_width(self) -> usize {
        self.hidden_size / self.num_tiles
    }

    fn tile_max_log_rows(self) -> usize {
        if self.sequence_length == DEFAULT_SEQUENCE_LENGTH
            && self.hidden_size == DEFAULT_HIDDEN_SIZE
            && self.expansion_size == DEFAULT_EXPANSION_SIZE
            && self.num_tiles == DEFAULT_NUM_TILES
        {
            FULL_TILE_MAX_LOG_ROWS
        } else {
            SMALL_TILE_MAX_LOG_ROWS
        }
    }
}

#[derive(Debug)]
struct Arguments {
    mode: Mode,
    shape: Shape,
    layer: usize,
    tile_dir: PathBuf,
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct TileManifest {
    layer: usize,
    tile: usize,
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
    transcript: Digest,
}

#[derive(Clone, Debug)]
struct JoinTileData {
    tile: usize,
    parameters: Digest,
    transcript: Digest,
    output: Vec<u16>,
}

#[derive(Clone, Debug)]
struct JoinData {
    upstream: Digest,
    input: Digest,
    tiles: Vec<JoinTileData>,
}

#[derive(Clone, Copy, Debug)]
struct GroupCommitments {
    upstream: Digest,
    input: Digest,
    parameters: Digest,
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
    child_verify_seconds: f64,
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
    let mut tile_dir = std::env::var_os("SP1_ZKGPT_MLP_PROJECTION_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-mlp-projection-output"));
    let mut output_dir = std::env::var_os("SP1_ZKGPT_MLP_PROJECTION_JOIN_DIR").map(PathBuf::from);
    let mut arguments = std::env::args_os().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--estimate") => mode = Mode::Estimate,
            Some("--build") => mode = Mode::Build,
            Some("--execute") => mode = Mode::Execute,
            Some("--prove") => mode = Mode::Prove,
            Some("--small") => shape = Shape::small(),
            Some("--layer") => layer = parse_usize(arguments.next(), "--layer"),
            Some("--tile-dir") => {
                tile_dir = PathBuf::from(
                    arguments.next().unwrap_or_else(|| panic!("--tile-dir requires a value")),
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
    let output_dir = output_dir.unwrap_or_else(|| tile_dir.clone());
    Arguments { mode, shape, layer, tile_dir, output_dir }
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

fn parse_tile_manifest(path: &Path) -> TileManifest {
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
    assert_eq!(required_field(&fields, "version", path), PROTOCOL_VERSION.to_string());
    assert_eq!(required_field(&fields, "stage", path), "mlp_projection_tile");
    let parse_index = |name: &str| {
        required_field(&fields, name, path)
            .parse()
            .unwrap_or_else(|error| panic!("invalid {name} in {}: {error}", path.display()))
    };
    TileManifest {
        layer: parse_index("layer"),
        tile: parse_index("tile"),
        upstream: parse_digest(required_field(&fields, "upstream", path)),
        input: parse_digest(required_field(&fields, "input", path)),
        parameters: parse_digest(required_field(&fields, "parameters", path)),
        output: parse_digest(required_field(&fields, "output", path)),
        transcript: parse_digest(required_field(&fields, "transcript", path)),
    }
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
        MLP_PROJECTION_TILE_STAGE,
        layer as u32,
        tile as u32,
        shape.sequence_length as u32,
        shape.expansion_size as u32,
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

fn compute_tile_transcript(
    shape: Shape,
    layer: usize,
    tile: usize,
    upstream: Digest,
    input: Digest,
    parameters: Digest,
    output: Digest,
) -> Digest {
    host_commit_fields(
        DOMAIN_MLP_PROJECTION_TILE_TRANSCRIPT,
        &tile_transcript_fields_host(shape, layer, tile, upstream, input, parameters, output),
    )
}

fn split_projection_output(shape: Shape, combined: &[u16]) -> Vec<Vec<u16>> {
    assert_eq!(combined.len(), shape.sequence_length * shape.hidden_size);
    let tile_width = shape.tile_width();
    let mut tiles = vec![Vec::with_capacity(shape.sequence_length * tile_width); shape.num_tiles];
    for token_row in combined.chunks_exact(shape.hidden_size) {
        for (tile, output) in tiles.iter_mut().enumerate() {
            let start = tile * tile_width;
            output.extend_from_slice(&token_row[start..start + tile_width]);
        }
    }
    tiles
}

fn concatenate_projection_output(shape: Shape, tiles: &[JoinTileData]) -> Vec<u16> {
    assert_eq!(tiles.len(), shape.num_tiles);
    let tile_width = shape.tile_width();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for tile in tiles {
            assert_eq!(tile.output.len(), shape.sequence_length * tile_width);
            let start = token * tile_width;
            output.extend_from_slice(&tile.output[start..start + tile_width]);
        }
    }
    output
}

fn validate_join_data(shape: Shape, layer: usize, data: &JoinData) -> GroupCommitments {
    assert_eq!(data.tiles.len(), shape.num_tiles, "MLP projection join requires every tile");
    let mut parameter_fields = Vec::with_capacity(shape.num_tiles * (DIGEST_SIZE + 1));
    for (expected_tile, tile) in data.tiles.iter().enumerate() {
        assert_eq!(tile.tile, expected_tile, "MLP projection tiles must be ordered from zero");
        let output = host_commit_u16(DOMAIN_MLP_PROJECTION_TILE_OUTPUT, &tile.output);
        let transcript = compute_tile_transcript(
            shape,
            layer,
            expected_tile,
            data.upstream,
            data.input,
            tile.parameters,
            output,
        );
        assert_eq!(transcript, tile.transcript, "tile {expected_tile} transcript mismatch");
        parameter_fields.push(SP1Field::from_canonical_usize(expected_tile));
        parameter_fields.extend(tile.parameters);
    }

    let parameters = host_commit_fields(DOMAIN_MLP_PROJECTION_GROUP_PARAMETERS, &parameter_fields);
    let concatenated = concatenate_projection_output(shape, &data.tiles);
    let output = host_commit_u16(DOMAIN_MLP_PROJECTION_GROUP_OUTPUT, &concatenated);
    let mut transcript_fields = [
        PROTOCOL_VERSION,
        MLP_PROJECTION_GROUP_STAGE,
        layer as u32,
        shape.sequence_length as u32,
        shape.expansion_size as u32,
        shape.hidden_size as u32,
        shape.num_tiles as u32,
        shape.tile_width() as u32,
    ]
    .map(SP1Field::from_canonical_u32)
    .to_vec();
    transcript_fields.extend(data.upstream);
    transcript_fields.extend(data.input);
    transcript_fields.extend(parameters);
    transcript_fields.extend(output);
    for tile in &data.tiles {
        transcript_fields.push(SP1Field::from_canonical_usize(tile.tile));
        transcript_fields.extend(tile.transcript);
    }
    let transcript = host_commit_fields(DOMAIN_MLP_PROJECTION_GROUP_TRANSCRIPT, &transcript_fields);
    GroupCommitments { upstream: data.upstream, input: data.input, parameters, output, transcript }
}

fn load_join_data(arguments: &Arguments) -> JoinData {
    let group_manifest_path = arguments
        .tile_dir
        .join(format!("zkgpt_mlp_projection_l{:02}.manifest.json", arguments.layer));
    let expected_upstream =
        parse_digest(&json_string_field(&group_manifest_path, "upstream_transcript"));
    let expected_input = parse_digest(&json_string_field(&group_manifest_path, "input_commitment"));
    let expected_parameters =
        parse_digest(&json_string_field(&group_manifest_path, "parameters_commitment"));
    let expected_output =
        parse_digest(&json_string_field(&group_manifest_path, "output_commitment"));
    let expected_transcript =
        parse_digest(&json_string_field(&group_manifest_path, "transcript_commitment"));

    let output_path = arguments
        .tile_dir
        .join(format!("zkgpt_mlp_projection_l{:02}.output.private.bf16.bin", arguments.layer));
    let combined = read_bf16_binary(&output_path);
    let tile_outputs = split_projection_output(arguments.shape, &combined);

    let mut tiles = Vec::with_capacity(arguments.shape.num_tiles);
    for (tile, output) in tile_outputs.into_iter().enumerate() {
        let manifest_path = arguments.tile_dir.join(format!(
            "zkgpt_mlp_projection_l{:02}_t{tile:02}.commitments.txt",
            arguments.layer
        ));
        let manifest = parse_tile_manifest(&manifest_path);
        assert_eq!(manifest.layer, arguments.layer, "MLP projection tile layer mismatch");
        assert_eq!(manifest.tile, tile, "MLP projection tile index mismatch");
        assert_eq!(manifest.upstream, expected_upstream, "upstream transcripts differ");
        assert_eq!(manifest.input, expected_input, "tile input commitments differ");
        let output_digest = host_commit_u16(DOMAIN_MLP_PROJECTION_TILE_OUTPUT, &output);
        assert_eq!(output_digest, manifest.output, "tile {tile} output commitment mismatch");
        let transcript = compute_tile_transcript(
            arguments.shape,
            arguments.layer,
            tile,
            manifest.upstream,
            manifest.input,
            manifest.parameters,
            output_digest,
        );
        assert_eq!(transcript, manifest.transcript, "tile {tile} manifest transcript mismatch");
        tiles.push(JoinTileData {
            tile,
            parameters: manifest.parameters,
            transcript: manifest.transcript,
            output,
        });
    }

    let data = JoinData { upstream: expected_upstream, input: expected_input, tiles };
    let commitments = validate_join_data(arguments.shape, arguments.layer, &data);
    assert_eq!(commitments.parameters, expected_parameters, "group parameter commitment mismatch");
    assert_eq!(commitments.output, expected_output, "group output commitment mismatch");
    assert_eq!(commitments.transcript, expected_transcript, "group transcript mismatch");
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
    let upstream: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let input: [Felt<SP1Field>; DIGEST_SIZE] =
        builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
    let mut parameter_digests = Vec::with_capacity(shape.num_tiles);
    let mut child_transcripts = Vec::with_capacity(shape.num_tiles);
    let mut tile_outputs = Vec::with_capacity(shape.num_tiles);

    for tile in 0..shape.num_tiles {
        let parameters: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let expected_transcript: [Felt<SP1Field>; DIGEST_SIZE] =
            builder.hint_felts_v2(DIGEST_SIZE).try_into().unwrap();
        let output = builder.hint_felts_v2(shape.sequence_length * shape.tile_width());
        let output_digest =
            circuit_commit_fields(builder, DOMAIN_MLP_PROJECTION_TILE_OUTPUT, &output);

        let constants = [
            PROTOCOL_VERSION,
            MLP_PROJECTION_TILE_STAGE,
            tile as u32,
            shape.sequence_length as u32,
            shape.expansion_size as u32,
            shape.hidden_size as u32,
            shape.num_tiles as u32,
            shape.tile_width() as u32,
        ]
        .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
        let mut transcript_fields = vec![constants[0], constants[1], layer, constants[2]];
        transcript_fields.extend_from_slice(&constants[3..]);
        transcript_fields.extend(upstream);
        transcript_fields.extend(input);
        transcript_fields.extend(parameters);
        transcript_fields.extend(output_digest);
        let computed_transcript = circuit_commit_fields(
            builder,
            DOMAIN_MLP_PROJECTION_TILE_TRANSCRIPT,
            &transcript_fields,
        );
        for (&computed, &expected) in computed_transcript.iter().zip(&expected_transcript) {
            builder.assert_felt_eq(computed, expected);
        }

        parameter_digests.push(parameters);
        child_transcripts.push(expected_transcript);
        tile_outputs.push(output);
    }

    let mut parameter_fields = Vec::with_capacity(shape.num_tiles * (DIGEST_SIZE + 1));
    for (tile, parameters) in parameter_digests.into_iter().enumerate() {
        parameter_fields.push(builder.constant(SP1Field::from_canonical_usize(tile)));
        parameter_fields.extend(parameters);
    }
    let parameters =
        circuit_commit_fields(builder, DOMAIN_MLP_PROJECTION_GROUP_PARAMETERS, &parameter_fields);

    let tile_width = shape.tile_width();
    let mut output = Vec::with_capacity(shape.sequence_length * shape.hidden_size);
    for token in 0..shape.sequence_length {
        for tile_output in &tile_outputs {
            let start = token * tile_width;
            output.extend_from_slice(&tile_output[start..start + tile_width]);
        }
    }
    let output_digest = circuit_commit_fields(builder, DOMAIN_MLP_PROJECTION_GROUP_OUTPUT, &output);

    let constants = [
        PROTOCOL_VERSION,
        MLP_PROJECTION_GROUP_STAGE,
        shape.sequence_length as u32,
        shape.expansion_size as u32,
        shape.hidden_size as u32,
        shape.num_tiles as u32,
        shape.tile_width() as u32,
    ]
    .map(|value| builder.constant(SP1Field::from_canonical_u32(value)));
    let mut group_fields = vec![constants[0], constants[1], layer];
    group_fields.extend_from_slice(&constants[2..]);
    group_fields.extend(upstream);
    group_fields.extend(input);
    group_fields.extend(parameters);
    group_fields.extend(output_digest);
    for (tile, transcript) in child_transcripts.into_iter().enumerate() {
        group_fields.push(builder.constant(SP1Field::from_canonical_usize(tile)));
        group_fields.extend(transcript);
    }
    let group_transcript =
        circuit_commit_fields(builder, DOMAIN_MLP_PROJECTION_GROUP_TRANSCRIPT, &group_fields);

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
    let values_per_tile = 2 * DIGEST_SIZE + shape.sequence_length * shape.tile_width();
    let mut values = Vec::with_capacity(1 + 2 * DIGEST_SIZE + shape.num_tiles * values_per_tile);
    values.push(SP1Field::from_canonical_usize(layer));
    values.extend(data.upstream);
    values.extend(data.input);
    for (expected_tile, tile) in data.tiles.iter().enumerate() {
        assert_eq!(tile.tile, expected_tile);
        values.extend(tile.parameters);
        values.extend(tile.transcript);
        values.extend(tile.output.iter().map(|&value| SP1Field::from_canonical_u16(value)));
    }
    values.into_iter().map(Into::into).collect()
}

fn verify_child_proofs(arguments: &Arguments, data: &JoinData) -> f64 {
    type A = RecursionAir<SP1Field, 3, 2>;
    let max_log_rows = arguments.shape.tile_max_log_rows();
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
    let vk_path = arguments
        .tile_dir
        .join(format!("zkgpt_mlp_projection_l{:02}.shared.vk.bin", arguments.layer));
    let vk: StoredVerifyingKey = bincode::deserialize_from(
        File::open(&vk_path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", vk_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to decode {}: {error}", vk_path.display()));

    let started = Instant::now();
    for tile in &data.tiles {
        let proof_path = arguments.tile_dir.join(format!(
            "zkgpt_mlp_projection_l{:02}_t{:02}.proof.bin",
            arguments.layer, tile.tile
        ));
        let proof: StoredTileProof =
            bincode::deserialize_from(File::open(&proof_path).unwrap_or_else(|error| {
                panic!("failed to open {}: {error}", proof_path.display())
            }))
            .unwrap_or_else(|error| panic!("failed to decode {}: {error}", proof_path.display()));
        assert_eq!(proof.shard_proofs.len(), 1, "MLP projection proof must contain one shard");
        let public_values: &RecursionPublicValues<SP1Field> =
            proof.shard_proofs[0].public_values.as_slice().borrow();
        assert_eq!(
            public_values.digest, tile.transcript,
            "tile {} proof public digest differs from its manifest",
            tile.tile
        );
        prover.verify(&vk, &proof).unwrap_or_else(|error| {
            panic!("tile {} proof failed verification: {error}", tile.tile)
        });
        println!("child proof verified: tile={}", tile.tile);
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!("verified {} child proofs: elapsed={elapsed:.3}s", data.tiles.len());
    elapsed
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
        .tiles
        .iter()
        .map(|tile| format!("\"{}\"", digest_hex(&tile.transcript)))
        .collect::<Vec<_>>()
        .join(", ");
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"version\": {protocol_version},\n",
            "  \"stage\": \"mlp_projection_block_join\",\n",
            "  \"layer\": {},\n",
            "  \"sequence_length\": {},\n",
            "  \"expansion_size\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"num_tiles\": {},\n",
            "  \"upstream_transcript\": \"{}\",\n",
            "  \"input_commitment\": \"{}\",\n",
            "  \"parameters_commitment\": \"{}\",\n",
            "  \"output_commitment\": \"{}\",\n",
            "  \"transcript_commitment\": \"{}\",\n",
            "  \"child_transcripts\": [{}],\n",
            "  \"private_output_file\": \"zkgpt_mlp_projection_l{:02}.output.private.bf16.bin\",\n",
            "  \"build_seconds\": {build_seconds:.6},\n",
            "  \"compile_seconds\": {compile_seconds:.6},\n",
            "  \"child_verify_seconds\": {child_verify_seconds:.6},\n",
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
        arguments.shape.expansion_size,
        arguments.shape.hidden_size,
        arguments.shape.num_tiles,
        digest_hex(&commitments.upstream),
        digest_hex(&commitments.input),
        digest_hex(&commitments.parameters),
        digest_hex(&commitments.output),
        digest_hex(&commitments.transcript),
        child_transcripts,
        arguments.layer,
        optional_seconds(report.setup_seconds),
        optional_seconds(report.prove_seconds),
        optional_seconds(report.verify_seconds),
        optional_path(report.proof_path.as_deref()),
        optional_bytes(report.proof_path.as_deref()),
        optional_path(report.vk_path.as_deref()),
        optional_bytes(report.vk_path.as_deref()),
        protocol_version = PROTOCOL_VERSION,
        build_seconds = report.build_seconds,
        compile_seconds = report.compile_seconds,
        child_verify_seconds = report.child_verify_seconds,
        execute_seconds = report.execute_seconds,
    );
    let path = arguments
        .output_dir
        .join(format!("zkgpt_mlp_projection_join_l{:02}.manifest.json", arguments.layer));
    fs::write(&path, manifest)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    println!("block-output join manifest: {}", path.display());
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
                "{} needs {rows} rows, exceeding block-output join maximum {max_rows}",
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
    let proof_shape =
        prover.shape_from_record(&record).expect("block-output join has no proof shape");
    println!("proof shape: {proof_shape:?}");

    let setup_started = Instant::now();
    let (pk, vk) = prover.setup(program).await;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    println!("block-output join setup: elapsed={setup_seconds:.3}s");
    let pk = unsafe { pk.into_inner() };

    let prove_started = Instant::now();
    let shard_proof = prover.prove_shard(pk, record).await;
    let proof = MachineProof::from(vec![shard_proof]);
    let prove_seconds = prove_started.elapsed().as_secs_f64();
    println!("block-output join proof generated: elapsed={prove_seconds:.3}s");

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated block-output join proof must verify");
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    println!("block-output join proof verified: elapsed={verify_seconds:.3}s");

    fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", arguments.output_dir.display())
    });
    let stem = format!("zkgpt_mlp_projection_join_l{:02}", arguments.layer);
    let proof_path = arguments.output_dir.join(format!("{stem}.proof.bin"));
    let vk_path = arguments.output_dir.join(format!("{stem}.vk.bin"));
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    println!(
        "block-output join artifacts: proof={} ({} bytes) vk={} ({} bytes)",
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
        "mode={:?} layer={} seq_len={} expansion={} hidden={} tiles={} tile_dir={}",
        arguments.mode,
        arguments.layer,
        arguments.shape.sequence_length,
        arguments.shape.expansion_size,
        arguments.shape.hidden_size,
        arguments.shape.num_tiles,
        arguments.tile_dir.display()
    );
    println!(
        "block-output join witness: child_outputs={} BF16 values; target_rows=2^{JOIN_MAX_LOG_ROWS}",
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
    println!("built block-output join: ir_ops={} elapsed={build_seconds:.3}s", block.ops.len());
    if arguments.mode == Mode::Build {
        println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
        return;
    }

    let load_started = Instant::now();
    let data = load_join_data(&arguments);
    let commitments = validate_join_data(arguments.shape, arguments.layer, &data);
    println!(
        "loaded and checked block-output join data: elapsed={:.3}s",
        load_started.elapsed().as_secs_f64()
    );
    println!("block output commitment: {}", digest_hex(&commitments.output));
    println!("block output transcript: {}", digest_hex(&commitments.transcript));
    let child_verify_seconds = verify_child_proofs(&arguments, &data);

    let compile_started = Instant::now();
    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    println!("compiled block-output join: elapsed={compile_seconds:.3}s");

    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
    executor.witness_stream = witness_stream(arguments.shape, arguments.layer, &data).into();
    let execute_started = Instant::now();
    executor.run().expect("valid block-output join witness must execute");
    let execute_seconds = execute_started.elapsed().as_secs_f64();
    println!("executed block-output join: elapsed={execute_seconds:.3}s");
    assert_eq!(
        executor.record.public_values.digest, commitments.transcript,
        "block-output join circuit transcript differs from host transcript"
    );
    println!("block-output join public digest matches all child transcripts");

    let mut report = JoinReport {
        build_seconds,
        compile_seconds,
        execute_seconds,
        child_verify_seconds,
        ..JoinReport::default()
    };
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
        let upstream = host_commit_u16(0x5001, &[1, 2, 3, 4]);
        let input = host_commit_u16(0x5002, &[5, 6, 7, 8]);
        let tiles = (0..shape.num_tiles)
            .map(|tile| {
                let parameters = host_commit_u16(0x5003, &[tile as u16, 11]);
                let output = (0..shape.sequence_length * shape.tile_width())
                    .map(|index| (tile * 100 + index) as u16)
                    .collect::<Vec<_>>();
                let output_digest = host_commit_u16(DOMAIN_MLP_PROJECTION_TILE_OUTPUT, &output);
                let transcript = compute_tile_transcript(
                    shape,
                    DEFAULT_LAYER,
                    tile,
                    upstream,
                    input,
                    parameters,
                    output_digest,
                );
                JoinTileData { tile, parameters, transcript, output }
            })
            .collect();
        JoinData { upstream, input, tiles }
    }

    #[test]
    fn split_and_concatenate_preserve_token_major_order() {
        let shape = Shape::small();
        let combined = (0..shape.sequence_length * shape.hidden_size)
            .map(|value| value as u16)
            .collect::<Vec<_>>();
        let tiles = split_projection_output(shape, &combined)
            .into_iter()
            .enumerate()
            .map(|(tile, output)| JoinTileData {
                tile,
                parameters: [SP1Field::zero(); DIGEST_SIZE],
                transcript: [SP1Field::zero(); DIGEST_SIZE],
                output,
            })
            .collect::<Vec<_>>();
        assert_eq!(concatenate_projection_output(shape, &tiles), combined);
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
        modified_output.tiles[0].output[0] ^= 1;
        assert!(!execute(&modified_output).0, "modified BF16 output must fail");

        let mut swapped_outputs = valid.clone();
        let tile_zero = swapped_outputs.tiles[0].output.clone();
        swapped_outputs.tiles[0].output = swapped_outputs.tiles[1].output.clone();
        swapped_outputs.tiles[1].output = tile_zero;
        assert!(!execute(&swapped_outputs).0, "swapped tile outputs must fail");

        let mut modified_transcript = valid.clone();
        modified_transcript.tiles[0].transcript[0] =
            modified_transcript.tiles[0].transcript[0] + SP1Field::one();
        assert!(!execute(&modified_transcript).0, "modified child transcript must fail");

        let mut different_input = valid.clone();
        different_input.input = host_commit_u16(0x5002, &[9, 8, 7, 6]);
        assert!(!execute(&different_input).0, "different common input commitment must fail");
    }
}
