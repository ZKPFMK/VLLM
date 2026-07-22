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
use sp1_hypercube::{
    air::MachineAir, inner_perm, prover::simple_prover, MachineProof, ShardVerifier,
};
use sp1_primitives::{
    fri_params::{unique_decoding_queries, SP1_PROOF_OF_WORK_BITS},
    SP1DiffusionMatrix, SP1ExtensionField, SP1Field,
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
    let mut shape = Shape::default();
    let mut allow_large_build = false;
    let mut output_dir = std::env::var_os("SP1_ZKGPT_LIKE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sp1-zkgpt-like-output"));
    let mut data_dir = std::env::var_os("SP1_ZKGPT_LIKE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    let mut synthetic = false;
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
    shape.validate();
    Arguments { mode, shape, allow_large_build, output_dir, data_dir, synthetic }
}

fn read_bf16_binary(path: &Path) -> Vec<u16> {
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert_eq!(bytes.len() % 2, 0, "{} has an odd byte length", path.display());
    bytes.chunks_exact(2).map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])).collect()
}

fn load_real_data(data_dir: &Path, shape: Shape) -> RealData {
    let load_started = Instant::now();
    let hidden_states = read_bf16_binary(&data_dir.join("hidden_state.bf16.bin"));
    assert_eq!(
        hidden_states.len(),
        shape.sequence_length * shape.hidden_size,
        "real BF16 hidden-state shape mismatch"
    );
    let mut layers = Vec::with_capacity(shape.layers);
    for layer in 0..shape.layers {
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
) {
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
    println!("proof generated: elapsed={:.3}s", prove_started.elapsed().as_secs_f64());

    let verify_started = Instant::now();
    prover.verify(&vk, &proof).expect("generated zkGPT-like proof must verify");
    println!("proof verified: elapsed={:.3}s", verify_started.elapsed().as_secs_f64());

    fs::create_dir_all(output_dir)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", output_dir.display()));
    let proof_path = output_dir.join("zkgpt_like.proof.bin");
    let vk_path = output_dir.join("zkgpt_like.vk.bin");
    bincode::serialize_into(File::create(&proof_path).unwrap(), &proof).unwrap();
    bincode::serialize_into(File::create(&vk_path).unwrap(), &vk).unwrap();
    println!(
        "proof artifacts: proof={} ({} bytes) vk={} ({} bytes)",
        proof_path.display(),
        fs::metadata(&proof_path).unwrap().len(),
        vk_path.display(),
        fs::metadata(&vk_path).unwrap().len(),
    );
    println!(
        "proof end-to-end (setup+prove+verify+serialize): elapsed={:.3}s",
        end_to_end_started.elapsed().as_secs_f64()
    );
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
    let real_data = if arguments.synthetic {
        println!(
            "fixture: synthetic hidden=zeros linear_weights=ones \
             layer_norm_weight=one layer_norm_bias=zero"
        );
        None
    } else {
        println!("fixture: real pretrained GPT-2 BF16 weights adapted to zkGPT layout");
        Some(load_real_data(&arguments.data_dir, shape))
    };
    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let mut hidden_states = match &real_data {
        Some(data) => constants(&mut builder, &data.hidden_states),
        None => repeated_constants(&mut builder, 0x0000, shape.sequence_length * shape.hidden_size),
    };
    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let attention_scale = builder.constant(SP1Field::from_canonical_u16(GPT2_ATTENTION_SCALE));
    let zero = builder.constant(SP1Field::zero());

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
    commit_bf16_output(&mut builder, &hidden_states);
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
    let execute_started = Instant::now();
    let mut executor = Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(
        program.clone(),
        inner_perm(),
    );
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
    println!(
        "public output digest: {}",
        executor
            .record
            .public_values
            .digest
            .iter()
            .map(|value| format!("{:08X}", value.as_canonical_u32()))
            .collect::<Vec<_>>()
            .join(":")
    );
    if arguments.mode == Mode::Prove {
        let record = std::mem::take(&mut executor.record);
        drop(executor);
        prove_and_verify(program, record, &arguments.output_dir).await;
    }
    println!("completed: elapsed={:.3}s", total_started.elapsed().as_secs_f64());
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let arguments = parse_arguments();
    let events = EventCounts::for_shape(arguments.shape);
    println!("mode={:?}", arguments.mode);
    print_estimate(arguments.shape, events);
    if arguments.mode == Mode::CheckData {
        assert!(!arguments.synthetic, "--check-data requires a real BF16 fixture");
        let data = load_real_data(&arguments.data_dir, arguments.shape);
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
