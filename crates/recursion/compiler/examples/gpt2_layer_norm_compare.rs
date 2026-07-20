use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use slop_algebra::{AbstractField, PrimeField32};
use sp1_hypercube::inner_perm;
use sp1_primitives::{SP1DiffusionMatrix, SP1ExtensionField, SP1Field};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler},
    prelude::{Builder, Felt},
};
use sp1_recursion_executor::Executor;

const GPT2_LAYER_NORM_EPSILON: u16 = 0x3727;

fn read_bf16_hex(path: &Path) -> Vec<u16> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(|line| {
            u16::from_str_radix(line, 16)
                .unwrap_or_else(|error| panic!("invalid BF16 value {line:?}: {error}"))
        })
        .collect()
}

fn read_fp32(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(|line| line.parse::<f64>().unwrap())
        .collect()
}

fn bf16_to_f64(raw: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(raw) << 16))
}

#[derive(Debug)]
struct ErrorMetrics {
    mean_absolute_error: f64,
    root_mean_squared_error: f64,
    max_absolute_error: f64,
    max_absolute_error_index: usize,
    mean_relative_error: f64,
    max_relative_error: f64,
    cosine_similarity: f64,
}

fn error_metrics(actual: &[f64], reference: &[f64]) -> ErrorMetrics {
    assert_eq!(actual.len(), reference.len());

    let mut absolute_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut max_absolute_error = 0.0;
    let mut max_absolute_error_index = 0;
    let mut relative_sum = 0.0;
    let mut max_relative_error = 0.0;
    let mut dot = 0.0;
    let mut actual_squared_sum = 0.0;
    let mut reference_squared_sum = 0.0;

    for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let absolute_error = (actual - reference).abs();
        absolute_sum += absolute_error;
        squared_sum += absolute_error * absolute_error;
        if absolute_error > max_absolute_error {
            max_absolute_error = absolute_error;
            max_absolute_error_index = index;
        }

        let relative_error = absolute_error / reference.abs().max(1e-6);
        relative_sum += relative_error;
        if relative_error > max_relative_error {
            max_relative_error = relative_error;
        }

        dot += actual * reference;
        actual_squared_sum += actual * actual;
        reference_squared_sum += reference * reference;
    }

    let count = actual.len() as f64;
    ErrorMetrics {
        mean_absolute_error: absolute_sum / count,
        root_mean_squared_error: (squared_sum / count).sqrt(),
        max_absolute_error,
        max_absolute_error_index,
        mean_relative_error: relative_sum / count,
        max_relative_error,
        cosine_similarity: dot / (actual_squared_sum * reference_squared_sum).sqrt(),
    }
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let data_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sp1-gpt2-ln1"));
    let hidden_path = repo_root.join("crates/recursion/data/gpt2/once/hidden_state.bf16.hex");

    let hidden = read_bf16_hex(&hidden_path);
    let weight = read_bf16_hex(&data_dir.join("weight.bf16.hex"));
    let bias = read_bf16_hex(&data_dir.join("bias.bf16.hex"));
    let pytorch_bf16 = read_bf16_hex(&data_dir.join("pytorch_output.bf16.hex"));
    let pytorch_fp32 = read_fp32(&data_dir.join("pytorch_output.fp32.txt"));
    assert_eq!(hidden.len(), 768);
    assert_eq!(weight.len(), hidden.len());
    assert_eq!(bias.len(), hidden.len());
    assert_eq!(pytorch_bf16.len(), hidden.len());
    assert_eq!(pytorch_fp32.len(), hidden.len());

    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let values = hidden
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let weights = weight
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let biases = bias
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let epsilon = builder.constant(SP1Field::from_canonical_u16(GPT2_LAYER_NORM_EPSILON));
    let output = builder.bf16_layer_norm(&values, &weights, &biases, epsilon);
    for &value in &output {
        builder.print_f(value);
    }

    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
    let print_path = data_dir.join("circuit_output.txt");
    let mut executor =
        Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
    executor.debug_stdout = Box::new(File::create(&print_path).unwrap());
    executor.run().unwrap();
    println!(
        "events: add_sub={} unary={} mul={} div={}",
        executor.record.bf16_add_sub_events.len(),
        executor.record.bf16_unary_events.len(),
        executor.record.bf16_mul_events.len(),
        executor.record.bf16_div_events.len(),
    );
    let circuit_mean_raw = executor.record.bf16_div_events[0].output.raw.as_canonical_u32() as u16;
    let circuit_variance_raw =
        executor.record.bf16_div_events[1].output.raw.as_canonical_u32() as u16;
    let circuit_variance_epsilon_raw =
        executor.record.bf16_add_sub_events[2302].output.raw.as_canonical_u32() as u16;
    let circuit_inverse_std_raw =
        executor.record.bf16_unary_events[768].output.as_canonical_u32() as u16;
    drop(executor);

    let circuit_raw = fs::read_to_string(&print_path)
        .unwrap()
        .lines()
        .map(|line| {
            line.strip_prefix("PRINTF=")
                .unwrap_or_else(|| panic!("unexpected circuit output line: {line}"))
                .parse::<u16>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(circuit_raw.len(), hidden.len());
    fs::write(
        data_dir.join("circuit_output.bf16.hex"),
        circuit_raw.iter().map(|value| format!("{value:04X}\n")).collect::<String>(),
    )
    .unwrap();

    let circuit = circuit_raw.iter().copied().map(bf16_to_f64).collect::<Vec<_>>();
    let pytorch_bf16_values = pytorch_bf16.iter().copied().map(bf16_to_f64).collect::<Vec<_>>();
    let bf16_metrics = error_metrics(&circuit, &pytorch_bf16_values);
    let fp32_metrics = error_metrics(&circuit, &pytorch_fp32);
    let exact_matches = circuit_raw.iter().zip(&pytorch_bf16).filter(|(a, b)| a == b).count();

    let hidden_values = hidden.iter().copied().map(bf16_to_f64).collect::<Vec<_>>();
    let pytorch_mean = hidden_values.iter().sum::<f64>() / hidden_values.len() as f64;
    let pytorch_variance =
        hidden_values.iter().map(|value| (value - pytorch_mean).powi(2)).sum::<f64>()
            / hidden_values.len() as f64;
    let pytorch_inverse_std = 1.0 / (pytorch_variance + 1e-5).sqrt();

    println!("exact BF16 matches: {exact_matches}/{}", hidden.len());
    println!(
        "circuit stats: mean={:.9} ({circuit_mean_raw:04X}) variance={:.9} \
         ({circuit_variance_raw:04X}) variance+epsilon={:.9} \
         ({circuit_variance_epsilon_raw:04X}) inverse_std={:.9} \
         ({circuit_inverse_std_raw:04X})",
        bf16_to_f64(circuit_mean_raw),
        bf16_to_f64(circuit_variance_raw),
        bf16_to_f64(circuit_variance_epsilon_raw),
        bf16_to_f64(circuit_inverse_std_raw),
    );
    println!(
        "PyTorch-style FP32 stats: mean={pytorch_mean:.9} variance={pytorch_variance:.9} \
         variance+epsilon={:.9} inverse_std={pytorch_inverse_std:.9}",
        pytorch_variance + 1e-5,
    );
    println!(
        "vs PyTorch BF16: MAE={:.9} RMSE={:.9} max_abs={:.9}@{} \
         mean_rel={:.6} max_rel={:.6} cosine={:.9}",
        bf16_metrics.mean_absolute_error,
        bf16_metrics.root_mean_squared_error,
        bf16_metrics.max_absolute_error,
        bf16_metrics.max_absolute_error_index,
        bf16_metrics.mean_relative_error,
        bf16_metrics.max_relative_error,
        bf16_metrics.cosine_similarity,
    );
    println!(
        "vs PyTorch FP32: MAE={:.9} RMSE={:.9} max_abs={:.9}@{} \
         mean_rel={:.6} max_rel={:.6} cosine={:.9}",
        fp32_metrics.mean_absolute_error,
        fp32_metrics.root_mean_squared_error,
        fp32_metrics.max_absolute_error,
        fp32_metrics.max_absolute_error_index,
        fp32_metrics.mean_relative_error,
        fp32_metrics.max_relative_error,
        fp32_metrics.cosine_similarity,
    );

    let max_index = fp32_metrics.max_absolute_error_index;
    println!(
        "largest FP32-reference error: index={max_index} circuit_raw={:04X} \
         pytorch_bf16_raw={:04X} circuit={:.9} pytorch_bf16={:.9} pytorch_fp32={:.9}",
        circuit_raw[max_index],
        pytorch_bf16[max_index],
        circuit[max_index],
        pytorch_bf16_values[max_index],
        pytorch_fp32[max_index],
    );
    println!("circuit output: {}", data_dir.join("circuit_output.bf16.hex").display());
}
