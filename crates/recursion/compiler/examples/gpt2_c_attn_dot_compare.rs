use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use slop_algebra::AbstractField;
use sp1_hypercube::inner_perm;
use sp1_primitives::{SP1DiffusionMatrix, SP1ExtensionField, SP1Field};
use sp1_recursion_compiler::{
    circuit::{AsmBuilder, AsmCompiler},
    prelude::{Builder, Felt},
};
use sp1_recursion_executor::Executor;

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

fn bf16_to_f64(raw: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(raw) << 16))
}

fn main() {
    let data_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/sp1-gpt2-ln1"));
    let input = read_bf16_hex(&data_dir.join("pytorch_output.bf16.hex"));
    let weight = read_bf16_hex(&data_dir.join("c_attn_weight_col0.bf16.hex"));
    let bias = read_bf16_hex(&data_dir.join("c_attn_bias0.bf16.hex"));
    let pytorch_bf16 = read_bf16_hex(&data_dir.join("c_attn_col0_pytorch_output.bf16.hex"));
    let pytorch_fp32 = fs::read_to_string(data_dir.join("c_attn_col0_pytorch_output.fp32.txt"))
        .unwrap()
        .trim()
        .parse::<f64>()
        .unwrap();
    assert_eq!(input.len(), 768);
    assert_eq!(weight.len(), input.len());
    assert_eq!(bias.len(), 1);
    assert_eq!(pytorch_bf16.len(), 1);

    let mut builder: Builder<sp1_recursion_compiler::circuit::AsmConfig> = AsmBuilder::default();
    let input = input
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let weight = weight
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let bias = bias
        .iter()
        .map(|&value| builder.constant(SP1Field::from_canonical_u16(value)))
        .collect::<Vec<Felt<_>>>();
    let output = builder.bf16_linear(&input, &weight, &bias);
    assert_eq!(output.len(), 1);
    builder.print_f(output[0]);

    let mut compiler = AsmCompiler::default();
    let program = Arc::new(compiler.compile_inner(builder.into_root_block()).validate().unwrap());
    let print_path = data_dir.join("c_attn_col0_circuit_output.txt");
    let mut executor =
        Executor::<SP1Field, SP1ExtensionField, SP1DiffusionMatrix>::new(program, inner_perm());
    executor.debug_stdout = Box::new(File::create(&print_path).unwrap());
    executor.run().unwrap();
    println!(
        "events: mul={} add_sub={}",
        executor.record.bf16_mul_events.len(),
        executor.record.bf16_add_sub_events.len(),
    );
    drop(executor);

    let circuit_raw = fs::read_to_string(&print_path)
        .unwrap()
        .trim()
        .strip_prefix("PRINTF=")
        .unwrap()
        .parse::<u16>()
        .unwrap();
    fs::write(data_dir.join("c_attn_col0_circuit_output.bf16.hex"), format!("{circuit_raw:04X}\n"))
        .unwrap();

    let circuit = bf16_to_f64(circuit_raw);
    let pytorch_bf16_value = bf16_to_f64(pytorch_bf16[0]);
    println!("circuit:       raw={circuit_raw:04X} value={circuit:.9}");
    println!("PyTorch BF16:  raw={:04X} value={pytorch_bf16_value:.9}", pytorch_bf16[0]);
    println!("PyTorch FP32:               value={pytorch_fp32:.9}");
    println!(
        "absolute error: BF16={:.9} FP32={:.9}",
        (circuit - pytorch_bf16_value).abs(),
        (circuit - pytorch_fp32).abs(),
    );
    println!("circuit output: {}", data_dir.join("c_attn_col0_circuit_output.bf16.hex").display());
}
