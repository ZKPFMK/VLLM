#!/usr/bin/env python3
"""Export GPT-2 block 0 as compact raw-BF16 files for the recursion example."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from gpt2_bf16 import DEFAULT_MODEL_DIR


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HIDDEN_STATE = (
    REPO_ROOT
    / "crates"
    / "recursion"
    / "data"
    / "gpt2"
    / "once"
    / "hidden_state.bf16.hex"
)

PARAMETERS = {
    "ln_1_weight": "h.0.ln_1.weight",
    "ln_1_bias": "h.0.ln_1.bias",
    "attention_qkv_weight": "h.0.attn.c_attn.weight",
    "attention_qkv_bias": "h.0.attn.c_attn.bias",
    "attention_projection_weight": "h.0.attn.c_proj.weight",
    "attention_projection_bias": "h.0.attn.c_proj.bias",
    "ln_2_weight": "h.0.ln_2.weight",
    "ln_2_bias": "h.0.ln_2.bias",
    "mlp_expansion_weight": "h.0.mlp.c_fc.weight",
    "mlp_expansion_bias": "h.0.mlp.c_fc.bias",
    "mlp_projection_weight": "h.0.mlp.c_proj.weight",
    "mlp_projection_bias": "h.0.mlp.c_proj.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export real GPT-2 block-0 BF16 tensors without copying the checkpoint "
            "into the repository."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hidden-state", type=Path, default=DEFAULT_HIDDEN_STATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "default: MODEL_DIR/recursion/block0-once; use /tmp while testing the exporter"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_bf16_hex(path: Path) -> torch.Tensor:
    raw = [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()]
    return torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16)


def raw_bf16_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor, found {tensor.dtype}")
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
    return raw.astype("<u2", copy=False).tobytes(order="C")


def write_bf16_binary(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    data = raw_bf16_bytes(tensor)
    path.write_bytes(data)
    return {
        "file": path.name,
        "shape": list(tensor.shape),
        "values": tensor.numel(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_bf16_hex(path: Path, tensor: torch.Tensor) -> None:
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).reshape(-1).tolist()
    path.write_text("".join(f"{value:04X}\n" for value in raw), encoding="ascii")


def gelu_new(values: torch.Tensor) -> torch.Tensor:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * values * (1.0 + torch.tanh(coefficient * (values + 0.044715 * values**3)))


def pytorch_block0_reference(
    hidden: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    num_heads: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = hidden.numel()
    if hidden_size % num_heads != 0:
        raise RuntimeError(f"hidden size {hidden_size} is not divisible by {num_heads} heads")
    head_dimension = hidden_size // num_heads

    normalized = torch.nn.functional.layer_norm(
        hidden,
        (hidden_size,),
        parameters["ln_1_weight"],
        parameters["ln_1_bias"],
        eps=epsilon,
    )
    qkv = torch.addmm(
        parameters["attention_qkv_bias"],
        normalized.reshape(1, hidden_size),
        parameters["attention_qkv_weight"],
    ).reshape(3, num_heads, head_dimension)
    query, key, value = qkv.unbind(dim=0)
    scale = torch.tensor(head_dimension**-0.5, dtype=torch.bfloat16)
    # The exported fixture contains one token, so every head has one attention score.
    attention_max_hints = torch.sum(query * key, dim=-1) * scale
    merged_values = value.reshape(hidden_size)
    projected_attention = torch.addmm(
        parameters["attention_projection_bias"],
        merged_values.reshape(1, hidden_size),
        parameters["attention_projection_weight"],
    ).reshape(hidden_size)
    attention_residual = hidden + projected_attention

    normalized_mlp = torch.nn.functional.layer_norm(
        attention_residual,
        (hidden_size,),
        parameters["ln_2_weight"],
        parameters["ln_2_bias"],
        eps=epsilon,
    )
    expanded = torch.addmm(
        parameters["mlp_expansion_bias"],
        normalized_mlp.reshape(1, hidden_size),
        parameters["mlp_expansion_weight"],
    )
    activated = gelu_new(expanded)
    projected_mlp = torch.addmm(
        parameters["mlp_projection_bias"],
        activated,
        parameters["mlp_projection_weight"],
    ).reshape(hidden_size)
    return attention_residual + projected_mlp, attention_max_hints


def validate_shapes(
    hidden: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    hidden_size: int,
    inner_size: int,
) -> None:
    expected_shapes = {
        "ln_1_weight": (hidden_size,),
        "ln_1_bias": (hidden_size,),
        "attention_qkv_weight": (hidden_size, 3 * hidden_size),
        "attention_qkv_bias": (3 * hidden_size,),
        "attention_projection_weight": (hidden_size, hidden_size),
        "attention_projection_bias": (hidden_size,),
        "ln_2_weight": (hidden_size,),
        "ln_2_bias": (hidden_size,),
        "mlp_expansion_weight": (hidden_size, inner_size),
        "mlp_expansion_bias": (inner_size,),
        "mlp_projection_weight": (inner_size, hidden_size),
        "mlp_projection_bias": (hidden_size,),
    }
    if hidden.shape != (hidden_size,):
        raise RuntimeError(f"hidden-state shape mismatch: {tuple(hidden.shape)}")
    for name, expected in expected_shapes.items():
        tensor = parameters[name]
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} must be BF16, found {tensor.dtype}")
        if tuple(tensor.shape) != expected:
            raise RuntimeError(f"{name} shape mismatch: {tuple(tensor.shape)} != {expected}")


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    hidden_path = args.hidden_state.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "recursion" / "block0-once"
    )
    for path in (model_path, config_path, hidden_path):
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["n_embd"])
    num_heads = int(config["n_head"])
    inner_size = int(config.get("n_inner") or 4 * hidden_size)
    epsilon = float(config["layer_norm_epsilon"])
    hidden = read_bf16_hex(hidden_path)
    with safe_open(model_path, framework="pt", device="cpu") as file:
        parameters = {name: file.get_tensor(key) for name, key in PARAMETERS.items()}
    validate_shapes(hidden, parameters, hidden_size, inner_size)

    with torch.inference_mode():
        pytorch_output, pytorch_attention_max_hints = pytorch_block0_reference(
            hidden, parameters, num_heads, epsilon
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, object] = {}
    files["hidden_state"] = write_bf16_binary(
        output_dir / "hidden_state.bf16.bin", hidden
    )
    for name, tensor in parameters.items():
        descriptor = write_bf16_binary(output_dir / f"{name}.bf16.bin", tensor)
        descriptor["safetensors_key"] = PARAMETERS[name]
        files[name] = descriptor
    files["pytorch_attention_max_hints"] = write_bf16_binary(
        output_dir / "pytorch_attention_max_hints.bf16.bin",
        pytorch_attention_max_hints,
    )
    files["pytorch_block0_output"] = write_bf16_binary(
        output_dir / "pytorch_block0_output.bf16.bin", pytorch_output
    )
    write_bf16_hex(
        output_dir / "pytorch_attention_max_hints.bf16.hex",
        pytorch_attention_max_hints,
    )
    write_bf16_hex(output_dir / "pytorch_block0_output.bf16.hex", pytorch_output)

    metadata = {
        "format_version": 1,
        "model_type": config["model_type"],
        "block": 0,
        "prompt_stage": "token_embedding_plus_position_embedding",
        "dtype": "bfloat16",
        "encoding": "IEEE 754 BF16 raw u16",
        "binary_endianness": "little",
        "flatten_order": "row-major",
        "hidden_size": hidden_size,
        "inner_size": inner_size,
        "num_heads": num_heads,
        "head_dimension": hidden_size // num_heads,
        "layer_norm_epsilon": epsilon,
        "attention_scale": (hidden_size // num_heads) ** -0.5,
        "attention_hint_semantics": (
            "one externally supplied BF16 maximum score per head; the circuit does not "
            "constrain that it is the true maximum"
        ),
        "files": files,
        "source": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "hidden_state": str(hidden_path),
            "hidden_state_sha256": sha256_file(hidden_path),
        },
        "torch_version": torch.__version__,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    parameter_count = sum(tensor.numel() for tensor in parameters.values())
    print(f"model={model_path}")
    print(f"hidden_state={hidden_path}")
    print(f"block=0 parameters={parameter_count} bytes={parameter_count * 2}")
    print(f"pytorch_max_hints={pytorch_attention_max_hints.numel()}")
    print(f"output_dir={output_dir}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
