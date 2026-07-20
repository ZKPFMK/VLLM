#!/usr/bin/env python3
"""Export real GPT-2 block-0 LayerNorm parameters and PyTorch reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from gpt2_bf16 import DEFAULT_MODEL_DIR


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HIDDEN_STATE = (
    REPO_ROOT / "crates" / "recursion" / "data" / "gpt2" / "once" / "hidden_state.bf16.hex"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/sp1-gpt2-ln1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real GPT-2 ln_1 data without copying model parameters into the repo."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hidden-state", type=Path, default=DEFAULT_HIDDEN_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_bf16_hex(path: Path) -> torch.Tensor:
    raw = [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()]
    return torch.tensor(raw, dtype=torch.uint16).view(torch.bfloat16)


def write_bf16_hex(path: Path, tensor: torch.Tensor) -> None:
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).reshape(-1).tolist()
    path.write_text("".join(f"{value:04X}\n" for value in raw), encoding="ascii")


def main() -> None:
    args = parse_args()
    model_path = args.model_dir.expanduser().resolve() / "model.safetensors"
    hidden_path = args.hidden_state.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden = read_bf16_hex(hidden_path)
    with safe_open(model_path, framework="pt", device="cpu") as file:
        weight = file.get_tensor("h.0.ln_1.weight")
        bias = file.get_tensor("h.0.ln_1.bias")
        c_attn_weight = file.get_tensor("h.0.attn.c_attn.weight")
        c_attn_bias = file.get_tensor("h.0.attn.c_attn.bias")

    if hidden.shape != weight.shape or hidden.shape != bias.shape:
        raise RuntimeError(
            f"shape mismatch: hidden={tuple(hidden.shape)}, "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )
    if {hidden.dtype, weight.dtype, bias.dtype} != {torch.bfloat16}:
        raise RuntimeError("hidden state and LayerNorm parameters must all be BF16")

    with torch.inference_mode():
        pytorch_bf16 = torch.nn.functional.layer_norm(
            hidden,
            (hidden.numel(),),
            weight,
            bias,
            eps=1e-5,
        )
        pytorch_fp32 = torch.nn.functional.layer_norm(
            hidden.float(),
            (hidden.numel(),),
            weight.float(),
            bias.float(),
            eps=1e-5,
        )
        c_attn_column_0 = c_attn_weight[:, 0].contiguous()
        c_attn_bias_0 = c_attn_bias[0].reshape(1).contiguous()
        c_attn_output_bf16 = torch.addmm(
            c_attn_bias,
            pytorch_bf16.reshape(1, -1),
            c_attn_weight,
        )[0, 0].reshape(1)
        c_attn_output_fp32 = (
            torch.dot(pytorch_bf16.float(), c_attn_column_0.float())
            + c_attn_bias_0.float()[0]
        ).reshape(1)

    write_bf16_hex(output_dir / "weight.bf16.hex", weight)
    write_bf16_hex(output_dir / "bias.bf16.hex", bias)
    write_bf16_hex(output_dir / "pytorch_output.bf16.hex", pytorch_bf16)
    (output_dir / "pytorch_output.fp32.txt").write_text(
        "".join(f"{value:.9e}\n" for value in pytorch_fp32.tolist()),
        encoding="ascii",
    )
    write_bf16_hex(output_dir / "c_attn_weight_col0.bf16.hex", c_attn_column_0)
    write_bf16_hex(output_dir / "c_attn_bias0.bf16.hex", c_attn_bias_0)
    write_bf16_hex(output_dir / "c_attn_col0_pytorch_output.bf16.hex", c_attn_output_bf16)
    (output_dir / "c_attn_col0_pytorch_output.fp32.txt").write_text(
        f"{c_attn_output_fp32.item():.9e}\n",
        encoding="ascii",
    )

    print(f"model={model_path}")
    print(f"hidden_state={hidden_path}")
    print(f"shape={tuple(hidden.shape)} dtype={hidden.dtype}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
