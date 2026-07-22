#!/usr/bin/env python3
"""Export real GPT-2 BF16 weights in the simplified zkGPT tensor layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from gpt2_bf16 import DEFAULT_MODEL_DIR


DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_LINEAR_SIZE = 2304
DEFAULT_PROMPT = "Once upon a time, a curious machine learned to verify every step of its reasoning."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the real pretrained GPT-2 BF16 tensors used by the zkGPT-like "
            "12-layer, 30-token circuit."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--linear-size", type=int, default=DEFAULT_LINEAR_SIZE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def raw_bf16_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor, found {tensor.dtype}")
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
    return raw.astype("<u2", copy=False).tobytes(order="C")


def write_bf16(path: Path, tensor: torch.Tensor, source: str) -> dict[str, object]:
    data = raw_bf16_bytes(tensor)
    path.write_bytes(data)
    return {
        "file": str(path.name),
        "source": source,
        "shape": list(tensor.shape),
        "values": tensor.numel(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def layer_keys(layer: int) -> dict[str, str]:
    prefix = f"h.{layer}"
    return {
        "ln_1_weight": f"{prefix}.ln_1.weight",
        "ln_1_bias": f"{prefix}.ln_1.bias",
        "attention_qkv_weight": f"{prefix}.attn.c_attn.weight",
        "attention_projection_weight": f"{prefix}.attn.c_proj.weight",
        "ln_2_weight": f"{prefix}.ln_2.weight",
        "ln_2_bias": f"{prefix}.ln_2.bias",
        "mlp_expansion_weight": f"{prefix}.mlp.c_fc.weight",
        "mlp_projection_weight": f"{prefix}.mlp.c_proj.weight",
    }


def gelu_new(values: torch.Tensor) -> torch.Tensor:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * values * (
        1.0 + torch.tanh(coefficient * (values + 0.044715 * values**3))
    )


def zkgpt_like_layer(
    hidden: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    num_heads: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length, hidden_size = hidden.shape
    head_dimension = hidden_size // num_heads

    normalized = torch.nn.functional.layer_norm(
        hidden,
        (hidden_size,),
        parameters["ln_1_weight"],
        parameters["ln_1_bias"],
        eps=epsilon,
    )
    qkv = normalized @ parameters["attention_qkv_weight"]
    query, key, value = qkv.split(hidden_size, dim=-1)
    query = query.reshape(sequence_length, num_heads, head_dimension).transpose(0, 1)
    key = key.reshape(sequence_length, num_heads, head_dimension).transpose(0, 1)
    value = value.reshape(sequence_length, num_heads, head_dimension).transpose(0, 1)

    scale = torch.tensor(head_dimension**-0.5, dtype=torch.bfloat16)
    scores = (query @ key.transpose(-1, -2)) * scale
    causal = torch.ones(
        (sequence_length, sequence_length), dtype=torch.bool
    ).tril()
    masked_scores = scores.masked_fill(~causal, float("-inf"))
    attention_max_hints = masked_scores.max(dim=-1).values.transpose(0, 1).contiguous()
    probabilities = torch.softmax(masked_scores, dim=-1)
    context = probabilities @ value
    context = context.transpose(0, 1).contiguous().reshape(sequence_length, hidden_size)

    projected_attention = context @ parameters["attention_projection_weight"]
    normalized_mlp = torch.nn.functional.layer_norm(
        projected_attention,
        (hidden_size,),
        parameters["ln_2_weight"],
        parameters["ln_2_bias"],
        eps=epsilon,
    )
    expanded = normalized_mlp @ parameters["mlp_expansion_weight"]
    activated = gelu_new(expanded)
    return activated @ parameters["mlp_projection_weight"], attention_max_hints


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    for path in (model_path, config_path, tokenizer_path):
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")
    if args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be positive")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["n_embd"])
    num_heads = int(config["n_head"])
    num_layers = int(config["n_layer"])
    checkpoint_inner_size = int(config.get("n_inner") or 4 * hidden_size)
    epsilon = float(config["layer_norm_epsilon"])
    if args.linear_size <= 0 or args.linear_size > checkpoint_inner_size:
        raise SystemExit(
            f"--linear-size must be in [1, {checkpoint_inner_size}], found {args.linear_size}"
        )
    if args.linear_size != 3 * hidden_size:
        raise SystemExit(
            f"zkGPT layout requires linear-size=3*hidden={3 * hidden_size}, "
            f"found {args.linear_size}"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir
        / "recursion"
        / f"zkgpt-like-{num_layers}x{args.sequence_length}-real-bf16"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        args.prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=args.sequence_length,
    )["input_ids"]
    original_token_count = len(encoded)
    input_ids = encoded[: args.sequence_length]
    input_ids.extend([tokenizer.pad_token_id] * (args.sequence_length - len(input_ids)))
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    position_ids = torch.arange(args.sequence_length, dtype=torch.long)

    layer_metadata: list[dict[str, object]] = []
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        hidden = (
            checkpoint.get_tensor("wte.weight")[input_ids_tensor]
            + checkpoint.get_tensor("wpe.weight")[position_ids]
        ).to(torch.bfloat16)
        input_descriptor = write_bf16(
            output_dir / "hidden_state.bf16.bin",
            hidden,
            "wte.weight[input_ids] + wpe.weight[position_ids]",
        )

        with torch.inference_mode():
            for layer in range(num_layers):
                keys = layer_keys(layer)
                parameters = {name: checkpoint.get_tensor(key) for name, key in keys.items()}
                parameters["mlp_expansion_weight"] = parameters[
                    "mlp_expansion_weight"
                ][:, : args.linear_size].contiguous()
                parameters["mlp_projection_weight"] = parameters[
                    "mlp_projection_weight"
                ][: args.linear_size, :].contiguous()

                layer_dir = output_dir / f"layer-{layer:02d}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                files: dict[str, object] = {}
                for name, tensor in parameters.items():
                    source = keys[name]
                    if name == "mlp_expansion_weight":
                        source += f"[:, :{args.linear_size}]"
                    elif name == "mlp_projection_weight":
                        source += f"[:{args.linear_size}, :]"
                    files[name] = write_bf16(
                        layer_dir / f"{name}.bf16.bin", tensor, source
                    )

                hidden, attention_max_hints = zkgpt_like_layer(
                    hidden, parameters, num_heads, epsilon
                )
                if not torch.isfinite(hidden.float()).all():
                    raise RuntimeError(f"non-finite reference hidden state after layer {layer}")
                if not torch.isfinite(attention_max_hints.float()).all():
                    raise RuntimeError(f"non-finite attention hints at layer {layer}")
                files["attention_max_hints"] = write_bf16(
                    layer_dir / "attention_max_hints.bf16.bin",
                    attention_max_hints,
                    "PyTorch BF16 causal QK score maxima for the simplified layer",
                )
                layer_metadata.append({"layer": layer, "files": files})
                print(
                    f"exported layer={layer} parameters="
                    f"{sum(tensor.numel() for tensor in parameters.values())}"
                )

        output_descriptor = write_bf16(
            output_dir / "pytorch_reference_output.bf16.bin",
            hidden,
            "PyTorch BF16 simplified zkGPT-like reference",
        )

    metadata = {
        "format_version": 1,
        "architecture": "zkGPT-like GPT-2 with BF16 arithmetic",
        "dtype": "bfloat16",
        "encoding": "IEEE 754 BF16 raw u16, little-endian",
        "flatten_order": "row-major",
        "layers": num_layers,
        "sequence_length": args.sequence_length,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dimension": hidden_size // num_heads,
        "linear_size": args.linear_size,
        "checkpoint_inner_size": checkpoint_inner_size,
        "mlp_adaptation": {
            "expansion": f"take columns [0, {args.linear_size}) from [768, 3072]",
            "projection": f"take rows [0, {args.linear_size}) from [3072, 768]",
        },
        "omitted": [
            "all four linear biases",
            "both residual additions",
            "ln_f",
            "LM head",
        ],
        "layer_norm_epsilon": epsilon,
        "attention_scale": (hidden_size // num_heads) ** -0.5,
        "prompt": args.prompt,
        "original_token_count": original_token_count,
        "input_ids": input_ids,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "input": input_descriptor,
        "layer_data": layer_metadata,
        "reference_output": output_descriptor,
        "source": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
        },
        "torch_version": torch.__version__,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total_parameters = sum(
        int(file["values"])
        for layer in layer_metadata
        for name, file in layer["files"].items()
        if name != "attention_max_hints"
    )
    print(f"model={model_path}")
    print(f"output_dir={output_dir}")
    print(f"shape=[{num_layers}, {args.sequence_length}, {hidden_size}]")
    print(f"real_bf16_parameters={total_parameters} bytes={total_parameters * 2}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
