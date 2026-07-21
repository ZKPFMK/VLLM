#!/usr/bin/env python3
"""Export GPT-2's post-embedding hidden state as raw BF16 values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from gpt2_bf16 import DEFAULT_MODEL_DIR, load_model


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "crates" / "recursion" / "data" / "gpt2" / "once"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export token_embedding + position_embedding before GPT-2 block 0."
        )
    )
    parser.add_argument("--prompt", default="once")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"invalid model directory {model_dir}: missing {', '.join(missing)}"
        )
    return model_dir


def main() -> None:
    args = parse_args()
    model_dir = validate_model_dir(args.model_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    tokenizer, model = load_model(model_dir, device)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(sequence_length, dtype=torch.long, device=device)

    with torch.inference_mode():
        token_embedding = model.transformer.wte(input_ids)
        position_embedding = model.transformer.wpe(position_ids)
        hidden_state = model.transformer.drop(token_embedding + position_embedding)

        # Hugging Face defines hidden_states[0] as the value entering block 0.
        transformer_output = model.transformer(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        library_hidden_state = transformer_output.hidden_states[0]

    if hidden_state.dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 hidden state, found {hidden_state.dtype}")
    if not torch.equal(hidden_state, library_hidden_state):
        raise RuntimeError("manual embedding result differs from model hidden_states[0]")

    hidden_state = hidden_state.detach().cpu().contiguous()
    raw_values = hidden_state.view(torch.uint16).reshape(-1).tolist()
    values_path = output_dir / "hidden_state.bf16.hex"
    values_text = "".join(f"{value:04X}\n" for value in raw_values)
    values_path.write_text(values_text, encoding="ascii")

    input_id_values = input_ids.detach().cpu().reshape(-1).tolist()
    metadata = {
        "format_version": 1,
        "model_type": model.config.model_type,
        "prompt": args.prompt,
        "input_ids": input_id_values,
        "tokens": tokenizer.convert_ids_to_tokens(input_id_values),
        "stage": "token_embedding_plus_position_embedding",
        "huggingface_hidden_states_index": 0,
        "transformer_block_input": 0,
        "dropout_active": False,
        "dtype": "bfloat16",
        "shape": list(hidden_state.shape),
        "flatten_order": "row-major",
        "value_encoding": "IEEE 754 BF16 raw bits as one uppercase hex u16 per line",
        "values_file": values_path.name,
        "values_count": len(raw_values),
        "values_sha256": hashlib.sha256(values_text.encode("ascii")).hexdigest(),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "model_weights_sha256": sha256_file(model_dir / "model.safetensors"),
        "tokenizer_sha256": sha256_file(model_dir / "tokenizer.json"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"prompt={args.prompt!r}")
    print(f"input_ids={input_id_values}")
    print(f"tokens={metadata['tokens']}")
    print(f"shape={list(hidden_state.shape)} dtype={hidden_state.dtype}")
    print(f"values={values_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
