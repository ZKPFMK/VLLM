#!/usr/bin/env python3
"""Run local GPT-2 text generation with BF16 safetensors weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT.parent / "sp1-models" / "gpt2-bf16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline inference with the converted GPT-2 BF16 checkpoint."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("GPT2_BF16_MODEL_DIR", DEFAULT_MODEL_DIR)),
        help="Local model directory (default: %(default)s)",
    )
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "mps", "cuda"),
        help="PyTorch device; CPU is the tested default",
    )
    parser.add_argument("--sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    model_dir = args.model_dir.expanduser().resolve()
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"invalid model directory {model_dir}: missing {', '.join(missing)}"
        )
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be greater than zero")
    if args.sample and args.temperature <= 0:
        raise SystemExit("--temperature must be greater than zero when sampling")
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    return model_dir


def load_model(model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        names = ", ".join(sorted(str(dtype) for dtype in floating_dtypes))
        raise RuntimeError(f"expected only BF16 model parameters, found: {names}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def main() -> None:
    args = parse_args()
    model_dir = validate_args(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tokenizer, model = load_model(model_dir, device)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and prompt_tokens + args.max_new_tokens > max_positions:
        raise SystemExit(
            f"prompt plus output exceeds the model context: "
            f"{prompt_tokens} + {args.max_new_tokens} > {max_positions}"
        )

    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.sample:
        generation.update(temperature=args.temperature, top_k=args.top_k)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model={model_dir}")
    print(f"device={device} dtype={next(model.parameters()).dtype}")
    print(f"parameters={parameter_count} prompt_tokens={prompt_tokens}")
    print("---")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
