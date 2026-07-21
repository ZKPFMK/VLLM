# GPT-2 BF16 inference

This directory runs offline text generation with the converted GPT-2 BF16 checkpoint.

The default model location is:

```text
/Users/dj/program/work/git/sp1-models/gpt2-bf16
```

## Environment

Install the dependencies into the existing Conda environment:

```bash
conda env update --name sp1-bf16 --file infer/environment.yml
conda activate sp1-bf16
```

## Run

Deterministic greedy decoding:

```bash
python infer/gpt2_bf16.py \
  --prompt "Once upon a time" \
  --max-new-tokens 32
```

Sampling:

```bash
python infer/gpt2_bf16.py \
  --prompt "The future of computing is" \
  --max-new-tokens 64 \
  --sample \
  --temperature 0.8 \
  --top-k 50 \
  --seed 42
```

Override the checkpoint with either `--model-dir PATH` or the
`GPT2_BF16_MODEL_DIR` environment variable. The script uses `local_files_only=True`
and does not download model files during inference.

The checkpoint parameters and public tensor dtype are BF16. Individual PyTorch
kernels may still use wider internal accumulators, so this program is an inference
reference, not a guarantee that every intermediate operation follows a custom
bit-exact recursion arithmetic policy.

## Export the embedding hidden state

Export the BF16 vector after adding token and position embeddings, immediately
before transformer block 0:

```bash
python infer/gpt2_embedding_bf16.py --prompt "once"
```

The default output is `crates/recursion/data/gpt2/once`. The hex file contains
the flattened raw BF16 bit patterns; `metadata.json` records its shape, token IDs,
file hashes, and the exact extraction stage.
