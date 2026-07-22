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

## zkGPT-like comparison circuit

The `zkgpt_like` recursion example models the computation boundary of the zkGPT
prototype: 12 layers, a fixed 30-token sequence, hidden size 768, 12 heads, a
shared QKV/MLP width of 2304, bias-free matrices, and no residual connections,
`ln_f`, or LM head. It keeps the repository's BF16 chips so architecture and
proof-system overhead can be compared independently from zkGPT's fixed-point
arithmetic.

Export the real pretrained GPT-2 BF16 parameters into this layout:

```bash
conda activate sp1-bf16
python infer/gpt2_zkgpt_bf16_export.py
```

The default output is
`../sp1-models/gpt2-bf16/recursion/zkgpt-like-12x30-real-bf16`. QKV,
attention projection, and LayerNorm tensors are copied exactly from all 12
checkpoint layers. GPT-2's `[768, 3072]` expansion is cropped to its first 2304
columns and its `[3072, 768]` projection to its first 2304 rows. Linear biases,
residuals, `ln_f`, and the LM head remain omitted to preserve the zkGPT
architecture. The exporter also writes a real 30-token BF16 embedding input and
causal-attention maximum hints.

Validate every binary tensor shape without building the circuit:

```bash
cargo run -p sp1-recursion-compiler --release --example zkgpt_like -- \
  --check-data
```

Estimate the full event and trace-row counts without materializing billions of
operations:

```bash
cargo run -p sp1-recursion-compiler --release --example zkgpt_like -- --estimate
```

Execute the same circuit at a safe reduced shape and verify the analytical event
counts against the executor record:

```bash
RAYON_NUM_THREADS=8 cargo run -p sp1-recursion-compiler --release \
  --example zkgpt_like -- --small --execute
```

For a sequence longer than one token, the two LayerNorm stages, four row-wise
matrix products, causal attention queries, and row-wise GELU stage are emitted as
`DslIr::Parallel` blocks. The recursion executor runs their token rows through
Rayon; set `RAYON_NUM_THREADS` to the number of CPU cores allocated to the job.
This setting accelerates the logged `executed` phase. `AsmCompiler` currently
compiles the parallel subprograms serially, so it does not accelerate the logged
`built` or `compiled` phases.

Run a small end-to-end proof (setup, prove, verify, and serialize):

```bash
cargo run -p sp1-recursion-compiler --release --example zkgpt_like -- \
  --small --prove --output-dir /tmp/sp1-zkgpt-like-small-proof
```

`--small` selects the deterministic synthetic fixture because checkpoint tensors
have the fixed full shape. Full-size `--build`, `--execute`, and `--prove` modes
load the real BF16 fixture by default. Use `--data-dir PATH` to override its
location; `--synthetic` is available only for explicit structural experiments.

On a machine provisioned for the full experiment, build the release binary first
and then run the complete 12-layer, 30-token proof without including Rust compile
time in the measurement:

```bash
cargo build -p sp1-recursion-compiler --release --example zkgpt_like

/usr/bin/time -l target/release/examples/zkgpt_like \
  --prove --allow-large-build \
  --output-dir /data/sp1-zkgpt-like-full-proof
```

Set `SP1_ZKGPT_LIKE_OUTPUT_DIR` instead of `--output-dir` if preferred. The log
reports circuit construction, compilation, execution, setup, proving,
verification, serialization, trace heights, and artifact sizes separately.

Materializing the full shape requires about 4.27 billion BF16 events. The BF16
executor record alone has a computed lower bound of about 429 GiB, before the IR,
main/preprocessed traces, PCS workspace, and proving keys. A monolithic proof
uses `max_log_row_count = 28`, the largest value accepted by this verifier. Plan
for substantially more than 429 GiB of RAM (practically a high-memory server;
1 TiB is the conservative starting point) and ample fast local storage. The
operation is blocked unless `--allow-large-build` is explicitly supplied; do
not run the full command on an ordinary workstation.

### Bounded attention-head leaf shards

The `zkgpt_leaf` example proves the first bounded stage without materializing the
monolithic 12-layer circuit. One leaf performs `ln_1`, one head's bias-free QKV
projection, and its full 30-token causal attention. Packing 12 BF16 arithmetic
events per trace row keeps the real leaf below `2^19` rows.

Build once and prove all 12 heads of layer zero sequentially:

```bash
cargo build -p sp1-recursion-compiler --release --example zkgpt_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_leaf \
  --all-heads --prove --layer 0 \
  --output-dir /tmp/sp1-zkgpt-layer0-attention
```

The batch reuses one setup and one verifying key, verifies every head proof, and
checks that all heads have the same input commitment. It then concatenates the
12 private `[30, 64]` outputs in head order into a private `[30, 768]` BF16 file
and writes a JSON manifest containing the ordered child commitments, proof
timings, artifact sizes, and an attention-group output commitment. The group
concatenation is currently checked by the host; it is not yet a recursive
aggregation proof.

Generate the cryptographic fan-in proof after the 12 leaf artifacts exist:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_attention_join

target/release/examples/zkgpt_attention_join \
  --prove --layer 0 \
  --leaf-dir /tmp/sp1-zkgpt-layer0-attention \
  --output-dir /tmp/sp1-zkgpt-layer0-attention-join
```

The host first verifies all 12 leaf proofs with their shared verifying key. The
join circuit then recomputes every private head-output commitment and leaf
transcript, enforces head order, concatenates the outputs, and proves the group
output commitment. The resulting manifest records the ordered child transcript
digests needed to verify the multi-proof statement. This stage composes the
proofs cryptographically but does not yet verify the child STARKs recursively
inside one circuit.

Prove the following bias-free attention projection as four bounded output-column
tiles, reusing one setup:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_c_proj_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_c_proj_leaf \
  --all-tiles --prove --layer 0 \
  --attention-dir /tmp/sp1-zkgpt-layer0-attention \
  --join-dir /tmp/sp1-zkgpt-layer0-attention-join \
  --output-dir /tmp/sp1-zkgpt-layer0-c-proj
```

Each tile computes `[30, 768] x [768, 192] -> [30, 192]`, using the real BF16
column slice of `attention_projection_weight.bf16.bin`. The circuit recomputes
the private input commitment and constrains it to the verified Attention join
output. Four ordered outputs are concatenated into a private `[30, 768]` file;
their group manifest is host-generated pending the next lightweight fan-in
proof.
