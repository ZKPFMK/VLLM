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

### One proof per Block and one final recursion proof

The server-oriented path avoids the 12-Block monolithic trace. Each invocation
of `zkgpt_like --block N` proves exactly one complete GPT-2 Block and commits its
input, private parameters, hints, output, and the previous Block transcript.
The following Block verifies the previous proof on the host, consumes its
committed private output, and proves the next link.

After all 12 Block proofs exist, `zkgpt_block_recursion` performs actual
in-circuit child-STARK verification. Every join verifies two child proofs and
their verifying-key hashes, checks the Block range and the input/output and
transcript boundaries, and emits one recursion proof. The runner constructs the
binary reduction:

```text
12 Block proofs -> 6 -> 3 -> 2 -> 1 final recursion proof
```

This produces 12 Block proofs plus 11 recursion proofs during the full run, but
the final artifact is one proof covering Blocks `0..12`. At an odd level the
unpaired node is carried forward; no redundant identity proof is generated.

Build and run this path on the Linux proving server:

```bash
python3 infer/zkgpt_block_recursion.py \
  --prove --build --resume \
  --threads "$(nproc)" \
  --data-dir /home/dj/VLLM-models/gpt2-bf16/recursion/zkgpt-like-12x30-real-bf16 \
  --output-root /home/dj/proofs/zkgpt-12-block-recursion
```

`--resume` validates and skips completed Block and recursion manifests. The
runner writes `zkgpt_block_recursion.run.json` and prints the final recursion
manifest path. After completion, verify the serialized final proof, its public
transcript, and its verifying-key commitment again with:

```bash
python3 infer/zkgpt_block_recursion.py \
  --check-only \
  --output-root /home/dj/proofs/zkgpt-12-block-recursion
```

To inspect the complete command plan without proving:

```bash
python3 infer/zkgpt_block_recursion.py \
  --dry-run \
  --output-root /home/dj/proofs/zkgpt-12-block-recursion
```

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

Prove the following bias-free attention projection as three bounded output-column
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

Each tile computes `[30, 768] x [768, 256] -> [30, 256]`, using the real BF16
column slice of `attention_projection_weight.bf16.bin`. The circuit recomputes
the private input commitment and constrains it to the verified Attention join
output. Three ordered outputs are concatenated into a private `[30, 768]` file;
their group manifest is host-generated for the following lightweight fan-in
proof.

Bind the three private tile outputs and their proof transcripts into one proved
`[30, 768]` c_proj output:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_c_proj_join

target/release/examples/zkgpt_c_proj_join \
  --prove --layer 0 \
  --tile-dir /tmp/sp1-zkgpt-layer0-c-proj \
  --output-dir /tmp/sp1-zkgpt-layer0-c-proj-join
```

The host verifies all three tile proofs with their shared verifying key. The join
circuit recomputes every tile-output commitment and tile transcript, enforces a
common Attention input and upstream transcript, preserves output-column order,
and proves exactly the c_proj group transcript already recorded by the tile
batch. As with the Attention fan-in, child STARK verification remains on the
host; the join circuit cryptographically binds the verified child public
transcripts to the combined private output.

Prove the second LayerNorm once over the complete c_proj output:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_ln2_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_ln2_leaf \
  --prove --layer 0 \
  --c-proj-dir /tmp/sp1-zkgpt-layer0-c-proj \
  --join-dir /tmp/sp1-zkgpt-layer0-c-proj-join \
  --output-dir /tmp/sp1-zkgpt-layer0-ln2
```

The host verifies the c_proj join proof, and the LN2 circuit recomputes the
private `[30, 768]` input commitment before applying the real private
`ln_2_weight` and `ln_2_bias` tensors with bit-exact BF16 operations. LayerNorm
is independent across tokens and this complete stage fits in `2^16` rows, so it
uses one proof and needs no fan-in join. Its private `[30, 768]` output is the
shared input to the following MLP-expansion tiles.

Prove the bias-free MLP expansion and `gelu_new` activation as nine output-column
tiles with one shared setup:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_mlp_expansion_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_mlp_expansion_leaf \
  --all-tiles --prove --layer 0 \
  --ln2-dir /tmp/sp1-zkgpt-layer0-ln2 \
  --output-dir /tmp/sp1-zkgpt-layer0-mlp-expansion
```

Each tile computes `[30, 768] x [768, 256] -> [30, 256]` and applies the BF16
`gelu_new` lookup to all 7,680 results. The nine circuits share the verified LN2
input commitment and use the corresponding private column slices of the real
`mlp_expansion_weight.bf16.bin`. Their ordered private outputs are concatenated
into `[30, 2304]`; the generated group manifest remains host-computed until the
following MLP-expansion fan-in proof binds all child transcripts.

Bind the nine expansion+GELU outputs into one proved `[30, 2304]` tensor:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_mlp_expansion_join

target/release/examples/zkgpt_mlp_expansion_join \
  --prove --layer 0 \
  --tile-dir /tmp/sp1-zkgpt-layer0-mlp-expansion \
  --output-dir /tmp/sp1-zkgpt-layer0-mlp-expansion-join
```

The host verifies all nine expansion proofs with their shared verifying key. The
join circuit checks the private tile-output commitments and transcripts,
enforces a common LN2 input and ordered, complete output-column coverage, and
proves the same group transcript as the tile batch. Its verified `[30, 2304]`
output is the input to the following MLP projection shards.

Prove the bias-free MLP projection as 12 output-column tiles with one shared
setup:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_mlp_projection_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_mlp_projection_leaf \
  --all-tiles --prove --layer 0 \
  --expansion-dir /tmp/sp1-zkgpt-layer0-mlp-expansion \
  --join-dir /tmp/sp1-zkgpt-layer0-mlp-expansion-join \
  --output-dir /tmp/sp1-zkgpt-layer0-mlp-projection
```

Each tile computes `[30, 2304] x [2304, 64] -> [30, 64]` using the matching
private output-column slice of the real `mlp_projection_weight.bf16.bin`. There
is no second activation, linear bias, or residual addition in the simplified
zkGPT comparison architecture. The 12 ordered private outputs are concatenated
into the block's `[30, 768]` output; its host group manifest is bound by the
following block-output join proof.

Verify the 12 projection proofs and bind their ordered outputs into the final
proved `[30, 768]` output of layer zero:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_mlp_projection_join

target/release/examples/zkgpt_mlp_projection_join \
  --prove --layer 0 \
  --tile-dir /tmp/sp1-zkgpt-layer0-mlp-projection \
  --output-dir /tmp/sp1-zkgpt-layer0-mlp-projection-join
```

The host verifies all 12 projection proofs with their shared verifying key. The
join circuit recomputes every private `[30, 64]` output commitment and child
transcript, enforces the common MLP-expansion input and tile order, concatenates
the columns, and proves the block-output transcript. As with the preceding join
stages, child STARK verification is currently host-side; the join circuit binds
the verified child statements but does not recursively verify the child STARKs
inside the circuit. Because this comparison architecture omits the residual,
the joined projection tensor is the simplified block output and can be chained
to the next layer's attention leaf shards.

Start the next block by using the preceding projection tensor instead of the
genesis embedding input:

```bash
cargo build -p sp1-recursion-compiler --release --example zkgpt_leaf

RAYON_NUM_THREADS=8 target/release/examples/zkgpt_leaf \
  --all-heads --prove --layer 1 \
  --previous-block-dir /tmp/sp1-zkgpt-layer0-mlp-projection \
  --previous-block-join-dir /tmp/sp1-zkgpt-layer0-mlp-projection-join \
  --output-dir /tmp/sp1-zkgpt-layer1-attention
```

For layer 1 or later, the host verifies the previous layer's block-output join
proof once and checks that its private `[30, 768]` tensor hashes to the proved
output commitment. Each attention leaf repeats that hash constraint inside the
circuit and includes both the preceding block transcript and output commitment
in its own public transcript. Layer zero retains the original genesis-input
transcript layout. Previous-proof verification remains host-side rather than an
in-circuit recursive STARK verification.

Join the 12 chained attention leaves before continuing with layer one's
attention projection:

```bash
cargo build -p sp1-recursion-compiler --release \
  --example zkgpt_attention_join

target/release/examples/zkgpt_attention_join \
  --prove --layer 1 \
  --leaf-dir /tmp/sp1-zkgpt-layer1-attention \
  --output-dir /tmp/sp1-zkgpt-layer1-attention-join
```

For a non-genesis layer, the join circuit reconstructs the chained leaf stage,
requires all heads to use the same previous-block transcript and input
commitment, binds every private head output, and proves the chained attention
group transcript. The resulting proof and private attention tensor can be fed
to `zkgpt_c_proj_leaf` with the same layer index.

For example, prove layer one's three attention-projection tiles with the chained
attention artifacts above:

```bash
RAYON_NUM_THREADS=8 target/release/examples/zkgpt_c_proj_leaf \
  --all-tiles --prove --layer 1 \
  --attention-dir /tmp/sp1-zkgpt-layer1-attention \
  --join-dir /tmp/sp1-zkgpt-layer1-attention-join \
  --output-dir /tmp/sp1-zkgpt-layer1-c-proj
```

The existing c_proj leaf protocol is layer-independent: it verifies the
matching attention join proof, constrains the private attention tensor to that
join's output commitment, and loads the real projection weights from the
requested checkpoint layer.

Bind those three layer-one tiles into a single proved c_proj output:

```bash
target/release/examples/zkgpt_c_proj_join \
  --prove --layer 1 \
  --tile-dir /tmp/sp1-zkgpt-layer1-c-proj \
  --output-dir /tmp/sp1-zkgpt-layer1-c-proj-join
```

This is the same bounded fan-in circuit used for layer zero. It verifies the
three tile proofs on the host, enforces ordered and complete output-column
coverage in the join circuit, and preserves the layer-one attention upstream
transcript for the following `ln_2` proof.

Prove layer one's complete second LayerNorm over the joined c_proj output:

```bash
RAYON_NUM_THREADS=8 target/release/examples/zkgpt_ln2_leaf \
  --prove --layer 1 \
  --c-proj-dir /tmp/sp1-zkgpt-layer1-c-proj \
  --join-dir /tmp/sp1-zkgpt-layer1-c-proj-join \
  --output-dir /tmp/sp1-zkgpt-layer1-ln2
```

The host verifies the layer-one c_proj join proof, while the LN2 circuit binds
its private output tensor and applies the real BF16 `ln_2_weight` and
`ln_2_bias` values from checkpoint layer one. The proved `[30, 768]` output is
the common input for layer one's MLP-expansion shards.

Prove layer one's nine MLP expansion and GELU tiles with one shared setup:

```bash
RAYON_NUM_THREADS=8 target/release/examples/zkgpt_mlp_expansion_leaf \
  --all-tiles --prove --layer 1 \
  --ln2-dir /tmp/sp1-zkgpt-layer1-ln2 \
  --output-dir /tmp/sp1-zkgpt-layer1-mlp-expansion
```

Each tile verifies the common layer-one LN2 proof on the host, constrains its
private `[30, 768]` input to the LN2 output commitment, and applies the matching
real BF16 expansion-weight columns followed by `gelu_new`. The nine ordered
outputs form the private `[30, 2304]` tensor consumed by the expansion join.

Bind the nine layer-one expansion outputs into one proved `[30, 2304]` tensor:

```bash
target/release/examples/zkgpt_mlp_expansion_join \
  --prove --layer 1 \
  --tile-dir /tmp/sp1-zkgpt-layer1-mlp-expansion \
  --output-dir /tmp/sp1-zkgpt-layer1-mlp-expansion-join
```

The join verifies all expansion proofs on the host and proves their common LN2
input, upstream transcript, output-column order, private tile commitments, and
group transcript. Its output is the shared input to layer one's 12 MLP
projection tiles.

Prove layer one's 12 bias-free MLP projection tiles with one shared setup:

```bash
RAYON_NUM_THREADS=8 target/release/examples/zkgpt_mlp_projection_leaf \
  --all-tiles --prove --layer 1 \
  --expansion-dir /tmp/sp1-zkgpt-layer1-mlp-expansion \
  --join-dir /tmp/sp1-zkgpt-layer1-mlp-expansion-join \
  --output-dir /tmp/sp1-zkgpt-layer1-mlp-projection
```

Each tile verifies the expansion join proof on the host, constrains the private
`[30, 2304]` input, and applies the matching real BF16 projection-weight
columns. The 12 ordered `[30, 64]` outputs form layer one's private `[30, 768]`
block output, ready for the final block-output join.

Complete layer one by binding those 12 outputs into its block-output proof:

```bash
target/release/examples/zkgpt_mlp_projection_join \
  --prove --layer 1 \
  --tile-dir /tmp/sp1-zkgpt-layer1-mlp-projection \
  --output-dir /tmp/sp1-zkgpt-layer1-mlp-projection-join
```

This produces the second complete block-output commitment in the chained proof
sequence. To start layer two, pass the projection tile directory as
`--previous-block-dir` and this join directory as
`--previous-block-join-dir` to `zkgpt_leaf --layer 2`.

For example, start layer two with 12 chained attention proofs:

```bash
RAYON_NUM_THREADS=8 target/release/examples/zkgpt_leaf \
  --all-heads --prove --layer 2 \
  --previous-block-dir /tmp/sp1-zkgpt-layer1-mlp-projection \
  --previous-block-join-dir /tmp/sp1-zkgpt-layer1-mlp-projection-join \
  --output-dir /tmp/sp1-zkgpt-layer2-attention
```

The same chained-leaf circuit supports every later layer. It verifies the
immediately preceding block-output proof once on the host, constrains that
private output inside every leaf, and includes the previous block transcript in
each new public attention transcript.

Join layer two's 12 attention leaves before its c_proj stage:

```bash
target/release/examples/zkgpt_attention_join \
  --prove --layer 2 \
  --leaf-dir /tmp/sp1-zkgpt-layer2-attention \
  --output-dir /tmp/sp1-zkgpt-layer2-attention-join
```

The resulting proof preserves the Block 1 upstream transcript and binds the
ordered private head outputs into layer two's `[30, 768]` attention tensor.

### Complete reusable 12-block proof runner

`zkgpt_full_inference.py` reuses the same nine bounded stage implementations for
every layer; it does not copy the Block circuit 12 times. Each completed
`mlp_projection_block_join` supplies the private `[30, 768]` block output and
proved transcript consumed by the next layer's chained attention leaves.

The shape-aware planner packs complete natural units without cutting a BF16 dot
product. With the default `2^19` row and Core-style trace-area limits, one block
contains 41 proof instances: 12 attention-head leaves, three c_proj
output-column leaves, one LN2 token-group proof, nine MLP-expansion
output-column leaves, 12 MLP-projection leaves, and four join proofs. The
12-block run therefore produces 492 proof instances. Stages and
layers run sequentially so a completed subprocess releases its memory before
the next stage begins.

Inspect the complete 108-command schedule without executing it:

```bash
python3 infer/zkgpt_full_inference.py \
  --dry-run \
  --output-root /data/sp1-zkgpt-full-12x30
```

Build the nine reusable binaries and generate the complete 12-block proof:

```bash
python3 infer/zkgpt_full_inference.py \
  --prove --build --resume --threads 8 \
  --output-root /data/sp1-zkgpt-full-12x30
```

`--resume` is safe on a new directory. On a later invocation it validates and
skips the complete prefix, then restarts at the first missing stage. Resume
without recompiling the Rust examples:

```bash
python3 infer/zkgpt_full_inference.py \
  --prove --resume --threads 8 \
  --output-root /data/sp1-zkgpt-full-12x30
```

Generate only one complete block by adding `--end-layer 0`. A later run can
continue the same directory with `--resume --start-layer 1`. `--end-stage` can
also stop at a named stage for bounded experiments.

Validate an existing complete run without generating proofs:

```bash
python3 infer/zkgpt_full_inference.py \
  --check-only \
  --output-root /data/sp1-zkgpt-full-12x30
```

Artifacts use `layer-00` through `layer-11`, with the same nine stage
subdirectories in each layer. The runner rejects gaps, missing or empty proof
files, incorrect tensor sizes, unordered child shards, group/join mismatches,
and broken intra-layer or cross-layer commitment links. Per-stage console logs
are stored under `logs/`; invocation status, elapsed wall time, proof counts,
artifact bytes, and the latest complete block commitment are recorded in
`zkgpt_full_inference.run.json` after every stage.

This completes the agreed zkGPT-like 12-block boundary: BF16, sequence length
30, hidden size 768, no residual connections or linear biases, and no `ln_f` or
LM head. Child STARK verification remains host-side as in the individual join
examples; the runner builds and validates the complete chained proof workflow,
not a single recursively aggregated proof.
