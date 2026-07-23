#!/usr/bin/env python3
"""Run the reusable zkGPT-like BF16 Block proof pipeline for all GPT-2 layers.

The arithmetic circuits remain in the nine existing Rust examples.  This host
runner only schedules them, validates every artifact, and connects one block's
private output commitment and transcript to the next block.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


NUM_LAYERS = 12
SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 768
NUM_HEADS = 12
EXPANSION_SIZE = 2304
BF16_BYTES = 2
RUN_STATE_NAME = "zkgpt_full_inference.run.json"
DIGEST_PATTERN = re.compile(r"^[0-9A-Fa-f]{8}(?::[0-9A-Fa-f]{8}){7}$")


class ValidationError(RuntimeError):
    """An artifact, model fixture, or proof chain is incomplete or inconsistent."""


@dataclass(frozen=True)
class Stage:
    key: str
    binary: str
    directory: str
    manifest_name: str
    manifest_stage: str
    proof_instances: int
    collection: str | None = None
    index_field: str | None = None
    child_count: int = 0
    commitments_name: str | None = None
    single_proof: bool = False
    private_output_bytes: int | None = None

    def manifest(self, layer: int) -> str:
        return self.manifest_name.format(layer=layer)

    def commitments(self, layer: int, index: int) -> str:
        if self.commitments_name is None:
            raise AssertionError(f"{self.key} has no child commitments")
        return self.commitments_name.format(layer=layer, index=index)


BLOCK_OUTPUT_BYTES = SEQUENCE_LENGTH * HIDDEN_SIZE * BF16_BYTES
EXPANSION_OUTPUT_BYTES = SEQUENCE_LENGTH * EXPANSION_SIZE * BF16_BYTES

STAGES = (
    Stage(
        key="attention",
        binary="zkgpt_leaf",
        directory="attention",
        manifest_name="zkgpt_attention_l{layer:02d}.manifest.json",
        manifest_stage="qkv_attention_group",
        proof_instances=12,
        collection="heads",
        index_field="head",
        child_count=12,
        commitments_name="zkgpt_leaf_l{layer:02d}_h{index:02d}.commitments.txt",
        private_output_bytes=BLOCK_OUTPUT_BYTES,
    ),
    Stage(
        key="attention-join",
        binary="zkgpt_attention_join",
        directory="attention-join",
        manifest_name="zkgpt_attention_join_l{layer:02d}.manifest.json",
        manifest_stage="attention_join",
        proof_instances=1,
        single_proof=True,
    ),
    Stage(
        key="c-proj",
        binary="zkgpt_c_proj_leaf",
        directory="c-proj",
        manifest_name="zkgpt_c_proj_l{layer:02d}.manifest.json",
        manifest_stage="attention_c_proj_group",
        proof_instances=3,
        collection="tiles",
        index_field="tile",
        child_count=3,
        commitments_name="zkgpt_c_proj_l{layer:02d}_t{index:02d}.commitments.txt",
        private_output_bytes=BLOCK_OUTPUT_BYTES,
    ),
    Stage(
        key="c-proj-join",
        binary="zkgpt_c_proj_join",
        directory="c-proj-join",
        manifest_name="zkgpt_c_proj_join_l{layer:02d}.manifest.json",
        manifest_stage="attention_c_proj_join",
        proof_instances=1,
        single_proof=True,
    ),
    Stage(
        key="ln2",
        binary="zkgpt_ln2_leaf",
        directory="ln2",
        manifest_name="zkgpt_ln2_l{layer:02d}.manifest.json",
        manifest_stage="ln_2",
        proof_instances=1,
        single_proof=True,
        private_output_bytes=BLOCK_OUTPUT_BYTES,
    ),
    Stage(
        key="mlp-expansion",
        binary="zkgpt_mlp_expansion_leaf",
        directory="mlp-expansion",
        manifest_name="zkgpt_mlp_expansion_l{layer:02d}.manifest.json",
        manifest_stage="mlp_expansion_gelu_group",
        proof_instances=9,
        collection="tiles",
        index_field="tile",
        child_count=9,
        commitments_name="zkgpt_mlp_expansion_l{layer:02d}_t{index:02d}.commitments.txt",
        private_output_bytes=EXPANSION_OUTPUT_BYTES,
    ),
    Stage(
        key="mlp-expansion-join",
        binary="zkgpt_mlp_expansion_join",
        directory="mlp-expansion-join",
        manifest_name="zkgpt_mlp_expansion_join_l{layer:02d}.manifest.json",
        manifest_stage="mlp_expansion_gelu_join",
        proof_instances=1,
        single_proof=True,
    ),
    Stage(
        key="mlp-projection",
        binary="zkgpt_mlp_projection_leaf",
        directory="mlp-projection",
        manifest_name="zkgpt_mlp_projection_l{layer:02d}.manifest.json",
        manifest_stage="mlp_projection_group",
        proof_instances=12,
        collection="tiles",
        index_field="tile",
        child_count=12,
        commitments_name="zkgpt_mlp_projection_l{layer:02d}_t{index:02d}.commitments.txt",
        private_output_bytes=BLOCK_OUTPUT_BYTES,
    ),
    Stage(
        key="mlp-projection-join",
        binary="zkgpt_mlp_projection_join",
        directory="mlp-projection-join",
        manifest_name="zkgpt_mlp_projection_join_l{layer:02d}.manifest.json",
        manifest_stage="mlp_projection_block_join",
        proof_instances=1,
        single_proof=True,
    ),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}
PROOF_INSTANCES_PER_BLOCK = sum(stage.proof_instances for stage in STAGES)
FULL_PROOF_INSTANCES = NUM_LAYERS * PROOF_INSTANCES_PER_BLOCK


@dataclass(frozen=True)
class Artifact:
    stage: Stage
    layer: int
    directory: Path
    manifest_path: Path
    data: dict[str, Any]
    proof_files: tuple[Path, ...]
    verifying_key: Path
    private_output: Path | None

    @property
    def upstream(self) -> str | None:
        value = self.data.get("upstream_transcript")
        return value if isinstance(value, str) else None

    @property
    def input(self) -> str:
        return str(self.data["input_commitment"])

    @property
    def output(self) -> str:
        return str(self.data["output_commitment"])

    @property
    def transcript(self) -> str:
        return str(self.data["transcript_commitment"])

    @property
    def proof_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.proof_files)


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_dir(repo_root: Path) -> Path:
    return (
        repo_root.parent
        / "sp1-models"
        / "gpt2-bf16"
        / "recursion"
        / "zkgpt-like-12x30-real-bf16"
    )


def default_bin_dir(repo_root: Path) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", repo_root / "target"))
    if not target.is_absolute():
        target = repo_root / target
    return target / "release" / "examples"


def layer_dir(output_root: Path, layer: int) -> Path:
    return output_root / f"layer-{layer:02d}"


def stage_dir(output_root: Path, layer: int, stage: Stage) -> Path:
    return layer_dir(output_root, layer) / stage.directory


def manifest_path(output_root: Path, layer: int, stage: Stage) -> Path:
    return stage_dir(output_root, layer, stage) / stage.manifest(layer)


def target_stages(end_layer: int, end_stage: str) -> list[tuple[int, Stage]]:
    end_index = next(index for index, stage in enumerate(STAGES) if stage.key == end_stage)
    targets: list[tuple[int, Stage]] = []
    for layer in range(end_layer + 1):
        limit = end_index + 1 if layer == end_layer else len(STAGES)
        targets.extend((layer, stage) for stage in STAGES[:limit])
    return targets


def selected_stages(start_layer: int, end_layer: int, end_stage: str) -> list[tuple[int, Stage]]:
    return [
        (layer, stage)
        for layer, stage in target_stages(end_layer, end_stage)
        if layer >= start_layer
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"manifest must contain a JSON object: {path}")
    return value


def _require_int(data: dict[str, Any], key: str, expected: int, path: Path) -> None:
    if data.get(key) != expected:
        raise ValidationError(
            f"{path}: expected {key}={expected}, found {data.get(key)!r}"
        )


def _require_digest(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{path}: {key} is not an eight-limb digest")
    return value.upper()


def _file_reference(
    directory: Path,
    value: Any,
    label: str,
    manifest: Path,
    expected_bytes: int | None = None,
    reported_bytes: Any = None,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{manifest}: missing {label}")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value:
        raise ValidationError(f"{manifest}: unsafe {label} path {value!r}")
    path = directory / relative
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValidationError(f"{manifest}: missing {label} {path}: {error}") from error
    if size <= 0:
        raise ValidationError(f"{manifest}: empty {label} {path}")
    if expected_bytes is not None and size != expected_bytes:
        raise ValidationError(
            f"{manifest}: {label} has {size} bytes, expected {expected_bytes}"
        )
    if isinstance(reported_bytes, int) and size != reported_bytes:
        raise ValidationError(
            f"{manifest}: {label} has {size} bytes, manifest reports {reported_bytes}"
        )
    return path


def load_artifact(output_root: Path, layer: int, stage: Stage) -> Artifact:
    directory = stage_dir(output_root, layer, stage)
    path = manifest_path(output_root, layer, stage)
    if not path.is_file():
        raise ValidationError(f"missing {stage.key} manifest for layer {layer}: {path}")
    data = _read_json(path)
    if data.get("stage") != stage.manifest_stage:
        raise ValidationError(
            f"{path}: expected stage={stage.manifest_stage!r}, found {data.get('stage')!r}"
        )
    _require_int(data, "layer", layer, path)
    _require_int(data, "sequence_length", SEQUENCE_LENGTH, path)
    _require_int(data, "hidden_size", HIDDEN_SIZE, path)
    if stage.key.startswith("attention"):
        _require_int(data, "num_heads", NUM_HEADS, path)
    if stage.key.startswith("c-proj"):
        _require_int(data, "num_tiles", 3, path)
    if stage.key.startswith("mlp-"):
        _require_int(data, "expansion_size", EXPANSION_SIZE, path)
    if stage.collection == "tiles":
        _require_int(data, "num_tiles", stage.child_count, path)

    for key in (
        "input_commitment",
        "parameters_commitment",
        "output_commitment",
        "transcript_commitment",
    ):
        _require_digest(data, key, path)
    upstream = data.get("upstream_transcript")
    if upstream is not None:
        _require_digest(data, "upstream_transcript", path)

    verifying_key = _file_reference(
        directory,
        data.get("verifying_key_file"),
        "verifying key",
        path,
        reported_bytes=data.get("verifying_key_bytes"),
    )
    proofs: list[Path] = []
    if stage.collection is not None:
        children = data.get(stage.collection)
        if not isinstance(children, list) or len(children) != stage.child_count:
            raise ValidationError(
                f"{path}: expected {stage.child_count} ordered {stage.collection}"
            )
        for index, child in enumerate(children):
            if not isinstance(child, dict) or child.get(stage.index_field) != index:
                raise ValidationError(
                    f"{path}: {stage.collection}[{index}] has the wrong index"
                )
            proofs.append(
                _file_reference(
                    directory,
                    child.get("proof_file"),
                    f"{stage.key} child {index} proof",
                    path,
                    reported_bytes=child.get("proof_bytes"),
                )
            )
            commitments = directory / stage.commitments(layer, index)
            try:
                commitments_size = commitments.stat().st_size
            except OSError as error:
                raise ValidationError(
                    f"{path}: missing child commitments {commitments}: {error}"
                ) from error
            if commitments_size <= 0:
                raise ValidationError(f"{path}: empty child commitments {commitments}")
    elif stage.single_proof:
        proofs.append(
            _file_reference(
                directory,
                data.get("proof_file"),
                f"{stage.key} proof",
                path,
                reported_bytes=data.get("proof_bytes"),
            )
        )

    private_output = None
    if stage.private_output_bytes is not None:
        private_output = _file_reference(
            directory,
            data.get("private_output_file"),
            f"{stage.key} private output",
            path,
            expected_bytes=stage.private_output_bytes,
        )
        output_values = data.get("output_values")
        expected_values = stage.private_output_bytes // BF16_BYTES
        if output_values is not None and output_values != expected_values:
            raise ValidationError(
                f"{path}: output_values={output_values!r}, expected {expected_values}"
            )

    if len(proofs) != stage.proof_instances:
        raise ValidationError(
            f"{path}: found {len(proofs)} proofs, expected {stage.proof_instances}"
        )
    return Artifact(
        stage=stage,
        layer=layer,
        directory=directory,
        manifest_path=path,
        data=data,
        proof_files=tuple(proofs),
        verifying_key=verifying_key,
        private_output=private_output,
    )


def _require_equal(
    left: Artifact, right: Artifact, field: str, description: str
) -> None:
    if left.data.get(field) != right.data.get(field):
        raise ValidationError(
            f"layer {left.layer} {description}: {field} differs between "
            f"{left.stage.key} and {right.stage.key}"
        )


def _validate_group_join(group: Artifact, join: Artifact) -> None:
    for field in (
        "upstream_transcript",
        "input_commitment",
        "parameters_commitment",
        "output_commitment",
        "transcript_commitment",
    ):
        _require_equal(group, join, field, "group/join mismatch")
    if group.stage.key == "attention":
        _require_equal(group, join, "hints_commitment", "group/join mismatch")


def _validate_transition(upstream: Artifact, downstream: Artifact) -> None:
    if downstream.input != upstream.output:
        raise ValidationError(
            f"layer {downstream.layer}: {downstream.stage.key} input does not match "
            f"{upstream.stage.key} output"
        )
    if downstream.upstream != upstream.transcript:
        raise ValidationError(
            f"layer {downstream.layer}: {downstream.stage.key} upstream transcript "
            f"does not match {upstream.stage.key} transcript"
        )


def validate_layer_prefix(
    layer: int,
    artifacts: dict[tuple[int, str], Artifact],
    previous_block: Artifact | None,
) -> None:
    def get(key: str) -> Artifact | None:
        return artifacts.get((layer, key))

    attention = get("attention")
    if attention is None:
        return
    if layer == 0:
        if attention.upstream is not None:
            raise ValidationError("layer 0 attention must not declare an upstream transcript")
    else:
        if previous_block is None:
            raise ValidationError(f"layer {layer} is missing the previous block proof")
        _validate_transition(previous_block, attention)

    attention_join = get("attention-join")
    if attention_join is None:
        return
    _validate_group_join(attention, attention_join)

    c_proj = get("c-proj")
    if c_proj is None:
        return
    _validate_transition(attention_join, c_proj)
    c_proj_join = get("c-proj-join")
    if c_proj_join is None:
        return
    _validate_group_join(c_proj, c_proj_join)

    ln2 = get("ln2")
    if ln2 is None:
        return
    _validate_transition(c_proj_join, ln2)

    expansion = get("mlp-expansion")
    if expansion is None:
        return
    _validate_transition(ln2, expansion)
    expansion_join = get("mlp-expansion-join")
    if expansion_join is None:
        return
    _validate_group_join(expansion, expansion_join)

    projection = get("mlp-projection")
    if projection is None:
        return
    _validate_transition(expansion_join, projection)
    projection_join = get("mlp-projection-join")
    if projection_join is None:
        return
    _validate_group_join(projection, projection_join)


def load_and_validate_targets(
    output_root: Path, targets: Iterable[tuple[int, Stage]]
) -> dict[tuple[int, str], Artifact]:
    artifacts: dict[tuple[int, str], Artifact] = {}
    for layer, stage in targets:
        artifact = load_artifact(output_root, layer, stage)
        artifacts[(layer, stage.key)] = artifact
        previous = artifacts.get((layer - 1, "mlp-projection-join"))
        validate_layer_prefix(layer, artifacts, previous)
    return artifacts


def validate_data_fixture(data_dir: Path) -> None:
    metadata_path = data_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    expected_metadata = {
        "layers": NUM_LAYERS,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "linear_size": EXPANSION_SIZE,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValidationError(
                f"{metadata_path}: expected {key}={expected}, found {metadata.get(key)!r}"
            )

    expected_files = {
        data_dir / "hidden_state.bf16.bin": SEQUENCE_LENGTH * HIDDEN_SIZE * BF16_BYTES,
        data_dir / "pytorch_reference_output.bf16.bin": BLOCK_OUTPUT_BYTES,
    }
    for layer in range(NUM_LAYERS):
        directory = data_dir / f"layer-{layer:02d}"
        values = {
            "ln_1_weight.bf16.bin": HIDDEN_SIZE,
            "ln_1_bias.bf16.bin": HIDDEN_SIZE,
            "attention_qkv_weight.bf16.bin": HIDDEN_SIZE * 3 * HIDDEN_SIZE,
            "attention_projection_weight.bf16.bin": HIDDEN_SIZE * HIDDEN_SIZE,
            "ln_2_weight.bf16.bin": HIDDEN_SIZE,
            "ln_2_bias.bf16.bin": HIDDEN_SIZE,
            "mlp_expansion_weight.bf16.bin": HIDDEN_SIZE * EXPANSION_SIZE,
            "mlp_projection_weight.bf16.bin": EXPANSION_SIZE * HIDDEN_SIZE,
            "attention_max_hints.bf16.bin": SEQUENCE_LENGTH * NUM_HEADS,
        }
        expected_files.update(
            {directory / name: count * BF16_BYTES for name, count in values.items()}
        )
    for path, expected_size in expected_files.items():
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValidationError(f"missing BF16 fixture file {path}: {error}") from error
        if size != expected_size:
            raise ValidationError(
                f"{path}: found {size} bytes, expected {expected_size}"
            )


def stage_command(
    stage: Stage,
    layer: int,
    output_root: Path,
    data_dir: Path,
    bin_dir: Path,
) -> list[str]:
    command = [str(bin_dir / stage.binary), "--prove", "--layer", str(layer)]
    output = stage_dir(output_root, layer, stage)
    directories = {item.key: stage_dir(output_root, layer, item) for item in STAGES}
    if stage.key == "attention":
        command.extend(["--all-heads", "--data-dir", str(data_dir)])
        if layer > 0:
            previous = layer_dir(output_root, layer - 1)
            command.extend(
                [
                    "--previous-block-dir",
                    str(previous / "mlp-projection"),
                    "--previous-block-join-dir",
                    str(previous / "mlp-projection-join"),
                ]
            )
    elif stage.key == "attention-join":
        command.extend(["--leaf-dir", str(directories["attention"])])
    elif stage.key == "c-proj":
        command.extend(
            [
                "--all-tiles",
                "--attention-dir",
                str(directories["attention"]),
                "--join-dir",
                str(directories["attention-join"]),
                "--data-dir",
                str(data_dir),
            ]
        )
    elif stage.key == "c-proj-join":
        command.extend(["--tile-dir", str(directories["c-proj"])])
    elif stage.key == "ln2":
        command.extend(
            [
                "--c-proj-dir",
                str(directories["c-proj"]),
                "--join-dir",
                str(directories["c-proj-join"]),
                "--data-dir",
                str(data_dir),
            ]
        )
    elif stage.key == "mlp-expansion":
        command.extend(
            [
                "--all-tiles",
                "--ln2-dir",
                str(directories["ln2"]),
                "--data-dir",
                str(data_dir),
            ]
        )
    elif stage.key == "mlp-expansion-join":
        command.extend(["--tile-dir", str(directories["mlp-expansion"])])
    elif stage.key == "mlp-projection":
        command.extend(
            [
                "--all-tiles",
                "--expansion-dir",
                str(directories["mlp-expansion"]),
                "--join-dir",
                str(directories["mlp-expansion-join"]),
                "--data-dir",
                str(data_dir),
            ]
        )
    elif stage.key == "mlp-projection-join":
        command.extend(["--tile-dir", str(directories["mlp-projection"])])
    else:
        raise AssertionError(f"no command builder for {stage.key}")
    command.extend(["--output-dir", str(output)])
    return command


def build_binaries(repo_root: Path) -> None:
    command = ["cargo", "build", "-p", "sp1-recursion-compiler", "--release"]
    for stage in STAGES:
        command.extend(["--example", stage.binary])
    print(f"[runner] build: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def require_binaries(bin_dir: Path) -> None:
    missing = [bin_dir / stage.binary for stage in STAGES if not (bin_dir / stage.binary).is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise ValidationError(
            f"missing stage binaries:\n  {formatted}\nrerun with --build or set --bin-dir"
        )


def run_stage_command(
    command: list[str], repo_root: Path, log_path: Path, threads: int
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    started = time.perf_counter()
    rendered = shlex.join(command)
    print(f"[runner] run: {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{timestamp()}] {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def progress_summary(artifacts: dict[tuple[int, str], Artifact]) -> dict[str, Any]:
    ordered = sorted(artifacts.values(), key=lambda item: (item.layer, STAGES.index(item.stage)))
    proof_instances = sum(item.stage.proof_instances for item in ordered)
    proof_bytes = sum(item.proof_bytes for item in ordered)
    completed_layers = sum(
        (layer, "mlp-projection-join") in artifacts for layer in range(NUM_LAYERS)
    )
    last = ordered[-1] if ordered else None
    final_blocks = [
        item for item in ordered if item.stage.key == "mlp-projection-join"
    ]
    final_block = final_blocks[-1] if final_blocks else None
    return {
        "completed_artifact_stages": len(ordered),
        "completed_layers": completed_layers,
        "proof_instances": proof_instances,
        "full_proof_instances": FULL_PROOF_INSTANCES,
        "proof_bytes": proof_bytes,
        "last_stage": (
            {"layer": last.layer, "stage": last.stage.key} if last is not None else None
        ),
        "latest_block_output": (
            {
                "layer": final_block.layer,
                "output_commitment": final_block.output,
                "transcript_commitment": final_block.transcript,
            }
            if final_block is not None
            else None
        ),
    }


class StateRecorder:
    def __init__(
        self,
        output_root: Path,
        data_dir: Path,
        argv: list[str],
        resume: bool,
        artifacts: dict[tuple[int, str], Artifact],
    ) -> None:
        self.path = output_root / RUN_STATE_NAME
        if resume and self.path.is_file():
            self.state = _read_json(self.path)
            if self.state.get("architecture") != "zkgpt-like-gpt2-bf16-12x30-shape-aware":
                raise ValidationError(f"{self.path}: incompatible run-state architecture")
        else:
            self.state = {
                "version": 2,
                "architecture": "zkgpt-like-gpt2-bf16-12x30-shape-aware",
                "layers": NUM_LAYERS,
                "proof_instances_per_block": PROOF_INSTANCES_PER_BLOCK,
                "full_proof_instances": FULL_PROOF_INSTANCES,
                "created_at": timestamp(),
                "invocations": [],
            }
        self.invocation_started = time.perf_counter()
        self.invocation: dict[str, Any] = {
            "started_at": timestamp(),
            "status": "running",
            "command": shlex.join(argv),
            "data_dir": str(data_dir),
            "events": [],
        }
        invocations = self.state.setdefault("invocations", [])
        if not isinstance(invocations, list):
            raise ValidationError(f"{self.path}: invocations must be a list")
        invocations.append(self.invocation)
        self.write(artifacts)

    def event(
        self,
        artifacts: dict[tuple[int, str], Artifact],
        layer: int,
        stage: Stage,
        status: str,
        wall_seconds: float = 0.0,
        error: str | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "time": timestamp(),
            "layer": layer,
            "stage": stage.key,
            "status": status,
            "wall_seconds": round(wall_seconds, 6),
        }
        if error is not None:
            event["error"] = error
        self.invocation["events"].append(event)
        self.write(artifacts)

    def finish(
        self, artifacts: dict[tuple[int, str], Artifact], status: str, error: str | None = None
    ) -> None:
        self.invocation["status"] = status
        self.invocation["finished_at"] = timestamp()
        self.invocation["wall_seconds"] = round(
            time.perf_counter() - self.invocation_started, 6
        )
        self.invocation.pop("wall_seconds_so_far", None)
        if error is not None:
            self.invocation["error"] = error
        self.write(artifacts)

    def write(self, artifacts: dict[tuple[int, str], Artifact]) -> None:
        self.state["updated_at"] = timestamp()
        self.state["progress"] = progress_summary(artifacts)
        if self.invocation.get("status") == "running":
            self.invocation["wall_seconds_so_far"] = round(
                time.perf_counter() - self.invocation_started, 6
            )
        invocations = self.state["invocations"]
        self.state["timing"] = {
            "completed_invocation_wall_seconds": round(
                sum(
                    float(invocation.get("wall_seconds", 0.0))
                    for invocation in invocations
                    if invocation.get("status") != "running"
                ),
                6,
            ),
            "produced_stage_wall_seconds": round(
                sum(
                    float(event.get("wall_seconds", 0.0))
                    for invocation in invocations
                    for event in invocation.get("events", [])
                    if event.get("status") == "produced"
                ),
                6,
            ),
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)


def check_manifest_prefix(output_root: Path, targets: list[tuple[int, Stage]]) -> None:
    missing_seen = False
    first_missing: Path | None = None
    for layer, stage in targets:
        path = manifest_path(output_root, layer, stage)
        if path.is_file():
            if missing_seen:
                raise ValidationError(
                    f"artifact gap: {path} exists after missing earlier manifest {first_missing}"
                )
        else:
            if not missing_seen:
                first_missing = path
            missing_seen = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Reuse the bounded zkGPT-like BF16 Block pipeline across all 12 GPT-2 "
            "layers, validating every proof artifact and commitment link."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prove", action="store_true", help="generate the selected proof prefix")
    mode.add_argument("--check-only", action="store_true", help="validate existing artifacts only")
    mode.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(repo_root))
    parser.add_argument("--bin-dir", type=Path, default=default_bin_dir(repo_root))
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=NUM_LAYERS - 1)
    parser.add_argument(
        "--end-stage",
        choices=[stage.key for stage in STAGES],
        default=STAGES[-1].key,
        help="stop after this stage of --end-layer",
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true", help="validate and skip completed stages")
    parser.add_argument(
        "--build", action="store_true", help="build all nine release examples first"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.start_layer <= args.end_layer < NUM_LAYERS:
        parser.error(f"layers must satisfy 0 <= start <= end < {NUM_LAYERS}")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.resume and not args.prove:
        parser.error("--resume is only valid with --prove")
    if args.build and not args.prove:
        parser.error("--build is only valid with --prove")
    if args.start_layer > 0 and args.prove and not args.resume:
        parser.error("--start-layer > 0 requires --resume so earlier block proofs are validated")
    return args


def run(args: argparse.Namespace, argv: list[str]) -> int:
    repo_root = repository_root()
    output_root = args.output_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    bin_dir = args.bin_dir.expanduser().resolve()
    targets = target_stages(args.end_layer, args.end_stage)
    selected = selected_stages(args.start_layer, args.end_layer, args.end_stage)

    if args.dry_run:
        selected_proofs = sum(stage.proof_instances for _, stage in selected)
        print(
            f"[runner] plan: layers={args.start_layer}..{args.end_layer} "
            f"end_stage={args.end_stage} commands={len(selected)} "
            f"proofs={selected_proofs}/{FULL_PROOF_INSTANCES}"
        )
        for layer, stage in selected:
            print(shlex.join(stage_command(stage, layer, output_root, data_dir, bin_dir)))
        return 0

    if args.check_only:
        artifacts = load_and_validate_targets(output_root, targets)
        progress = progress_summary(artifacts)
        print(
            f"[runner] valid proof chain: layers=0..{args.end_layer} "
            f"end_stage={args.end_stage} proofs={progress['proof_instances']}/"
            f"{FULL_PROOF_INSTANCES} bytes={progress['proof_bytes']}"
        )
        latest = progress["latest_block_output"]
        if latest is not None:
            print(
                f"[runner] latest complete block={latest['layer']} "
                f"output={latest['output_commitment']} transcript={latest['transcript_commitment']}"
            )
        return 0

    validate_data_fixture(data_dir)
    if args.build:
        build_binaries(repo_root)
    require_binaries(bin_dir)

    if output_root.exists() and not args.resume:
        try:
            next(output_root.iterdir())
        except StopIteration:
            pass
        else:
            raise ValidationError(
                f"output root is not empty: {output_root}; use --resume or a new directory"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        check_manifest_prefix(output_root, targets)

    prerequisite_targets = [item for item in targets if item[0] < args.start_layer]
    initial_targets = list(prerequisite_targets)
    if args.resume:
        initial_targets.extend(
            item
            for item in selected
            if manifest_path(output_root, item[0], item[1]).is_file()
        )
    artifacts = load_and_validate_targets(output_root, initial_targets)
    recorder = StateRecorder(output_root, data_dir, argv, args.resume, artifacts)
    try:
        for layer, stage in selected:
            path = manifest_path(output_root, layer, stage)
            if path.is_file():
                if not args.resume:
                    raise ValidationError(f"existing manifest requires --resume: {path}")
                artifact = load_artifact(output_root, layer, stage)
                artifacts[(layer, stage.key)] = artifact
                previous = artifacts.get((layer - 1, "mlp-projection-join"))
                validate_layer_prefix(layer, artifacts, previous)
                print(f"[runner] skip valid layer={layer} stage={stage.key}")
                recorder.event(artifacts, layer, stage, "skipped")
                continue

            output = stage_dir(output_root, layer, stage)
            output.mkdir(parents=True, exist_ok=True)
            command = stage_command(stage, layer, output_root, data_dir, bin_dir)
            log_path = output_root / "logs" / f"layer-{layer:02d}.{stage.key}.log"
            recorder.event(artifacts, layer, stage, "running")
            try:
                elapsed = run_stage_command(command, repo_root, log_path, args.threads)
                artifact = load_artifact(output_root, layer, stage)
                artifacts[(layer, stage.key)] = artifact
                previous = artifacts.get((layer - 1, "mlp-projection-join"))
                validate_layer_prefix(layer, artifacts, previous)
            except Exception as error:
                recorder.event(artifacts, layer, stage, "failed", error=str(error))
                raise
            recorder.event(artifacts, layer, stage, "produced", elapsed)
            print(
                f"[runner] valid layer={layer} stage={stage.key} "
                f"proofs={stage.proof_instances} elapsed={elapsed:.3f}s"
            )
        recorder.finish(artifacts, "completed")
    except KeyboardInterrupt:
        recorder.finish(artifacts, "interrupted", "keyboard interrupt")
        raise
    except Exception as error:
        recorder.finish(artifacts, "failed", str(error))
        raise

    progress = progress_summary(artifacts)
    print(
        f"[runner] completed selected prefix: proofs={progress['proof_instances']}/"
        f"{FULL_PROOF_INSTANCES} proof_bytes={progress['proof_bytes']}"
    )
    print(f"[runner] run state: {output_root / RUN_STATE_NAME}")
    return 0


def main(argv: list[str] | None = None) -> int:
    actual = sys.argv[1:] if argv is None else argv
    try:
        args = parse_args(actual)
        return run(args, [sys.argv[0], *actual])
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"error: command exited with status {error.returncode}: {shlex.join(error.cmd)}",
            file=sys.stderr,
        )
        return error.returncode or 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
