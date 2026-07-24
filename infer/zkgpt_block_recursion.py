#!/usr/bin/env python3
"""Generate 12 trace-sharded GPT-2 Block proofs and reduce them to one proof.

Each Block is first executed once, split by event/trace area, and compressed into
one recursion proof. Blocks remain sequential because Block N consumes the
host-carried private output of Block N-1. Once all 12 Block proofs exist, they are
reduced as a binary tree. With 12 leaves the tree contains 11 joins:

    12 -> 6 -> 3 -> 2 -> 1

An unpaired node is carried to the next level without creating a redundant
identity proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUM_BLOCKS = 12
SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 768
NUM_HEADS = 12
LINEAR_SIZE = 2304
BLOCK_STAGE = "zkgpt_block_shard_recursion"
BLOCK_PROTOCOL_VERSION = 1
EVENT_SHARD_STAGE = "zkgpt_event_shards"
RECURSION_STAGE = "zkgpt_block_recursion"
RUN_MANIFEST = "zkgpt_block_recursion.run.json"


class ValidationError(RuntimeError):
    """A proof artifact or recursion-range invariant is invalid."""


@dataclass(frozen=True)
class Node:
    manifest: Path
    kind: str
    start_block: int
    end_block: int
    transcript_commitment: str | None


@dataclass(frozen=True)
class Join:
    level: int
    left: Node
    right: Node
    output_dir: Path

    @property
    def manifest(self) -> Path:
        stem = (
            f"zkgpt_recursion_b{self.left.start_block:02d}"
            f"_b{self.right.end_block:02d}"
        )
        return self.output_dir / f"{stem}.manifest.json"


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_dir(root: Path) -> Path:
    return (
        root.parent
        / "sp1-models"
        / "gpt2-bf16"
        / "recursion"
        / "zkgpt-like-12x30-real-bf16"
    )


def default_bin_dir(root: Path) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    return target / "release" / "examples"


def block_dir(output_root: Path, block: int) -> Path:
    return output_root / "blocks" / f"block-{block:02d}"


def block_manifest(output_root: Path, block: int) -> Path:
    return (
        block_dir(output_root, block)
        / f"zkgpt_block_{block:02d}_shard_recursion.manifest.json"
    )


def event_shard_manifest(output_root: Path, block: int) -> Path:
    return block_dir(output_root, block) / "zkgpt_event_shards.manifest.json"


def recursion_dir(output_root: Path, level: int, start: int, end: int) -> Path:
    return output_root / "recursion" / f"level-{level:02d}" / f"b{start:02d}-b{end:02d}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return value


def require_file(directory: Path, value: Any, label: str, manifest: Path) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValidationError(f"{manifest}: invalid {label} file name {value!r}")
    path = directory / value
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValidationError(f"{manifest}: missing {label} {path}: {error}") from error
    if size <= 0:
        raise ValidationError(f"{manifest}: empty {label} {path}")
    return path


def require_shape(data: dict[str, Any], path: Path) -> None:
    expected = {
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "linear_size": LINEAR_SIZE,
        "shards": 1,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValidationError(
                f"{path}: expected {key}={value}, found {data.get(key)!r}"
            )


def validate_event_shard_manifest(path: Path, expected_block: int) -> dict[str, Any]:
    data = read_json(path)
    if data.get("stage") != EVENT_SHARD_STAGE or data.get("block") != expected_block:
        raise ValidationError(
            f"{path}: not the expected Block {expected_block} event-shard manifest"
        )
    for key, value in {
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "linear_size": LINEAR_SIZE,
    }.items():
        if data.get(key) != value:
            raise ValidationError(
                f"{path}: expected {key}={value}, found {data.get(key)!r}"
            )
    if data.get("lookup_protocol") != "shared_challenge_batch":
        raise ValidationError(f"{path}: unsupported lookup protocol")
    if data.get("lookup_batch_closed") is not True:
        raise ValidationError(f"{path}: lookup batch is not closed")
    if (
        data.get("transcript_binding")
        != "ordered_verifying_keys_main_trace_commitments_real_heights"
    ):
        raise ValidationError(f"{path}: unsupported trace transcript binding")
    require_digest(data, "block_transcript_commitment", path)
    require_file(path.parent, data.get("private_output_file"), "private output", path)
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or data.get("shards") != len(artifacts) or not artifacts:
        raise ValidationError(f"{path}: invalid event-shard artifact count")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or artifact.get("index") != index:
            raise ValidationError(f"{path}: event shard {index} is missing or unordered")
        require_file(path.parent, artifact.get("proof_file"), f"shard {index} proof", path)
        require_file(
            path.parent,
            artifact.get("verifying_key_file"),
            f"shard {index} verifying key",
            path,
        )
    return data


def require_digest(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"{path}: missing {key}")
    limbs = value.split(":")
    if len(limbs) != 8 or any(
        len(limb) != 8 or any(character not in "0123456789abcdefABCDEF" for character in limb)
        for limb in limbs
    ):
        raise ValidationError(f"{path}: invalid digest in {key}")
    return value.upper()


def load_block_node(path: Path, expected_block: int) -> Node:
    data = read_json(path)
    if data.get("stage") != BLOCK_STAGE or data.get("block") != expected_block:
        raise ValidationError(f"{path}: not the expected Block {expected_block} manifest")
    require_shape(data, path)
    directory = path.parent
    require_file(directory, data.get("proof_file"), "proof", path)
    require_file(directory, data.get("verifying_key_file"), "verifying key", path)
    require_file(directory, data.get("private_output_file"), "private output", path)
    if data.get("version") != BLOCK_PROTOCOL_VERSION:
        raise ValidationError(
            f"{path}: expected Block shard-recursion protocol version "
            f"{BLOCK_PROTOCOL_VERSION}"
        )
    source_event_shards = data.get("source_event_shards")
    if not isinstance(source_event_shards, int) or source_event_shards <= 0:
        raise ValidationError(f"{path}: invalid source_event_shards")
    require_digest(data, "trace_transcript_commitment", path)
    require_digest(data, "verifying_key_commitment", path)
    return Node(
        manifest=path,
        kind="recursion",
        start_block=expected_block,
        end_block=expected_block + 1,
        transcript_commitment=require_digest(data, "transcript_commitment", path),
    )


def load_recursion_node(path: Path, expected_start: int, expected_end: int) -> Node:
    data = read_json(path)
    if data.get("stage") != RECURSION_STAGE:
        raise ValidationError(f"{path}: not a Block recursion manifest")
    if data.get("version") != 2:
        raise ValidationError(f"{path}: expected recursion protocol version 2")
    require_shape(data, path)
    if data.get("start_block") != expected_start or data.get("end_block") != expected_end:
        raise ValidationError(
            f"{path}: expected range {expected_start}..{expected_end}, found "
            f"{data.get('start_block')}..{data.get('end_block')}"
        )
    directory = path.parent
    require_file(directory, data.get("proof_file"), "proof", path)
    require_file(directory, data.get("verifying_key_file"), "verifying key", path)
    return Node(
        manifest=path,
        kind="recursion",
        start_block=expected_start,
        end_block=expected_end,
        transcript_commitment=require_digest(data, "transcript_commitment", path),
    )


def validate_adjacent(left: Node, right: Node) -> None:
    if left.end_block != right.start_block:
        raise ValidationError(
            f"non-adjacent ranges {left.start_block}..{left.end_block} and "
            f"{right.start_block}..{right.end_block}"
        )


def load_blocks(output_root: Path) -> list[Node]:
    nodes: list[Node] = []
    for block in range(NUM_BLOCKS):
        node = load_block_node(block_manifest(output_root, block), block)
        if nodes:
            validate_adjacent(nodes[-1], node)
        nodes.append(node)
    return nodes


def block_command(
    output_root: Path, data_dir: Path, bin_dir: Path, block: int
) -> list[str]:
    command = [
        str(bin_dir / "zkgpt_like"),
        "--prove-shards",
        "--allow-large-build",
        "--block",
        str(block),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(block_dir(output_root, block)),
    ]
    if block > 0:
        command.extend(
            ["--previous-block-dir", str(block_dir(output_root, block - 1))]
        )
    return command


def block_recursion_command(
    output_root: Path, bin_dir: Path, block: int
) -> list[str]:
    return [
        str(bin_dir / "zkgpt_block_recursion"),
        "--prove",
        "--event-shard-manifest",
        str(event_shard_manifest(output_root, block)),
        "--output-dir",
        str(block_dir(output_root, block)),
    ]


def join_command(join: Join, bin_dir: Path) -> list[str]:
    return [
        str(bin_dir / "zkgpt_block_recursion"),
        "--prove",
        "--left-manifest",
        str(join.left.manifest),
        "--right-manifest",
        str(join.right.manifest),
        "--output-dir",
        str(join.output_dir),
    ]


def build_binaries(root: Path) -> None:
    commands = (
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "sp1-recursion-compiler",
            "--example",
            "zkgpt_like",
        ],
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "sp1-recursion-circuit",
            "--example",
            "zkgpt_block_recursion",
        ],
    )
    for command in commands:
        print(f"[runner] build: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=root, check=True)


def run_command(
    command: list[str], root: Path, log_path: Path, threads: int
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    started = time.perf_counter()
    print(f"[runner] run: {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status != 0:
        raise subprocess.CalledProcessError(status, command)
    return time.perf_counter() - started


def reduce_nodes(
    output_root: Path,
    bin_dir: Path,
    root: Path,
    nodes: list[Node],
    threads: int,
    resume: bool,
    dry_run: bool,
    check_only: bool,
    events: list[dict[str, Any]],
) -> Node:
    level = 0
    while len(nodes) > 1:
        next_level: list[Node] = []
        index = 0
        while index + 1 < len(nodes):
            left, right = nodes[index], nodes[index + 1]
            validate_adjacent(left, right)
            output_dir = recursion_dir(
                output_root, level, left.start_block, right.end_block
            )
            join = Join(level, left, right, output_dir)
            path = join.manifest
            if resume and path.is_file():
                node = load_recursion_node(
                    path, left.start_block, right.end_block
                )
                print(
                    f"[runner] skip recursion level={level} "
                    f"range={node.start_block}..{node.end_block}"
                )
                events.append(
                    {
                        "kind": "recursion",
                        "level": level,
                        "start_block": node.start_block,
                        "end_block": node.end_block,
                        "status": "skipped",
                    }
                )
            elif check_only:
                raise ValidationError(f"missing recursion manifest: {path}")
            elif dry_run:
                print(shlex.join(join_command(join, bin_dir)))
                node = Node(
                    manifest=path,
                    kind="recursion",
                    start_block=left.start_block,
                    end_block=right.end_block,
                    transcript_commitment="<generated>",
                )
            else:
                if path.exists():
                    raise ValidationError(
                        f"{path} already exists; pass --resume or use a new output root"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                elapsed = run_command(
                    join_command(join, bin_dir),
                    root,
                    output_root
                    / "logs"
                    / (
                        f"recursion-l{level:02d}-"
                        f"b{left.start_block:02d}-b{right.end_block:02d}.log"
                    ),
                    threads,
                )
                node = load_recursion_node(
                    path, left.start_block, right.end_block
                )
                events.append(
                    {
                        "kind": "recursion",
                        "level": level,
                        "start_block": node.start_block,
                        "end_block": node.end_block,
                        "status": "produced",
                        "wall_seconds": round(elapsed, 6),
                    }
                )
            next_level.append(node)
            index += 2
        if index < len(nodes):
            carried = nodes[index]
            print(
                f"[runner] carry recursion level={level} "
                f"range={carried.start_block}..{carried.end_block}"
            )
            next_level.append(carried)
        nodes = next_level
        level += 1
    return nodes[0]


def write_run_manifest(
    output_root: Path,
    data_dir: Path,
    final: Node,
    events: list[dict[str, Any]],
    started: float,
) -> None:
    final_data = read_json(final.manifest)
    value = {
        "version": 2,
        "stage": "zkgpt_full_block_recursion",
        "blocks": NUM_BLOCKS,
        "block_proofs": NUM_BLOCKS,
        "recursion_proofs": NUM_BLOCKS - 1,
        "event_leaf_proofs": sum(
            read_json(block_manifest(output_root, block))["source_event_shards"]
            for block in range(NUM_BLOCKS)
        ),
        "total_persisted_block_and_join_proofs": 2 * NUM_BLOCKS - 1,
        "data_dir": str(data_dir),
        "final_manifest": str(final.manifest),
        "final_transcript_commitment": require_digest(
            final_data, "transcript_commitment", final.manifest
        ),
        "final_verifying_key_commitment": require_digest(
            final_data, "verifying_key_commitment", final.manifest
        ),
        "final_proof_file": str(
            require_file(
                final.manifest.parent,
                final_data.get("proof_file"),
                "final proof",
                final.manifest,
            )
        ),
        "final_verifying_key_file": str(
            require_file(
                final.manifest.parent,
                final_data.get("verifying_key_file"),
                "final verifying key",
                final.manifest,
            )
        ),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "events": events,
    }
    path = output_root / RUN_MANIFEST
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(f"[runner] final recursion proof manifest: {final.manifest}")
    print(f"[runner] run manifest: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate one proof per GPT-2 Block and recursively reduce all "
            "12 Block proofs to one final proof."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prove", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(root))
    parser.add_argument("--bin-dir", type=Path, default=default_bin_dir(root))
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.resume and not args.prove:
        parser.error("--resume is only valid with --prove")
    if args.build and not args.prove:
        parser.error("--build is only valid with --prove")
    return args


def run(args: argparse.Namespace) -> int:
    root = repository_root()
    output_root = args.output_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    bin_dir = args.bin_dir.expanduser().resolve()
    started = time.perf_counter()
    events: list[dict[str, Any]] = []

    if args.check_only:
        verifier = bin_dir / "zkgpt_block_recursion"
        if not verifier.is_file():
            raise ValidationError(
                f"missing {verifier}; build the recursion verifier on the server first"
            )
        blocks = load_blocks(output_root)
        final = reduce_nodes(
            output_root,
            bin_dir,
            root,
            blocks,
            args.threads,
            True,
            False,
            True,
            events,
        )
        if final.start_block != 0 or final.end_block != NUM_BLOCKS:
            raise ValidationError("final recursion proof does not cover all Blocks")
        subprocess.run(
            [str(verifier), "--check", "--verify-manifest", str(final.manifest)],
            cwd=root,
            check=True,
        )
        final_data = read_json(final.manifest)
        print(
            f"[runner] valid final proof range=0..{NUM_BLOCKS} "
            f"transcript={require_digest(final_data, 'transcript_commitment', final.manifest)}"
        )
        return 0

    if args.build and not args.dry_run:
        build_binaries(root)
    if args.prove:
        for binary in ("zkgpt_like", "zkgpt_block_recursion"):
            if not (bin_dir / binary).is_file():
                raise ValidationError(
                    f"missing {bin_dir / binary}; pass --build or build it on the server"
                )

    if output_root.exists() and not args.resume and not args.dry_run:
        try:
            next(output_root.iterdir())
        except StopIteration:
            pass
        else:
            raise ValidationError(
                f"output root is not empty: {output_root}; use --resume or a new directory"
            )
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    blocks: list[Node] = []
    for block in range(NUM_BLOCKS):
        path = block_manifest(output_root, block)
        if args.resume and path.is_file():
            node = load_block_node(path, block)
            if blocks:
                validate_adjacent(blocks[-1], node)
            print(f"[runner] skip valid Block {block}")
            events.append({"kind": "block", "block": block, "status": "skipped"})
        elif args.dry_run:
            print(shlex.join(block_command(output_root, data_dir, bin_dir, block)))
            print(shlex.join(block_recursion_command(output_root, bin_dir, block)))
            node = Node(
                manifest=path,
                kind="recursion",
                start_block=block,
                end_block=block + 1,
                transcript_commitment=None,
            )
        else:
            output = block_dir(output_root, block)
            output.mkdir(parents=True, exist_ok=True)
            shard_path = event_shard_manifest(output_root, block)
            if args.resume and shard_path.is_file():
                validate_event_shard_manifest(shard_path, block)
                print(f"[runner] skip valid event leaves for Block {block}")
            else:
                leaf_elapsed = run_command(
                    block_command(output_root, data_dir, bin_dir, block),
                    root,
                    output_root / "logs" / f"block-{block:02d}-event-leaves.log",
                    args.threads,
                )
                events.append(
                    {
                        "kind": "event_leaves",
                        "block": block,
                        "status": "produced",
                        "wall_seconds": round(leaf_elapsed, 6),
                    }
                )
                validate_event_shard_manifest(shard_path, block)
            elapsed = run_command(
                block_recursion_command(output_root, bin_dir, block),
                root,
                output_root / "logs" / f"block-{block:02d}-recursion.log",
                args.threads,
            )
            node = load_block_node(path, block)
            if blocks:
                validate_adjacent(blocks[-1], node)
            events.append(
                {
                    "kind": "block",
                    "block": block,
                    "status": "produced",
                    "wall_seconds": round(elapsed, 6),
                }
            )
        blocks.append(node)

    if args.dry_run:
        reduce_nodes(
            output_root,
            bin_dir,
            root,
            blocks,
            args.threads,
            False,
            True,
            False,
            events,
        )
        print(
            "[runner] recursion plan: 12 -> 6 -> 3 -> 2 -> 1 "
            "(per-Block event leaves -> 1 Block proof, then 11 joins)"
        )
        return 0

    final = reduce_nodes(
        output_root,
        bin_dir,
        root,
        blocks,
        args.threads,
        args.resume,
        False,
        False,
        events,
    )
    if final.start_block != 0 or final.end_block != NUM_BLOCKS:
        raise ValidationError("final recursion proof does not cover all 12 Blocks")
    subprocess.run(
        [
            str(bin_dir / "zkgpt_block_recursion"),
            "--check",
            "--verify-manifest",
            str(final.manifest),
        ],
        cwd=root,
        check=True,
    )
    write_run_manifest(output_root, data_dir, final, events, started)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"error: command failed ({error.returncode}): {shlex.join(error.cmd)}",
            file=sys.stderr,
        )
        return error.returncode or 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
