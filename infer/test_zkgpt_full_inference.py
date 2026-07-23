from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zkgpt_full_inference import (
    FULL_PROOF_INSTANCES,
    HIDDEN_SIZE,
    NUM_HEADS,
    PROOF_INSTANCES_PER_BLOCK,
    SEQUENCE_LENGTH,
    STAGES,
    ValidationError,
    check_manifest_prefix,
    load_and_validate_targets,
    manifest_path,
    progress_summary,
    stage_command,
    stage_dir,
    target_stages,
)


def digest(seed: int) -> str:
    return ":".join(f"{seed + index:08X}" for index in range(8))


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def write_artifact(
    root: Path,
    layer: int,
    stage_index: int,
    upstream: str | None,
    input_commitment: str,
    output_commitment: str,
    transcript_commitment: str,
) -> None:
    stage = STAGES[stage_index]
    directory = stage_dir(root, layer, stage)
    directory.mkdir(parents=True, exist_ok=True)
    verifying_key_file = f"{stage.key}.vk.bin"
    write_bytes(directory / verifying_key_file, 5)
    parameter_stage = {1: 0, 3: 2, 6: 5, 8: 7}.get(stage_index, stage_index)
    data: dict[str, object] = {
        "version": 1,
        "stage": stage.manifest_stage,
        "layer": layer,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "input_commitment": input_commitment,
        "parameters_commitment": digest(10_000 + parameter_stage),
        "output_commitment": output_commitment,
        "transcript_commitment": transcript_commitment,
        "verifying_key_file": verifying_key_file,
        "verifying_key_bytes": 5,
    }
    if upstream is not None:
        data["upstream_transcript"] = upstream
    if stage.key.startswith("attention"):
        data["num_heads"] = NUM_HEADS
        data["hints_commitment"] = digest(20_000 + layer)
    if stage.key.startswith("c-proj"):
        data["num_tiles"] = 3
    if stage.key.startswith("mlp-"):
        data["expansion_size"] = 2304
    if stage.collection == "tiles":
        data["num_tiles"] = stage.child_count

    if stage.collection is not None:
        children = []
        for index in range(stage.child_count):
            proof_file = f"{stage.key}-{index:02d}.proof.bin"
            write_bytes(directory / proof_file, 7)
            write_bytes(directory / stage.commitments(layer, index), 9)
            children.append(
                {
                    stage.index_field: index,
                    "proof_file": proof_file,
                    "proof_bytes": 7,
                }
            )
        data[stage.collection] = children
    elif stage.single_proof:
        proof_file = f"{stage.key}.proof.bin"
        write_bytes(directory / proof_file, 11)
        data["proof_file"] = proof_file
        data["proof_bytes"] = 11

    if stage.private_output_bytes is not None:
        private_output = f"{stage.key}.private.bf16.bin"
        write_bytes(directory / private_output, stage.private_output_bytes)
        data["private_output_file"] = private_output
        data["output_values"] = stage.private_output_bytes // 2

    manifest_path(root, layer, stage).write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def write_layer(
    root: Path,
    layer: int,
    previous_output: str | None,
    previous_transcript: str | None,
) -> tuple[str, str]:
    if layer == 0:
        current_input = digest(100)
        current_upstream = None
    else:
        assert previous_output is not None and previous_transcript is not None
        current_input = previous_output
        current_upstream = previous_transcript

    seed = 1_000 + layer * 100
    attention_output = digest(seed)
    attention_transcript = digest(seed + 10)
    write_artifact(
        root,
        layer,
        0,
        current_upstream,
        current_input,
        attention_output,
        attention_transcript,
    )
    write_artifact(
        root,
        layer,
        1,
        current_upstream,
        current_input,
        attention_output,
        attention_transcript,
    )

    c_proj_output = digest(seed + 20)
    c_proj_transcript = digest(seed + 30)
    for stage_index in (2, 3):
        write_artifact(
            root,
            layer,
            stage_index,
            attention_transcript,
            attention_output,
            c_proj_output,
            c_proj_transcript,
        )

    ln2_output = digest(seed + 40)
    ln2_transcript = digest(seed + 50)
    write_artifact(
        root,
        layer,
        4,
        c_proj_transcript,
        c_proj_output,
        ln2_output,
        ln2_transcript,
    )

    expansion_output = digest(seed + 60)
    expansion_transcript = digest(seed + 70)
    for stage_index in (5, 6):
        write_artifact(
            root,
            layer,
            stage_index,
            ln2_transcript,
            ln2_output,
            expansion_output,
            expansion_transcript,
        )

    projection_output = digest(seed + 80)
    projection_transcript = digest(seed + 90)
    for stage_index in (7, 8):
        write_artifact(
            root,
            layer,
            stage_index,
            expansion_transcript,
            expansion_output,
            projection_output,
            projection_transcript,
        )
    return projection_output, projection_transcript


class FullInferenceRunnerTests(unittest.TestCase):
    def test_full_plan_reuses_one_block_pipeline(self) -> None:
        targets = target_stages(11, "mlp-projection-join")
        self.assertEqual(len(targets), 12 * len(STAGES))
        self.assertEqual(PROOF_INSTANCES_PER_BLOCK, 41)
        self.assertEqual(FULL_PROOF_INSTANCES, 492)
        self.assertEqual(
            sum(stage.proof_instances for _, stage in targets), FULL_PROOF_INSTANCES
        )
        self.assertEqual(
            [stage.binary for layer, stage in targets if stage.key == "attention"],
            ["zkgpt_leaf"] * 12,
        )

    def test_two_complete_blocks_form_one_valid_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, transcript = write_layer(root, 0, None, None)
            write_layer(root, 1, output, transcript)
            artifacts = load_and_validate_targets(
                root, target_stages(1, "mlp-projection-join")
            )
            progress = progress_summary(artifacts)
            self.assertEqual(progress["completed_layers"], 2)
            self.assertEqual(progress["proof_instances"], 82)

            command = stage_command(
                STAGES[0], 1, root, Path("/model"), Path("/binaries")
            )
            self.assertEqual(command[0], "/binaries/zkgpt_leaf")
            self.assertIn(str(root / "layer-00" / "mlp-projection"), command)
            self.assertIn(str(root / "layer-00" / "mlp-projection-join"), command)

    def test_cross_layer_commitment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, transcript = write_layer(root, 0, None, None)
            write_layer(root, 1, output, transcript)
            path = manifest_path(root, 1, STAGES[0])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["input_commitment"] = digest(999_999)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "input does not match"):
                load_and_validate_targets(
                    root, target_stages(1, "mlp-projection-join")
                )

    def test_resume_rejects_a_manifest_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_layer(root, 0, None, None)
            manifest_path(root, 0, STAGES[3]).unlink()
            with self.assertRaisesRegex(ValidationError, "artifact gap"):
                check_manifest_prefix(
                    root, target_stages(0, "mlp-projection-join")
                )


if __name__ == "__main__":
    unittest.main()
