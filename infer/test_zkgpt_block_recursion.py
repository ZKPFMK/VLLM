from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from zkgpt_block_recursion import (
    Node,
    ValidationError,
    block_command,
    parse_args,
    run,
    validate_adjacent,
)


def node(start: int, end: int, input_value: str, output_value: str) -> Node:
    transcript = f"transcript-{end}"
    return Node(
        manifest=Path(f"/proofs/{start}-{end}.json"),
        kind="block" if end - start == 1 else "recursion",
        start_block=start,
        end_block=end,
        input_commitment=input_value,
        output_commitment=output_value,
        first_upstream_transcript=None if start == 0 else f"transcript-{start}",
        last_block_transcript=transcript,
        transcript_commitment=f"node-{start}-{end}",
    )


class BlockRecursionRunnerTests(unittest.TestCase):
    def test_dry_run_has_twelve_leaves_and_eleven_joins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--dry-run", "--output-root", temporary])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(run(args), 0)
            lines = output.getvalue().splitlines()
            leaf_commands = [
                line for line in lines if "/zkgpt_like " in line and " --block " in line
            ]
            join_commands = [
                line for line in lines if "/zkgpt_block_recursion " in line
            ]
            self.assertEqual(len(leaf_commands), 12)
            self.assertEqual(len(join_commands), 11)
            self.assertIn("12 -> 6 -> 3 -> 2 -> 1", lines[-1])

    def test_later_block_consumes_the_previous_directory(self) -> None:
        command = block_command(
            Path("/proofs"), Path("/model"), Path("/bin"), 7
        )
        self.assertEqual(command[0], "/bin/zkgpt_like")
        self.assertIn("/proofs/blocks/block-06", command)
        self.assertIn("/proofs/blocks/block-07", command)

    def test_boundary_requires_both_output_and_transcript_continuity(self) -> None:
        left = node(0, 1, "input-0", "boundary")
        right = node(1, 2, "boundary", "output-2")
        validate_adjacent(left, right)

        broken = Node(
            manifest=right.manifest,
            kind=right.kind,
            start_block=right.start_block,
            end_block=right.end_block,
            input_commitment=right.input_commitment,
            output_commitment=right.output_commitment,
            first_upstream_transcript="wrong",
            last_block_transcript=right.last_block_transcript,
            transcript_commitment=right.transcript_commitment,
        )
        with self.assertRaisesRegex(ValidationError, "transcript chain"):
            validate_adjacent(left, broken)


if __name__ == "__main__":
    unittest.main()
