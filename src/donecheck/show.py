"""Print one run for hand-labelling. The hidden-test result is never shown.

    uv run python -m donecheck.show 7                                   # run 7 in labels/sample_100.tsv
    uv run python -m donecheck.show data/<model folder>/trajs/<bug>.traj.json
    uv run python -m donecheck.show 7 | less                            # to page through it
"""

import csv
import sys
from pathlib import Path

from donecheck.loader import load_run

SAMPLE_FILE = Path("labels/sample_100.tsv")
HEAD_LINES, TAIL_LINES = 12, 6  # how much of each command's output to print
STATUS = {"ran": "exit code {}", "timeout": "timed out", "format_error": "nothing ran (format error)",
          "submitted": "submitted", "no_reply": "no reply"}


def run_path(arg: str) -> Path:
    """A path as given, or the path of that order number in the sample list."""
    if not arg.isdigit():
        return Path(arg)
    with SAMPLE_FILE.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["order"] == arg:
                return Path(row["path"])
    raise SystemExit(f"{arg} is not an order number in {SAMPLE_FILE}")


def trim(text: str) -> str:
    lines = text.rstrip("\n").splitlines()
    if len(lines) <= HEAD_LINES + TAIL_LINES:
        return "\n".join(lines)
    hidden = len(lines) - HEAD_LINES - TAIL_LINES
    return "\n".join(lines[:HEAD_LINES] + [f"[... {hidden} more lines ...]"] + lines[-TAIL_LINES:])


def show(path: Path) -> str:
    run = load_run(path, labels={})  # no labels passed, so the hidden-test result is unknown here
    out = [f"{run.model} / {run.instance_id}: {run.wording} wording, agent {run.agent_version}",
           f"Ended: {run.exit_status} after {len(run.steps)} steps. Hidden-test result: not shown."]
    for step in run.steps:
        out.append(f"\n--- messages[{step.message}]: {STATUS[step.outcome].format(step.exit_code)}")
        if step.note:
            out.append(step.note)
        if step.command is not None:
            out.append(f"$ {step.command}")
        if step.output and step.outcome != "submitted":
            out.append(trim(step.output))
    out.append(f"\n=== Submission ({len(run.submission)} characters)")
    out.append(run.submission or "(empty)")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(show(run_path(sys.argv[1])))
