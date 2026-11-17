"""Day 1 counts: runs, submitted, fixed and false "done" runs per model and per instruction wording.

A false "done" is a run that submitted but failed the hidden tests.
Run it with: uv run python -m donecheck.counts data
"""

import sys
from collections import Counter
from collections.abc import Iterable

from donecheck.loader import Run, iter_runs

# Plan section 10: fewer false "done" runs than this means add models or switch to the backup.
STOP_SIGN = 150
COLUMNS = ["runs", "submitted", "fixed", "false done", "fixed, not submitted", "no result"]


def count_runs(runs: Iterable[Run]) -> dict[str, Counter]:
    """Counts for each model, each instruction wording, and all runs together."""
    counts: dict[str, Counter] = {}
    for run in runs:
        for group in (run.model, f"{run.wording} wording", "all"):
            c = counts.setdefault(group, Counter())
            c["runs"] += 1
            c["submitted"] += run.submitted
            c["fixed"] += run.resolved is True
            c["false done"] += run.submitted and run.resolved is False
            c["fixed, not submitted"] += run.resolved is True and not run.submitted
            c["no result"] += run.resolved is None
    return counts


def main(data_dir: str) -> None:
    counts = count_runs(iter_runs(data_dir))
    print("group\t" + "\t".join(COLUMNS))
    for group in sorted(counts, key=lambda g: (g == "all", g.endswith(" wording"), g)):
        print(group + "\t" + "\t".join(str(counts[group][c]) for c in COLUMNS))
    false_done = counts["all"]["false done"]
    verdict = "passes" if false_done >= STOP_SIGN else "FAILS"
    print(f"\nStop sign: {false_done} false 'done' runs in total, so it {verdict} (it needs {STOP_SIGN}).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
