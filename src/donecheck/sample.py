"""Draw the 100 runs to hand-label, at random and before reading any of them (plan sections 9 and 10).

From each of the five building models' submitted runs: 14 that failed the hidden tests and
6 that passed, so 70 and 30 in all, shuffled into the order to label them in. The saved list
leaves out the hidden-test results, so the labelling stays blind to them.
Run it with: uv run python -m donecheck.sample data
"""

import csv
import io
import random
import sys
from collections.abc import Iterable
from pathlib import Path

from donecheck.loader import Run, iter_runs

SEED = 20261117
PER_MODEL = {False: 14, True: 6}  # runs to draw per model: not fixed, fixed
SAMPLE_FILE = Path("labels/sample_100.tsv")


def draw_sample(runs: Iterable[Run], seed: int = SEED) -> list[tuple[str, str]]:
    """(model, bug id) pairs in labelling order."""
    pools: dict[tuple[str, bool], list[str]] = {}
    for run in runs:
        if run.submitted and run.resolved is not None:
            pools.setdefault((run.model, run.resolved), []).append(run.instance_id)
    rng = random.Random(seed)
    picked = []
    for (model, fixed), bug_ids in sorted(pools.items()):
        picked += [(model, bug_id) for bug_id in rng.sample(sorted(bug_ids), PER_MODEL[fixed])]
    rng.shuffle(picked)
    return picked


def sample_table(picked: list[tuple[str, str]], data_dir: Path) -> str:
    """The sample as a tab-separated table: order, model, bug id and the run file's path."""
    folders = {p.name.split("_", 2)[2]: p for p in data_dir.iterdir() if (p / "trajs").is_dir()}
    out = io.StringIO()
    table = csv.writer(out, delimiter="\t", lineterminator="\n")
    table.writerow(["order", "model", "instance_id", "path"])
    for order, (model, bug_id) in enumerate(picked, start=1):
        table.writerow([order, model, bug_id, (folders[model] / "trajs" / f"{bug_id}.traj.json").as_posix()])
    return out.getvalue()


def main(data_dir: str) -> None:
    data = Path(data_dir)
    table = sample_table(draw_sample(iter_runs(data)), data)
    if SAMPLE_FILE.exists():
        # The list is drawn once; refuse to replace it with a different one after labelling has begun.
        if SAMPLE_FILE.read_text(encoding="utf-8") != table:
            raise SystemExit(f"{SAMPLE_FILE} already exists and differs from this draw; not replacing it.")
        print(f"{SAMPLE_FILE} is already this draw; nothing changed.")
        return
    SAMPLE_FILE.parent.mkdir(exist_ok=True)
    SAMPLE_FILE.write_text(table, encoding="utf-8")
    print(f"Wrote {table.count(chr(10)) - 1} runs to {SAMPLE_FILE}.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
