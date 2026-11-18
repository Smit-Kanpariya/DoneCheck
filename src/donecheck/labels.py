"""The hand-labelling study (docs/claims.md): checking the label sheets, and the analyses once complete.

Human labels live only in labels/claims.tsv and, for the blind relabel of runs 1-30,
labels/claims_repeat.tsv. Nothing here writes or suggests a label.
- check: progress and format; reads only the label sheets, so it reveals no hidden-test result.
- report: runs only when all 100 runs are labelled; joins the labels with the hidden-test results
  and writes eval/results/labels.json.
"""

import csv
import dataclasses
import json
from collections import Counter
from pathlib import Path

from donecheck.checks import check_run
from donecheck.evaluate import done_label
from donecheck.loader import Run, load_labels, load_run

CLAIMS, REPEAT, SAMPLE = "labels/claims.tsv", "labels/claims_repeat.tsv", "labels/sample_100.tsv"
HEADER = ["order", "instance_id", "message", "claim", "label", "comment"]
LABELS = ("supported", "unsupported", "contradicted", "not claimed")
RANGES = ((1, 40), (41, 70), (71, 100))


def read_sample(path: str = SAMPLE) -> dict[int, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["order"]): r for r in csv.DictReader(fh, delimiter="\t")}


def read_sheet(path: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """The header and the non-blank rows, each with its line number."""
    if not Path(path).exists():
        return HEADER, []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, [])
        return header, [(n, row) for n, row in enumerate(reader, 2) if any(c.strip() for c in row)]


def check(path: str = CLAIMS, sample_path: str = SAMPLE, runs: range = range(1, 101)) -> dict:
    """Format problems and how many runs have labels; complete when every run in `runs` has
    exactly one done row and there are no problems."""
    sample = read_sample(sample_path)
    header, rows = read_sheet(path)
    problems = [] if header == HEADER else [f"header is {header}, expected {HEADER}"]
    labelled, done_rows = set(), Counter()
    for n, row in rows:
        if len(row) > len(HEADER):
            problems.append(f"line {n}: {len(row)} columns, expected at most {len(HEADER)}")
        order, bug, message, claim, label = (c.strip() for c in (row + [""] * 5)[:5])
        if not order.isdigit() or int(order) not in sample:
            problems.append(f"line {n}: order {order!r} is not a run in the sample")
            continue
        labelled.add(int(order))
        done_rows[int(order)] += claim == "done"
        if bug != sample[int(order)]["instance_id"]:
            problems.append(f"line {n}: bug id {bug!r} is not run {order}'s")
        if not message.isdigit():
            problems.append(f"line {n}: message {message!r} is not a number")
        if not claim:
            problems.append(f"line {n}: the claim is empty")
        if label not in LABELS:
            problems.append(f"line {n}: label {label!r} is not one of {list(LABELS)}")
        elif label == "not claimed" and claim != "done":
            problems.append(f"line {n}: 'not claimed' is only for the done claim")
    bad_done = sorted(o for o in labelled if done_rows[o] != 1)
    if bad_done:
        problems.append(f"runs without exactly one 'done' row: {bad_done}")
    progress = {f"{lo}-{hi}": sum(o in labelled for o in range(lo, hi + 1)) for lo, hi in RANGES}
    return {"rows": len(rows), "labelled_runs": progress, "problems": problems,
            "complete": not problems and all(o in labelled for o in runs)}


def kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa: agreement between two label lists beyond what chance would give."""
    if not a or len(a) != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / len(a) ** 2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def last_edit_message(run: Run) -> int | None:
    """The message number of the step holding the run's last code edit, as checks.py finds it:
    the shortest prefix of steps that already has every code edit."""
    total = check_run(run)["code_edits"]
    if not total:
        return None
    lo, hi = 1, len(run.steps)
    while lo < hi:
        mid = (lo + hi) // 2
        if check_run(dataclasses.replace(run, steps=run.steps[:mid]))["code_edits"] == total:
            hi = mid
        else:
            lo = mid + 1
    return run.steps[lo - 1].message


def rows_by_run(path: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for _, row in read_sheet(path)[1]:
        r = dict(zip(HEADER, (c.strip() for c in row + [""] * 6)))
        out.setdefault(int(r["order"]), []).append(r)
    return out


def report(path: str = CLAIMS, repeat: str = REPEAT, sample_path: str = SAMPLE, runs: range = range(1, 101),
           repeat_runs: range = range(1, 31)) -> dict:
    status = check(path, sample_path, runs)
    if not status["complete"]:
        raise SystemExit(f"the label sheet is not complete: {status}")
    sample, human = read_sample(sample_path), rows_by_run(path)
    results_cache: dict[str, dict] = {}
    per_run = []
    for order, s in sorted((o, s) for o, s in sample.items() if o in runs):
        model_dir = Path(s["path"]).parent.parent
        results = results_cache.setdefault(str(model_dir), load_labels(model_dir))
        run = load_run(s["path"], results)
        facts = check_run(dataclasses.replace(run, resolved=None))
        rows = human[order]
        done = next(r["label"] for r in rows if r["claim"] == "done")
        cut = last_edit_message(run)
        late_contradicted = any(r["label"] == "contradicted" and r["claim"] != "done" and cut is not None
                                and int(r["message"]) > cut for r in rows)
        per_run.append({"order": order, "wording": run.wording, "fixed": run.resolved, "human_done": done,
                        "rules_done": done_label(facts), "overclaim": done == "contradicted" or late_contradicted})
    out = {"runs": len(per_run), "rows": status["rows"],
           "done_claim_agreement_with_rules": {
               "share": sum(r["human_done"] == r["rules_done"] for r in per_run) / len(per_run),
               "kappa": kappa([r["human_done"] for r in per_run], [r["rules_done"] for r in per_run])},
           "overclaims": {}}
    for outcome, fixed in (("not fixed", False), ("fixed", True)):
        rs = [r for r in per_run if r["fixed"] is fixed]
        out["overclaims"][outcome] = {"runs": len(rs), "overclaim": sum(r["overclaim"] for r in rs),
                                      "by_wording": {w: [sum(r["overclaim"] for r in rs if r["wording"] == w),
                                                         sum(r["wording"] == w for r in rs)] for w in ("older", "newer")}}
    if check(repeat, sample_path, runs=repeat_runs)["complete"]:
        again = rows_by_run(repeat)
        first = [next(r["label"] for r in human[o] if r["claim"] == "done") for o in repeat_runs]
        second = [next(r["label"] for r in again[o] if r["claim"] == "done") for o in repeat_runs]
        out["self_agreement_done_claim"] = {"runs": len(first), "share": sum(a == b for a, b in zip(first, second)) / len(first),
                                            "kappa": kappa(first, second)}
    else:
        out["self_agreement_done_claim"] = "pending: labels/claims_repeat.tsv does not yet cover runs 1-30"
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/labels.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
