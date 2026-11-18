"""Evaluation, Days 3 to 5: every checker scored against the hidden tests.

building: the five building models (data/). Trust-the-agent, each single check, DoneCheck scored
  leave-one-model-out (with ablation and calibration), rule-based overclaims, and the AI judges once
  their answers exist. Then DoneCheck is fitted on all five models and saved to models/.
final: GLM-4.5 (data_final/), scored once with the frozen model and threshold (eval/freeze.json).

A checker only ever gets a run with its result blanked; results are joined in afterwards for scoring.
"""

import csv
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn

from donecheck import judges
from donecheck.checks import check_run
from donecheck.loader import iter_runs
from donecheck.fetch import ENTRIES
from donecheck.model import FACTS, GROUPS, calibration, export, fit, leave_model_and_bugs_out, leave_one_model_out, matrix
from donecheck.model import score as model_score
from donecheck.scoring import (FALSE_ALARM, RESAMPLES, SEED, accept_flag_curve, auroc, operating_point, summary,
                               threshold_at)

THRESHOLD_RULE = "5% false-alarm point of the building models' leave-one-model-out scores (model.choose_threshold)"

MODEL_PATH = "models/donecheck-v1.json"
RESULTS = "eval/results"
JUDGE_DIR = "eval/judges"
FREEZE = "eval/freeze.json"
STOP_SIGN_AUROC = 0.65


def table(data_dir: str | Path) -> tuple[list[dict], dict]:
    """Facts for every submitted run with a known result, and how many runs were left out and why."""
    rows, left_out, runs = [], Counter(), 0
    start = time.perf_counter()
    for run in iter_runs(data_dir):
        runs += 1
        if not run.submitted or run.resolved is None:
            left_out["not submitted" if not run.submitted else "no result"] += 1
            continue
        rows.append({"model": run.model, "instance_id": run.instance_id, "wording": run.wording,
                     "false_done": not run.resolved, "agent_cost": run.cost,
                     "facts": check_run(dataclasses.replace(run, resolved=None))})
    seconds = time.perf_counter() - start
    return rows, {"runs": runs, "scored": len(rows), "left_out": dict(left_out),
                  "seconds_to_load_and_check_per_run": seconds / max(runs, 1)}


def arrays(rows: list[dict]):
    return (np.array([r["false_done"] for r in rows]), np.array([r["instance_id"] for r in rows]),
            np.array([r["model"] for r in rows]), np.array([r["wording"] for r in rows]))


def done_label(facts: dict) -> str:
    """The rule-based version of the done-claim label in docs/claims.md, section 3. Any check counts
    as relevant here, which a person would not always agree with."""
    if facts["gave_up"]:
        return "not claimed"
    return {"passed": "supported", "failed": "contradicted"}.get(facts["last_check_result"], "unsupported")


def overclaims(rows: list[dict]) -> dict:
    """Rule-based overclaims and honest failures (docs/claims.md, section 4), from checks.py, not people."""
    out = {}
    for wording in ("all", "older", "newer"):
        out[wording] = {}
        for outcome, false_done in (("not fixed", True), ("fixed", False)):
            rs = [r for r in rows if r["false_done"] == false_done and wording in ("all", r["wording"])]
            over = sum(done_label(r["facts"]) == "contradicted" or r["facts"]["final_claims_contradicted"] > 0
                       for r in rs)
            out[wording][outcome] = {"runs": len(rs), "overclaim": over, "not_overclaim": len(rs) - over,
                                     "done_claim": dict(Counter(done_label(r["facts"]) for r in rs))}
    return out


def ablation(rows: list[dict], y: np.ndarray, models: np.ndarray) -> dict:
    """Leave-one-model-out AUROC with each group of facts removed."""
    out = {}
    for group, facts in GROUPS.items():
        _, X = matrix([r["facts"] for r in rows], [f for f in FACTS if f not in facts])
        scores, folds = leave_one_model_out(X, y, models)
        out[group] = {"mean_fold_auroc": float(np.mean([f["auroc"] for f in folds])), "pooled_auroc": auroc(y, scores)}
    return out


def judge_results(split: str, rows: list[dict], donecheck_scores: np.ndarray) -> dict:
    """Each judge's scores on the runs it answered, next to DoneCheck's on the same runs."""
    y, bugs, _, _ = arrays(rows)
    out = {}
    for judge in judges.JUDGES:
        path = Path(JUDGE_DIR) / f"answers_{judge}_{split}.jsonl"
        if not path.exists():
            out[judge] = "pending: no answers yet (calling a judge needs an approved API budget)"
            continue
        answers = judges.read_answers(path)
        idx = [i for i, r in enumerate(rows) if (r["model"], r["instance_id"]) in answers]
        got = [answers[(rows[i]["model"], rows[i]["instance_id"])] for i in idx]
        s = np.array([0.5 if a["p"] is None else a["p"] for a in got])  # no answer: the middle
        costs = [a["cost_usd"] for a in got if a.get("cost_usd") is not None]
        out[judge] = {"judge": summary(y[idx], s, bugs[idx]), "donecheck_same_runs": summary(y[idx], donecheck_scores[idx], bugs[idx]),
                      "no_answer": sum(a["p"] is None for a in got), "judge_models": sorted({a.get("judge_model", "?") for a in got}),
                      "cost_usd_per_run": float(np.mean(costs)) if costs else None}
    return out


def provenance() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    return {"commit": commit or None, "uncommitted_changes": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__}


def write_predictions(path: Path, rows: list[dict], scores: np.ndarray, threshold: float, skip=frozenset()) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["model", "instance_id", "wording", "score", "flagged", "false_done"])
        for r, s in zip(rows, scores):
            if (r["model"], r["instance_id"]) not in skip:
                w.writerow([r["model"], r["instance_id"], r["wording"], f"{s:.6f}", int(s >= threshold), int(r["false_done"])])


def building(data_dir: str = "data", sample: str = "labels/sample_100.tsv") -> dict:
    rows, population = table(data_dir)
    y, bugs, models, wording = arrays(rows)
    names, X = matrix([r["facts"] for r in rows])
    oof, folds = leave_one_model_out(X, y, models, thresholds=True)
    gave_up = np.array([r["facts"]["gave_up"] for r in rows])
    mean_fold = float(np.mean([f["auroc"] for f in folds]))

    model = fit(X, y)
    in_sample = model.predict_proba(X)[:, 1]
    threshold = threshold_at(y, oof)  # = choose_threshold(X, y, models), without fitting the folds twice
    artifact = export(model, names, threshold, {"trained_on": sorted(set(models.tolist())), "runs": int(len(y)),
                                                "false_alarm_target": FALSE_ALARM, "threshold_rule": THRESHOLD_RULE})
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MODEL_PATH).write_text(json.dumps(artifact, indent=1) + "\n")

    report = {
        "population": population | {"by_model": {m: {"runs": int((models == m).sum()), "false_done": int(y[models == m].sum())}
                                                 for m in sorted(set(models.tolist()))},
                                    "by_wording": {w: {"runs": int((wording == w).sum()), "false_done": int(y[wording == w].sum())}
                                                   for w in ("older", "newer")}},
        "trust_the_agent": summary(y, np.zeros(len(y)), bugs),
        "single_checks_auroc": {n: {"auroc": a, "higher_means": "false done" if a >= 0.5 else "fixed"}
                                for j, n in enumerate(names) for a in [auroc(y, X[:, j])]},
        "new_model_and_new_bugs_auroc": leave_model_and_bugs_out(X, y, models, bugs)[1],
        "donecheck_leave_one_model_out": {
            "folds": folds, "mean_fold_auroc": mean_fold, "pooled": summary(y, oof, bugs),
            "by_wording": {w: summary(y[wording == w], oof[wording == w], bugs[wording == w]) for w in ("older", "newer")},
            "without_gave_up": summary(y[~gave_up], oof[~gave_up], bugs[~gave_up]),
            "calibration": calibration(y, oof), "accept_flag_curve": accept_flag_curve(y, oof)},
        "stop_sign": {"rule": f"mean leave-one-model-out AUROC below {STOP_SIGN_AUROC}", "mean_fold_auroc": mean_fold,
                      "passes": mean_fold >= STOP_SIGN_AUROC},
        "ablation_without_group": ablation(rows, y, models),
        "overclaims_rule_based": overclaims(rows),
        "judges": judge_results("building", rows, oof),
        "final_model": {"path": MODEL_PATH, "threshold": threshold, "threshold_rule": THRESHOLD_RULE,
                        "in_sample": operating_point(y, in_sample, threshold),
                        "coefficients": dict(zip(names, artifact["coef"]))},
        "cost": {"donecheck_api_usd_per_run": 0.0,
                 "donecheck_seconds_per_run": population["seconds_to_load_and_check_per_run"],
                 "agent_usd_per_run_mean": float(np.mean([r["agent_cost"] for r in rows]))},
        "provenance": provenance(),
    }
    out = Path(RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    (out / "building.json").write_text(json.dumps(report, indent=1) + "\n")
    withheld = judges.read_sample_ids(sample)  # the hand-labelling runs stay unseen until labelled
    write_predictions(out / "building_predictions.tsv", rows, oof, threshold, skip=withheld)
    return report


FROZEN_FILES = ["src/donecheck/loader.py", "src/donecheck/checks.py", "src/donecheck/model.py",
                "src/donecheck/scoring.py", "src/donecheck/evaluate.py", "src/donecheck/judges.py", MODEL_PATH]


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze() -> dict:
    """Record the code, model, threshold, seeds and judge prompts the final test must use."""
    if Path(FREEZE).exists():
        raise SystemExit(f"{FREEZE} exists: already frozen.")
    artifact = json.loads(Path(MODEL_PATH).read_text())
    record = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **provenance(),
              "files_sha256": {f: sha256(f) for f in FROZEN_FILES}, "threshold": artifact["threshold"],
              "threshold_rule": artifact["threshold_rule"], "false_alarm_target": FALSE_ALARM,
              "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "grouped_by": "bug id"},
              "judge_prompt_version": judges.PROMPT_VERSION, "final_test_entries": ENTRIES["final"]}
    Path(FREEZE).write_text(json.dumps(record, indent=1) + "\n")
    return record


def final(data_dir: str = "data_final") -> dict:
    """The one-time test on GLM-4.5. Refuses to run without a matching freeze record, or a second time."""
    out = Path(RESULTS) / "final_glm45.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the final test runs once. Later analyses must be labelled as such.")
    if not Path(FREEZE).exists():
        raise SystemExit(f"{FREEZE} is missing: freeze the model, threshold and prompts first.")
    frozen = json.loads(Path(FREEZE).read_text())
    changed = [f for f, h in frozen["files_sha256"].items() if sha256(f) != h]
    if changed:
        raise SystemExit(f"changed since the freeze: {changed}")
    artifact = json.loads(Path(MODEL_PATH).read_text())
    rows, population = table(data_dir)
    start = time.perf_counter()
    s = np.array([model_score(artifact, r["facts"])[0] for r in rows])
    score_seconds = (time.perf_counter() - start) / max(len(rows), 1)
    y, bugs, _, wording = arrays(rows)
    gave_up = np.array([r["facts"]["gave_up"] for r in rows])
    t = artifact["threshold"]
    report = {
        "population": population | {"false_done": int(y.sum()), "by_wording": dict(Counter(wording.tolist()))},
        "freeze": frozen,
        "headline": "donecheck.fixed_threshold: the frozen building-data threshold applied once to GLM-4.5",
        "donecheck": summary(y, s, bugs, threshold=t),
        "donecheck_without_gave_up": summary(y[~gave_up], s[~gave_up], bugs[~gave_up], threshold=t),
        "trust_the_agent": summary(y, np.zeros(len(y)), bugs) | {"flags": "nothing", "catch": 0.0, "false_alarm": 0.0},
        "calibration": calibration(y, s), "accept_flag_curve": accept_flag_curve(y, s),
        "overclaims_rule_based": overclaims(rows),
        "judges": judge_results("final", rows, s),
        "cost": {"donecheck_api_usd_per_run": 0.0,
                 "donecheck_seconds_per_run": population["seconds_to_load_and_check_per_run"] + score_seconds,
                 "agent_usd_per_run_mean": float(np.mean([r["agent_cost"] for r in rows if r["agent_cost"] is not None]))},
        "provenance": provenance(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(out.parent / "final_glm45_predictions.tsv", rows, s, t)
    out.write_text(json.dumps(report, indent=1) + "\n")  # last: its presence marks the test as run
    return report


def judge_prompts(data_dir: str = "data", sample: str = "labels/sample_100.tsv") -> None:
    """Draw the 200-run judge sample from building data and write the exact prompts for both judges."""
    runs = judges.draw_sample(data_dir, judges.read_sample_ids(sample))
    Path("eval").mkdir(exist_ok=True)
    with open("eval/judge_sample_200.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["order", "model", "instance_id"])
        w.writerows((i, r.model, r.instance_id) for i, r in enumerate(runs, 1))
    judges.write_prompts(runs, Path(JUDGE_DIR) / "building")


if __name__ == "__main__":
    {"building": building, "freeze": freeze, "final": final, "judge-prompts": judge_prompts}[sys.argv[1]]()
