"""After the final test: what DoneCheck caught, missed and falsely flagged on GLM-4.5.

Exploratory, run after the one-time test; nothing here changes the model, the threshold or the
reported numbers. `python -m donecheck.analysis` writes eval/results/final_glm45_errors.json.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from donecheck.checks import check_run
from donecheck.loader import iter_runs

GROUPS = {(1, 1): "caught", (0, 1): "missed", (1, 0): "false alarm", (0, 0): "accepted, fixed"}


def read_predictions(path: str = "eval/results/final_glm45_predictions.tsv") -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["instance_id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def profile(data_dir: str = "data_final") -> dict:
    """For each outcome group, how often each kind of evidence appears."""
    predictions = read_predictions()
    rows: dict[str, list[dict]] = {g: [] for g in GROUPS.values()}
    notes: dict[str, list[dict]] = {g: [] for g in GROUPS.values()}
    for run in iter_runs(data_dir):
        p = predictions[run.instance_id]
        group = GROUPS[(int(p["flagged"]), int(p["false_done"]))]
        facts = check_run(run)
        rows[group].append(facts)
        notes[group].append({"instance_id": run.instance_id, "score": float(p["score"]), "steps": facts["steps"],
                             "last_check_result": facts["last_check_result"], "submit_note": run.steps[-1].note[:300]})
    out = {}
    for group, facts in rows.items():
        n = len(facts)
        share = lambda test: round(sum(map(test, facts)) / n, 3) if n else None  # noqa: E731
        out[group] = {
            "runs": n,
            "empty change": share(lambda f: f["submission_empty"]),
            "a check passed after the last edit": share(lambda f: f["passed_after_last_edit"]),
            "no check after the last edit": share(lambda f: f["checks_after_last_edit"] == 0),
            "repository tests after the last edit": share(lambda f: f["tests_after_last_edit"] > 0),
            "failed check passed after the fix": share(lambda f: f["fail_then_pass"]),
            "a claim contradicted": share(lambda f: f["claims_contradicted"] > 0),
            "an edited file never opened": share(lambda f: f["edited_unopened"] > 0),
            "last check result": dict(Counter(f["last_check_result"] for f in facts)),
            "median steps": sorted(f["steps"] for f in facts)[n // 2] if n else None,
        }
    examples = {g: sorted(v, key=lambda e: -e["score"]) for g, v in notes.items()}
    out["examples"] = {"caught (highest scores)": examples["caught"][:5], "false alarm": examples["false alarm"],
                       "missed (lowest scores)": examples["missed"][-5:]}
    Path("eval/results/final_glm45_errors.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def threshold_rules(data_dir: str = "data") -> dict:
    """The pre-freeze comparison behind the threshold rule (docs/report.md, section 2), on building
    data only. For each building model left out: rule A takes the 5% point of the training models'
    in-sample scores, rule B (model.choose_threshold) the 5% point of their leave-one-model-out
    scores; both are applied to the model left out."""
    from donecheck.evaluate import arrays, table
    from donecheck.model import choose_threshold, fit, matrix
    from donecheck.scoring import operating_point, threshold_at

    rows, _ = table(data_dir)
    y, _, models, _ = arrays(rows)
    _, X = matrix([r["facts"] for r in rows])
    out, held_out = {}, sorted(set(models.tolist()))
    for held in held_out:
        train, test = models != held, models == held
        model = fit(X[train], y[train])
        s = model.predict_proba(X[test])[:, 1]
        a = threshold_at(y[train], model.predict_proba(X[train])[:, 1])
        b = choose_threshold(X[train], y[train], models[train])
        out[held] = {"in_sample": operating_point(y[test], s, a), "leave_one_model_out": operating_point(y[test], s, b)}
    for rule in ("in_sample", "leave_one_model_out"):
        fa = [out[held][rule]["false_alarm"] for held in held_out]
        out[f"{rule}_mean_false_alarm"] = sum(fa) / len(fa)
        out[f"{rule}_models_above_5"] = sum(x > 0.05 for x in fa)
    Path("eval/results/threshold_rules.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["threshold-rules"]:
        print(json.dumps(threshold_rules(), indent=1))
        raise SystemExit
    result = profile()
    for group in GROUPS.values():
        print(group, json.dumps(result[group]))
    for title, items in result["examples"].items():
        print(f"\n{title}")
        for e in items:
            print(f"  {e['instance_id']} score={e['score']:.3f} steps={e['steps']} last={e['last_check_result']}: "
                  f"{e['submit_note'][:160]!r}")
