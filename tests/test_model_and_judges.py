"""Tests for the DoneCheck score (donecheck.model), the AI-judge prompts (donecheck.judges) and the
evaluation's guards (donecheck.evaluate)."""

import dataclasses
import math

import numpy as np
import pytest

from donecheck import evaluate
from donecheck.checks import check_run
from donecheck.judges import cut, parse_answer, prompt
from donecheck.model import export, fit, leave_model_and_bugs_out, leave_one_model_out, matrix, score, vectorize
from test_checks import fake_run, step

FIX = "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"


def a_run(resolved=None):
    run = fake_run([step("sed -i 's/a/b/' pkg/x.py", note="THOUGHT: Fix it."),
                    step("pytest -q", "3 passed in 0.10s\n", note="THOUGHT: Run the tests.")], FIX)
    return dataclasses.replace(run, resolved=resolved, task="Calling f() returns 1, not 2.")


def test_vectorize_one_hot_and_counts():
    x = vectorize(check_run(a_run()))
    assert x["last_check_result=passed"] == 1.0 and x["last_check_result=none"] == 0.0
    assert x["submission_empty"] == 0.0 and math.isclose(x["code_edits"], math.log1p(1))
    assert "cost" not in x and "api_calls" not in x  # they would identify the model


def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(int)
    return X, y, np.array(["a", "b", "c"] * (n // 3))


def test_exported_model_scores_like_the_scikit_learn_pipeline():
    runs = [a_run(), fake_run([step("python r.py")], ""), fake_run([step("pytest", "1 failed in 0.1s\n", 1)], FIX)]
    facts = [check_run(r) for r in runs] * 4
    names, X = matrix(facts)
    y = np.array([0, 1, 1] * 4)
    model = fit(X, y)
    artifact = export(model, names, 0.5, {})
    assert all(math.isclose(score(artifact, f)[0], p, rel_tol=1e-9)
               for f, p in zip(facts, model.predict_proba(X)[:, 1]))


def test_leave_one_model_out_never_trains_on_the_model_it_scores():
    X, y, models = synthetic()
    before, folds = leave_one_model_out(X, y, models)
    flipped = y.copy()
    flipped[models == "c"] = 1 - flipped[models == "c"]
    after, _ = leave_one_model_out(X, flipped, models)
    assert np.allclose(before[models == "c"], after[models == "c"])  # c's labels never reached c's fold
    assert not np.allclose(before[models == "a"], after[models == "a"])  # but they do train the others
    assert [f["held_out"] for f in folds] == ["a", "b", "c"]


def test_leave_model_and_bugs_out_never_trains_on_the_bug_it_scores():
    X, y, models = synthetic()
    bugs = np.repeat([f"b{i}" for i in range(100)], 3)  # each bug run once by each of a, b, c
    before, per_model = leave_model_and_bugs_out(X, y, models, bugs)
    flipped = y.copy()
    flipped[bugs == "b0"] = 1 - flipped[bugs == "b0"]  # b0's labels in every model
    after, _ = leave_model_and_bugs_out(X, flipped, models, bugs)
    assert set(per_model) == {"a", "b", "c"}
    assert np.allclose(before[bugs == "b0"], after[bugs == "b0"])  # no b0 label reached a b0 score
    assert not np.allclose(before, after)  # but they do train the scores of other bugs


def test_judge_prompts_hold_no_result_and_no_commands():
    for judge in ("notes", "task_diff"):
        assert prompt(judge, a_run(resolved=True)) == prompt(judge, a_run(resolved=False))
    notes = prompt("notes", a_run())
    assert "Fix it." in notes and "Run the tests." in notes
    assert "sed -i" not in notes and "3 passed" not in notes and "diff --git" not in notes
    task_diff = prompt("task_diff", a_run())
    assert "Calling f() returns 1, not 2." in task_diff and "+b" in task_diff and "Fix it." not in task_diff
    empty = dataclasses.replace(a_run(), submission="")
    assert "(empty: the agent submitted no change)" in prompt("task_diff", empty)


def test_cut_and_parse_answer():
    text = "a" * 10 + "b" * 10
    assert cut(text, 8) == "aaaa\n[... 12 characters cut ...]\nbbbb" and cut(text, 20) == text
    assert parse_answer('{"p_not_fixed": 85, "reason": "no test ran"}') == 0.85
    assert parse_answer('Sure. {"p_not_fixed": "12.5"}') == 0.125
    assert parse_answer('{"p_not_fixed": 250}') == 1.0
    assert parse_answer("I cannot tell.") is None


def test_rule_based_done_label():
    facts = check_run(a_run())
    assert evaluate.done_label(facts) == "supported"
    assert evaluate.done_label(facts | {"last_check_result": "error"}) == "unsupported"
    assert evaluate.done_label(facts | {"last_check_result": "failed"}) == "contradicted"
    assert evaluate.done_label(facts | {"gave_up": True}) == "not claimed"


def test_final_test_refuses_without_a_freeze_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="freeze"):
        evaluate.final()
    (tmp_path / "eval").mkdir()
    (tmp_path / "model.json").write_text("changed")
    (tmp_path / "eval/freeze.json").write_text('{"files_sha256": {"model.json": "%s"}}' % ("0" * 64))
    with pytest.raises(SystemExit, match="changed since the freeze"):
        evaluate.final()
    (tmp_path / "eval/results").mkdir()
    (tmp_path / "eval/results/final_glm45.json").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        evaluate.final()
