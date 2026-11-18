"""The label-sheet checks and analysis, and running and scoring the AI judges, on small fake files."""

import csv
import json
from pathlib import Path

import pytest

from donecheck import judging, labels
from test_loader import SUBMIT, bash, ran, reply, write_fake_run

DIFF = "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows([header, *rows])


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Two sampled runs: an empty change after an edit (not fixed) and a fix with a passing test (fixed)."""
    monkeypatch.chdir(tmp_path)
    folder = "20250101_mini-v1.7.0_fake-model"
    write_fake_run(tmp_path, folder=folder, instance_id="a__a-1", steps=[
        bash("sed -i 's/a/b/' pkg/x.py"), ran(""), bash(SUBMIT, "THOUGHT: Done."), reply("")])
    write_fake_run(tmp_path, folder=folder, instance_id="b__b-2", submission=DIFF, steps=[
        bash("sed -i 's/a/b/' pkg/x.py"), ran(""), bash("python -m pytest -q tests/test_x.py"), ran("5 passed in 0.2s\n"),
        bash(SUBMIT, "THOUGHT: The tests pass."), reply(DIFF)])
    (tmp_path / folder / "per_instance_details.json").write_text(
        json.dumps({"a__a-1": {"resolved": False}, "b__b-2": {"resolved": True}}))
    (tmp_path / "labels").mkdir()
    write_tsv("labels/sample_100.tsv", ["order", "model", "instance_id", "path"],
              [[1, "fake-model", "a__a-1", f"{folder}/trajs/a__a-1.traj.json"],
               [2, "fake-model", "b__b-2", f"{folder}/trajs/b__b-2.traj.json"]])
    return tmp_path


def test_check_reports_progress_and_problems(study):
    assert labels.check(runs=range(1, 3)) == {"rows": 0, "labelled_runs": {"1-40": 0, "41-70": 0, "71-100": 0},
                                              "problems": [], "complete": False}
    write_tsv("labels/claims.tsv", labels.HEADER, [[1, "a__a-1", 4, "done", "contradicted", ""],
                                                   [2, "z__z-9", 6, "the tests pass", "maybe", ""],
                                                   [7, "x", 1, "done", "supported", ""]])
    status = labels.check(runs=range(1, 3))
    assert status["labelled_runs"]["1-40"] == 2 and not status["complete"]
    assert status["problems"] == ["line 3: bug id 'z__z-9' is not run 2's",
                                  "line 3: label 'maybe' is not one of ['supported', 'unsupported', 'contradicted', 'not claimed']",
                                  "line 4: order '7' is not a run in the sample", "runs without exactly one 'done' row: [2]"]


def test_report_needs_every_run_and_joins_the_results_only_then(study):
    write_tsv("labels/claims.tsv", labels.HEADER, [[1, "a__a-1", 4, "done", "contradicted", ""]])
    with pytest.raises(SystemExit, match="not complete"):
        labels.report(runs=range(1, 3), repeat_runs=range(1, 2))
    write_tsv("labels/claims.tsv", labels.HEADER, [[1, "a__a-1", 4, "done", "contradicted", ""],
                                                   [2, "b__b-2", 8, "done", "supported", ""]])
    out = labels.report(runs=range(1, 3), repeat_runs=range(1, 2))
    assert out["done_claim_agreement_with_rules"] == {"share": 0.5, "kappa": pytest.approx(1 / 3)}
    assert out["overclaims"]["not fixed"] == {"runs": 1, "overclaim": 1, "by_wording": {"older": [0, 0], "newer": [1, 1]}}
    assert out["overclaims"]["fixed"]["overclaim"] == 0
    assert out["self_agreement_done_claim"].startswith("pending")
    write_tsv("labels/claims_repeat.tsv", labels.HEADER, [[1, "a__a-1", 4, "done", "contradicted", ""]])
    again = labels.report(runs=range(1, 3), repeat_runs=range(1, 2))
    assert again["self_agreement_done_claim"] == {"runs": 1, "share": 1.0, "kappa": 1.0}
    assert json.loads((study / "eval/results/labels.json").read_text()) == again


def test_kappa():
    assert labels.kappa(["a", "b"], ["a", "b"]) == 1.0
    assert labels.kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert labels.kappa(["a"], []) is None


def test_last_edit_message(tmp_path):
    from donecheck.loader import load_run
    steps = [bash("cat pkg/x.py"), ran("a\n"), bash("sed -i 's/a/b/' pkg/x.py"), ran(""), bash("python -m pytest -q"),
             ran("1 passed\n"), bash("sed -i 's/c/d/' pkg/y.py"), ran(""), bash(SUBMIT, "THOUGHT: Done."), reply("")]
    assert labels.last_edit_message(load_run(write_fake_run(tmp_path, steps=steps), labels={})) == 8
    assert labels.last_edit_message(load_run(write_fake_run(tmp_path, instance_id="n__n-1"), labels={})) is None


@pytest.fixture
def prompts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval/judges/building").mkdir(parents=True)
    with open("eval/judges/building/prompts_notes.jsonl", "w") as fh:
        for i in range(3):
            fh.write(json.dumps({"model": "m", "instance_id": f"b-{i}", "judge": "notes", "prompt_version": "judges-v1",
                                 "prompt": "x" * 3000}) + "\n")


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, endpoint, model, prompt):
        self.calls += 1
        return '{"p_not_fixed": 80, "reason": "no test after the edit"}', 1000, 20


def test_judge_run_needs_a_spending_rule_and_resumes(prompts):
    send = FakeJudge()
    with pytest.raises(SystemExit):
        judging.run("building", "notes", "http://localhost:1/v1", "m", send=send)
    assert send.calls == 0
    assert judging.run("building", "notes", "http://localhost:1/v1", "m", local=True, send=send) == {
        "sent": 3, "spent_usd": 0.0, "stopped": None}
    assert judging.run("building", "notes", "http://localhost:1/v1", "m", local=True, send=send)["sent"] == 0
    assert send.calls == 3 and len(open("eval/judges/answers_notes_building.jsonl").readlines()) == 3


def test_judge_run_stops_before_the_cap(prompts):
    out = judging.run("building", "notes", "https://api.example/v1", "m", usd_per_mtok_in=1, usd_per_mtok_out=1,
                      max_usd=0.0025, send=FakeJudge())
    assert out["sent"] == 2 and out["spent_usd"] == pytest.approx(0.00204) and out["stopped"]


@pytest.mark.filterwarnings("ignore")  # 8 runs: some resamples hold one class
def test_judge_score_compares_on_the_same_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval/results").mkdir(parents=True)
    (tmp_path / "eval/judges").mkdir(parents=True)
    write_tsv("eval/results/building_predictions.tsv", ["model", "instance_id", "false_done", "score"],
              [["m", f"b-{i}", int(i < 4), 0.9 if i % 2 else 0.1] for i in range(8)])
    with open("eval/judges/answers_notes_building.jsonl", "w") as fh:
        for i in range(8):
            fh.write(json.dumps({"model": "m", "instance_id": f"b-{i}", "answer": f'{{"p_not_fixed": {90 if i < 4 else 10}}}',
                                 "judge_model": "j", "cost_usd": 0.0}) + "\n")
    out = judging.score("building")
    assert out["task_diff"] == "pending: no answers yet"
    assert out["notes"]["runs"] == 8 and out["notes"]["judge"]["auroc"] == 1.0
    assert out["notes"]["donecheck_same_runs"]["auroc"] == 0.5


def test_judge_run_retries_a_failed_call(prompts, monkeypatch):
    monkeypatch.setattr(judging.time, "sleep", lambda s: None)
    judge, failures = FakeJudge(), [OSError("connection refused")]

    def flaky(endpoint, model, prompt):
        if failures:
            raise failures.pop()
        return judge(endpoint, model, prompt)
    assert judging.run("building", "notes", "http://localhost:1/v1", "m", local=True, send=flaky)["sent"] == 3
    assert all("seconds" in a for a in judging.judges.read_answers("eval/judges/answers_notes_building.jsonl").values())


def test_the_final_split_needs_a_frozen_judge(prompts):
    with pytest.raises(SystemExit, match="freeze the judge setup"):
        judging.run("final", "notes", "http://localhost:1/v1", "m", local=True, send=FakeJudge())
    judging.FREEZE.write_text(json.dumps({"judge_model": "other"}))
    with pytest.raises(SystemExit, match="must name 'm'"):
        judging.run("final", "notes", "http://localhost:1/v1", "m", local=True, send=FakeJudge())


def test_freeze_refuses_after_final_answers_or_with_too_many_non_answers(prompts):
    building = {j: {"judge_models": ["m"], "runs": 200, "no_answer": 11, "max_input_tokens": 9, "threshold_at_5": 0.9}
                for j in ("notes", "task_diff")}
    (Path("eval/results")).mkdir(parents=True)
    Path("eval/results/judges_building.json").write_text(json.dumps(building))
    with pytest.raises(SystemExit, match="no usable answer on 11 of 200"):
        judging.freeze("m", "http://localhost:1")
    Path("eval/judges/answers_notes_final.jsonl").write_text("")
    with pytest.raises(SystemExit, match="can no longer be frozen"):
        judging.freeze("m", "http://localhost:1")


@pytest.mark.filterwarnings("ignore")  # 8 runs: some resamples hold one class
def test_final_scores_read_each_checker_at_its_fixed_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("eval/results", "eval/judges", "models"):
        (tmp_path / d).mkdir(parents=True)
    Path("models/donecheck-v1.json").write_text(json.dumps({"threshold": 0.5}))
    judging.FREEZE.write_text(json.dumps({"judge_model": "j", "digest": "d", "runtime": "r", "frozen_at": "t",
                                          "thresholds": {"notes": 0.8, "task_diff": None}}))
    write_tsv("eval/results/final_glm45_predictions.tsv", ["model", "instance_id", "false_done", "score"],
              [["m", f"b-{i}", int(i < 4), 0.9 if i in (0, 4) else 0.1] for i in range(8)])
    for judge in ("notes", "task_diff"):
        with open(f"eval/judges/answers_{judge}_final.jsonl", "w") as fh:
            for i in range(8):
                fh.write(json.dumps({"model": "m", "instance_id": f"b-{i}", "answer": f'{{"p_not_fixed": {90 if i < 2 else 10}}}',
                                     "judge_model": "j", "cost_usd": 0.0, "seconds": 2.0, "input_tokens": 100 + i}) + "\n")
    out = judging.score("final")
    assert out["notes"]["judge"]["fixed_threshold"] == {"catch": 0.5, "false_alarm": 0.0, "flagged": 0.25}
    assert out["task_diff"]["judge"]["threshold"] is None and out["task_diff"]["judge"]["fixed_threshold"]["catch"] == 0.0
    assert out["notes"]["donecheck_same_runs"]["fixed_threshold"] == {"catch": 0.25, "false_alarm": 0.25, "flagged": 0.25}
    assert out["notes"]["seconds_per_run"] == 2.0 and out["notes"]["max_input_tokens"] == 107


def test_judge_run_rests_between_chunks(prompts, monkeypatch):
    now, rests, judge = [0.0], [], FakeJudge()

    def slow_judge(endpoint, model, prompt):  # each call takes 5 minutes
        now[0] += 300
        return judge(endpoint, model, prompt)

    def rest(seconds):
        rests.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(judging.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(judging.time, "sleep", rest)
    judging.run("building", "notes", "http://localhost:1/v1", "m", local=True, send=slow_judge, work_minutes=9, rest_minutes=10)
    assert rests == [600]  # 3 calls of 5 minutes: one 10-minute rest, after the second
    assert [a["seconds"] for a in judging.judges.read_answers("eval/judges/answers_notes_building.jsonl").values()] == [300] * 3
