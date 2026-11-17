"""Tests for the Day 1 counts, the labelling sample and the run viewer."""

from collections import Counter

from donecheck.counts import count_runs
from donecheck.loader import Run, iter_runs
from donecheck.sample import PER_MODEL, draw_sample
from donecheck.show import show
from test_loader import DATA, SUBMIT, bash, needs_data, ran, reply, write_fake_run

# Plan section 7, from the leaderboard's own files: fixed runs per model, and false "done"
# runs (submitted, then failed the hidden tests) for the models that have an exit-status summary.
LEADERBOARD_FIXED = {"gpt-oss-120b": 130, "gpt-5-nano": 174, "kimi-k2-instruct": 219,
                     "devstral-2512": 269, "deepseek-v3.2-reasoner": 300}
LEADERBOARD_FALSE_DONE = {"gpt-5-nano": 302, "devstral-2512": 211, "deepseek-v3.2-reasoner": 198}


def fake(model, wording, submitted, resolved, bug_id="x__x-1"):
    status = "Submitted" if submitted else "LimitsExceeded"
    return Run(bug_id, model, "1.7.0", status, submitted, "", wording, resolved, ())


def test_count_runs():
    counts = count_runs([fake("m1", "older", True, True), fake("m1", "older", True, False),
                         fake("m2", "newer", False, False), fake("m2", "newer", True, None)])
    assert counts["m1"] == Counter({"runs": 2, "submitted": 2, "fixed": 1, "false done": 1})
    assert counts["m2"] == Counter({"runs": 2, "submitted": 1, "no result": 1})
    assert counts["older wording"] == counts["m1"]
    assert counts["all"]["false done"] == 1  # a run with no result is never a false "done"


@needs_data
def test_counts_match_the_leaderboard():
    counts = count_runs(iter_runs(DATA))
    assert {model: counts[model]["fixed"] for model in LEADERBOARD_FIXED} == LEADERBOARD_FIXED
    assert {model: counts[model]["false done"] for model in LEADERBOARD_FALSE_DONE} == LEADERBOARD_FALSE_DONE
    assert counts["all"]["runs"] == 2500
    assert counts["all"]["fixed, not submitted"] == 0


def test_draw_sample():
    runs = [fake(model, "newer", submitted, fixed, f"{model}__{i}")
            for model in ("m1", "m2") for i, (submitted, fixed) in enumerate(
                [(True, False)] * 20 + [(True, True)] * 10 + [(False, False)] * 5)]
    assert PER_MODEL == {False: 14, True: 6}  # plan section 10: 70 not fixed and 30 fixed over five models
    picked = draw_sample(runs)
    by_id = {r.instance_id: r for r in runs}
    first_half = {model for model, _ in picked[:len(picked) // 2]}
    assert first_half == {"m1", "m2"}  # shuffled: the labelling order mixes the models
    assert len(picked) == len(set(picked)) == 2 * sum(PER_MODEL.values())
    assert all(by_id[bug_id].submitted for _, bug_id in picked)
    for model in ("m1", "m2"):
        mine = [by_id[bug_id].resolved for m, bug_id in picked if m == model]
        assert Counter(mine) == {False: PER_MODEL[False], True: PER_MODEL[True]}
    assert draw_sample(runs) == picked  # the same seed gives the same list
    assert draw_sample(runs, seed=1) != picked


def test_show_hides_the_result_and_prints_each_step(tmp_path):
    steps = [bash("pytest -q", "THOUGHT: run the tests."), ran("1 failed\n", exit_code=1),
             bash(SUBMIT, "THOUGHT: done."), reply("diff --git a/x b/x\n")]
    text = show(write_fake_run(tmp_path, steps, submission="diff --git a/x b/x\n"))
    assert "Hidden-test result: not shown." in text
    assert "--- messages[2]: exit code 1\nTHOUGHT: run the tests.\n$ pytest -q\n1 failed" in text
    assert text.endswith("=== Submission (19 characters)\ndiff --git a/x b/x\n")
