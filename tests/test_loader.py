"""Tests for the loader. Expected values for real runs were read from the raw run files with jq."""

import json
from collections import Counter
from pathlib import Path

import pytest

from donecheck.loader import Run, RunFileError, Step, iter_runs, load_labels, load_run

DATA = Path(__file__).resolve().parent.parent / "data"
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="run data not downloaded (data/ is never committed)")

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
OLDER_SUBMIT = "echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached"


def real_run(folder, bug_id):
    return load_run(DATA / folder / "trajs" / f"{bug_id}.traj.json", load_labels(DATA / folder))


# --- Facts about the whole run ---------------------------------------------------------

# case: (model folder, bug id, expected fields, submission length, how the submission starts)
# Chosen so that a wrong rule fails: an empty change that was submitted, a crash whose
# submission is an error message, a run stopped by the limit, and both instruction wordings.
REAL_RUNS = {
    "older wording, fixed": (
        "20250807_mini-v1.7.0_gpt-oss-120b", "sympy__sympy-20154",
        dict(model="gpt-oss-120b", agent_version="1.7.0", exit_status="Submitted",
             submitted=True, wording="older", resolved=True),
        2104, "diff --git",
    ),
    "newer wording, fixed": (
        "20250807_mini-v1.7.0_gpt-5-nano", "django__django-10914",
        dict(model="gpt-5-nano", agent_version="1.7.0", exit_status="Submitted",
             submitted=True, wording="newer", resolved=True),
        1178, "diff --git",
    ),
    "submitted an empty change": (
        "20251209_mini-v1.17.2_devstral-2512", "django__django-15569",
        dict(model="devstral-2512", agent_version="1.17.1", exit_status="Submitted",
             submitted=True, wording="newer", resolved=False),
        0, "",
    ),
    "stopped by the limit": (
        "20250807_mini-v1.7.0_kimi-k2-instruct", "psf__requests-6028",
        dict(model="kimi-k2-instruct", agent_version="1.7.0", exit_status="LimitsExceeded",
             submitted=False, wording="older", resolved=False),
        0, "",
    ),
    "crashed": (
        "20250807_mini-v1.7.0_gpt-oss-120b", "pylint-dev__pylint-6528",
        dict(model="gpt-oss-120b", agent_version="1.7.0", exit_status="APIError",
             submitted=False, wording="older", resolved=False),
        1984, "litellm.APIError",
    ),
}


@needs_data
@pytest.mark.parametrize("case", list(REAL_RUNS))
def test_real_run(case):
    folder, bug_id, expected, submission_chars, submission_start = REAL_RUNS[case]
    run = real_run(folder, bug_id)
    assert run.instance_id == bug_id
    for field, value in expected.items():
        assert getattr(run, field) == value, field
    assert len(run.submission) == submission_chars
    assert run.submission.startswith(submission_start)


# --- The steps -----------------------------------------------------------------------------

@needs_data
def test_steps_with_every_kind_of_reply():
    # A cut output, timeouts, a format error, failing commands, then the submit.
    run = real_run("20250807_mini-v1.7.0_gpt-oss-120b", "sympy__sympy-20154")
    assert len(run.steps) == 23
    assert Counter(s.outcome for s in run.steps) == {"ran": 16, "timeout": 5, "format_error": 1, "submitted": 1}
    step = {s.message: s for s in run.steps}
    assert (step[2].outcome, step[2].command, step[2].exit_code, step[2].elided_chars) == ("ran", "ls -R", 0, 24262)
    assert (step[12].exit_code, step[12].output) == (127, "bash: line 1: apply_patch: command not found\n")
    assert (step[26].outcome, step[26].command, step[26].exit_code, step[26].output) == (
        "timeout", "python -m pytest -q", None, "")
    assert (step[38].outcome, step[38].command) == ("format_error", None)
    last = run.steps[-1]
    assert (last.message, last.outcome, last.command) == (46, "submitted", OLDER_SUBMIT)
    assert last.output == run.submission
    assert all("```bash" not in s.note for s in run.steps)


@needs_data
@pytest.mark.parametrize("folder, bug_id, n_steps, outcomes, last_step", [
    ("20250807_mini-v1.7.0_gpt-5-nano", "django__django-10914", 12, {"ran": 11, "submitted": 1}, (24, "submitted")),
    ("20251209_mini-v1.17.2_devstral-2512", "django__django-15569", 2, {"format_error": 1, "submitted": 1}, (4, "submitted")),
    ("20250807_mini-v1.7.0_kimi-k2-instruct", "psf__requests-6028", 167, {"format_error": 1, "ran": 166}, (334, "ran")),
    ("20250807_mini-v1.7.0_gpt-oss-120b", "pylint-dev__pylint-6528", 5, {"ran": 5}, (10, "ran")),
], ids=["plain", "empty change", "stopped by the limit", "crashed"])
def test_step_counts(folder, bug_id, n_steps, outcomes, last_step):
    run = real_run(folder, bug_id)
    assert len(run.steps) == n_steps
    assert Counter(s.outcome for s in run.steps) == outcomes
    assert (run.steps[-1].message, run.steps[-1].outcome) == last_step


@needs_data
def test_step_details():
    nano = real_run("20250807_mini-v1.7.0_gpt-5-nano", "django__django-10914")
    assert (nano.steps[0].command, nano.steps[0].exit_code) == ('grep -R "FILE_UPLOAD_PERMISSIONS" -n .', 0)
    assert nano.steps[-1].command == SUBMIT
    devstral = real_run("20251209_mini-v1.17.2_devstral-2512", "django__django-15569")
    assert (devstral.steps[0].command, devstral.steps[1].command, devstral.steps[1].output) == (None, SUBMIT, "")
    crashed = real_run("20250807_mini-v1.7.0_gpt-oss-120b", "pylint-dev__pylint-6528")
    assert crashed.steps[0].elided_chars == 67261


@needs_data
def test_every_downloaded_run_loads():
    runs = list(iter_runs(DATA))
    assert len(runs) == 2500
    assert sum(run.submitted for run in runs) == sum(s.outcome == "submitted" for run in runs for s in run.steps)


# --- Small fake run files: no downloaded data needed ---------------------------------------

TASK = {"role": "user", "content": "<pr_description>\nA bug.\n</pr_description>\n"
                                   "Submit with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."}
CUT_REPLY = ("<returncode>0</returncode>\n<warning>\nThe output of your last command was too long.\n</warning>"
             "<output_head>\n" + "a" * 5000 + "\n</output_head>\n<elided_chars>\n123 characters elided\n"
             "</elided_chars>\n<output_tail>\n" + "b" * 5000 + "\n</output_tail>")
TIMEOUT_REPLY = ("The last command <command>sleep 999</command> timed out and has been killed.\n"
                 "The output of the command was:\n <output>\npartial\n</output>\n"
                 "Please try another command and make sure to avoid those requiring interactive input.")
FORMAT_REPLY = ("Please always provide EXACTLY ONE action in triple backticks, found 0 actions.\n\n"
                "Please format your action in triple backticks as shown in <response_example>.")


def agent(text):
    return {"role": "assistant", "content": text}


def reply(text):
    return {"role": "user", "content": text}


def bash(command, note="THOUGHT: next step."):
    return agent(f"{note}\n\n```bash\n{command}\n```")


def ran(output, exit_code=0):
    return reply(f"<returncode>{exit_code}</returncode>\n<output>\n{output}</output>")


def write_fake_run(tmp_path, steps=None, exit_status="Submitted", submission="", run_format="mini-swe-agent-1",
                   task=TASK, drop_info_key=None, folder="20250101_mini-v1.7.0_fake-model", instance_id="fake__fake-1"):
    """Write a small run file in the text shape and return its path.

    steps: the messages after the task; by default one submit that hands in `submission`.
    """
    if steps is None:
        steps = [bash(SUBMIT, "THOUGHT: done."), reply(submission)]
    info = {"exit_status": exit_status, "submission": submission, "mini_version": "1.7.0"}
    if drop_info_key:
        del info[drop_info_key]
    run = {"trajectory_format": run_format, "instance_id": instance_id, "info": info,
           "messages": [{"role": "system", "content": "You are a helpful assistant."}, task, *steps]}
    trajs = tmp_path / folder / "trajs"
    trajs.mkdir(parents=True, exist_ok=True)
    path = trajs / f"{instance_id}.traj.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    return path


def test_fake_run_loads_and_a_missing_result_is_none(tmp_path):
    run = load_run(write_fake_run(tmp_path), labels={})
    assert run == Run(instance_id="fake__fake-1", model="fake-model", agent_version="1.7.0",
                      exit_status="Submitted", submitted=True, submission="", wording="newer", resolved=None,
                      steps=(Step(2, "THOUGHT: done.", "submitted", SUBMIT, None, "", 0),), task="A bug.")


@pytest.mark.parametrize("bad", [
    {"run_format": "mini-swe-agent-1.1"},
    {"task": reply("<pr_description>\nA bug.\n</pr_description>\nNo submit word here.")},
    {"task": reply("<pr_description>\nA bug.\n</pr_description>\n"
                   "Use MINI_SWE_AGENT_FINAL_OUTPUT or COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.")},
    {"drop_info_key": "exit_status"},
], ids=["wrong format", "no submit word", "both submit words", "missing exit_status"])
def test_bad_run_file_raises(tmp_path, bad):
    with pytest.raises(RunFileError):
        load_run(write_fake_run(tmp_path, **bad), labels={})


def test_load_labels(tmp_path):
    results = {"a__a-1": {"cost": 0.1, "api_calls": 3, "resolved": True},
               "b__b-2": {"cost": 0.2, "api_calls": 5, "resolved": False}}
    (tmp_path / "per_instance_details.json").write_text(json.dumps(results), encoding="utf-8")
    assert load_labels(tmp_path) == {"a__a-1": True, "b__b-2": False}


def test_results_file_must_say_true_or_false(tmp_path):
    (tmp_path / "per_instance_details.json").write_text(json.dumps({"a__a-1": {"resolved": "yes"}}), encoding="utf-8")
    with pytest.raises(RunFileError):
        load_labels(tmp_path)


def test_every_kind_of_reply(tmp_path):
    steps = [
        bash("pytest -q", "THOUGHT: run the tests."), ran("1 failed\n", exit_code=1),
        bash("ls -R"), reply(CUT_REPLY),
        bash("sleep 999"), reply(TIMEOUT_REPLY),
        agent("All tests pass now."), reply(FORMAT_REPLY),
        bash(SUBMIT, "THOUGHT: done."), reply("diff --git a/x b/x\n"),
    ]
    run = load_run(write_fake_run(tmp_path, steps, submission="diff --git a/x b/x\n"), labels={})
    assert run.steps == (
        Step(2, "THOUGHT: run the tests.", "ran", "pytest -q", 1, "1 failed\n", 0),
        Step(4, "THOUGHT: next step.", "ran", "ls -R", 0,
             "a" * 5000 + "\n[... 123 characters elided ...]\n" + "b" * 5000, 123),
        Step(6, "THOUGHT: next step.", "timeout", "sleep 999", None, "partial", 0),
        Step(8, "All tests pass now.", "format_error", None, None, "", 0),
        Step(10, "THOUGHT: done.", "submitted", SUBMIT, None, "diff --git a/x b/x\n", 0),
    )


def test_run_stopped_by_the_limit(tmp_path):
    steps = [bash("ls"), ran("setup.py\n"), reply("")]
    run = load_run(write_fake_run(tmp_path, steps, exit_status="LimitsExceeded"), labels={})
    assert not run.submitted
    assert [(s.outcome, s.output) for s in run.steps] == [("ran", "setup.py\n")]


def test_run_that_ends_on_an_agent_message(tmp_path):
    path = write_fake_run(tmp_path, [bash("ls")], exit_status="APIError", submission="litellm.APIError: down")
    assert load_run(path, labels={}).steps == (Step(2, "THOUGHT: next step.", "no_reply", "ls", None, "", 0),)


@pytest.mark.parametrize("steps, exit_status, submission", [
    ([bash("ls"), reply("something the framework never writes")], "LimitsExceeded", ""),
    ([bash(SUBMIT), reply("diff --git a/x b/x\n")], "Submitted", "a different change"),
    ([bash("ls"), ran("\n"), reply("not empty")], "LimitsExceeded", ""),
    ([agent("```bash\nls\n```\n```bash\npwd\n```"), ran("\n")], "LimitsExceeded", ""),
], ids=["unknown reply", "last message is not the submission", "extra user message", "ran with two blocks"])
def test_bad_steps_raise(tmp_path, steps, exit_status, submission):
    with pytest.raises(RunFileError):
        load_run(write_fake_run(tmp_path, steps, exit_status=exit_status, submission=submission), labels={})


def test_iter_runs_reads_every_model_folder(tmp_path):
    for folder, bug_id, fixed in [("20250101_mini-v1.7.0_model-a", "a__a-1", True),
                                  ("20250102_mini-v1.17.1_model-b", "b__b-1", False)]:
        write_fake_run(tmp_path, folder=folder, instance_id=bug_id)
        results = {bug_id: {"resolved": fixed}}
        (tmp_path / folder / "per_instance_details.json").write_text(json.dumps(results), encoding="utf-8")
    (tmp_path / "MANIFEST.tsv").write_text("not a model folder\n", encoding="utf-8")
    runs = [(r.model, r.instance_id, r.resolved) for r in iter_runs(tmp_path)]
    assert runs == [("model-a", "a__a-1", True), ("model-b", "b__b-1", False)]
