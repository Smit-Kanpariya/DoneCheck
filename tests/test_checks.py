"""Tests for the evidence checks (donecheck.checks). No run from labels/sample_100.tsv is opened here
by name: those runs are the hand-checked test set for later."""

from donecheck.checks import check_of, check_run, diff_files, edits_of, is_scratch, result_of, split_command
from donecheck.loader import Run, Step, iter_runs, load_run
from test_loader import DATA, SUBMIT, needs_data


def step(command, output="", exit_code=0, note="THOUGHT: next.", outcome="ran"):
    return Step(0, note, outcome, command, exit_code if outcome == "ran" else None, output, 0)


def edits(command, scripts=None):
    return [e for part in split_command(command) for e in edits_of(part, "/testbed", scripts or {})]


def checks_in(command):
    return [check_of(part, "/testbed") for part in split_command(command)]


def test_split_command_keeps_quotes_heredocs_and_substitutions_together():
    parts = split_command("cd /testbed && grep -n 'a && b | c' x.py | head -5; echo $(ls | wc -l)")
    assert [p.text for p in parts] == ["cd /testbed", "grep -n 'a && b | c' x.py", "head -5", "echo $(ls | wc -l)"]
    parts = split_command("cat > repro.py <<'EOF' && python repro.py\nimport x\nprint(x.f())\nEOF\npytest -q t.py")
    assert [(p.text, p.body) for p in parts] == [("cat > repro.py <<'EOF'", "import x\nprint(x.f())"),
                                                ("python repro.py", ""), ("pytest -q t.py", "")]
    assert [p.text for p in split_command("sed -i 's/a/b/' \\\n  x.py")] == ["sed -i 's/a/b/'   x.py"]


def test_edits_of_reads_each_way_of_changing_a_file():
    assert edits("sed -i 's/a/b/' pkg/x.py pkg/y.py") == [("pkg/x.py", "write"), ("pkg/y.py", "write")]
    assert edits("sed -i.bak -e 's/a/b/' -e 's/c/d/' x.py") == [("x.py", "write")]
    assert edits("sed -n '1,20p' x.py && perl -pi -e 's/a/b/' y.py") == [("y.py", "write")]
    assert edits("echo hi > a.txt 2>&1; cat b.py >> c.py; ls > /dev/null") == [("a.txt", "write"), ("c.py", "write")]
    assert edits("python - <<'PY'\nfrom pathlib import Path\np = Path('pkg/x.py')\n"
                 "p.write_text(p.read_text().replace('a', 'b'))\nPY") == [("pkg/x.py", "write")]
    assert edits("python - <<'PY'\nfor app in apps:\n    open(f'pkg/{app}/admin.py', 'w').write(s)\nPY") == [
        ("pkg/*/admin.py", "write")]
    assert edits("python -c \"open('o.txt', 'w').write('1')\"") == [("o.txt", "write")]
    assert edits("python - <<'PY'\nimport shutil\nshutil.move(src, dst)\nPY") == [("?", "write")]
    assert edits("rm -f repro.py && mv a.py b.py") == [("repro.py", "delete"), ("b.py", "write"), ("a.py", "delete")]
    assert edits("git checkout -- pkg/x.py && git stash && git diff") == [("pkg/x.py", "revert"), ("?", "revert")]
    assert edits("python edit.py", scripts={"edit.py": "open('pkg/x.py', 'w').write(s)"}) == [("pkg/x.py", "write")]
    assert edits("python - <<'PY'\nprint(open('pkg/x.py').read())\nPY") == []


def test_check_of_tells_tests_from_scripts():
    assert checks_in("cd /testbed && python -m pytest -q tests/test_x.py | tail -5") == [
        None, ("test", "pytest tests/test_x.py"), None]
    assert checks_in("timeout 60 pytest tests/a.py::test_b -x") == [("test", "pytest tests/a.py::test_b")]
    assert checks_in("python tests/runtests.py admin_views") == [("test", "tests/runtests.py admin_views")]
    assert checks_in("python repro.py") == [("script", "repro.py")]
    assert checks_in("python -c 'import pkg; print(pkg.f(1))'") == [("script", "import pkg; print(pkg.f(1))")]
    assert checks_in("python - <<'PY'\nprint(open('pkg/x.py').read())\nPY") == [None]  # only looks at a file
    assert checks_in("pip install -e . && python -m pip list && python --version && grep -rn pytest .") == [None] * 4


def test_result_of_reads_runner_summaries_before_exit_codes():
    assert result_of(step("pytest -q | tail -3", "F.\n1 failed, 3 passed in 0.12s\n")) == "failed"  # tail's exit code
    assert result_of(step("pytest", "===== 4 passed in 0.31s =====\n")) == "passed"
    assert result_of(step("pytest", "no tests ran in 0.01s\n", exit_code=5)) == "error"
    assert result_of(step("python tests/runtests.py x", "Ran 3 tests in 0.010s\n\nOK\n")) == "passed"
    assert result_of(step("python tests/runtests.py x", "Ran 3 tests in 0.010s\n\nFAILED (failures=1)\n", 1)) == "failed"
    assert result_of(step("bin/test", "tests finished: 5 passed, 1 failed, in 0.32 seconds\n", 1)) == "failed"
    assert result_of(step("python r.py", "Traceback (most recent call last):\nValueError: x\n", 1)) == "failed"
    assert result_of(step("python r.py", "Traceback (most recent call last):\nModuleNotFoundError: No module named 'erfa'\n",
                          1)) == "error"
    assert result_of(step("python r.py", "0.5\n")) == "passed"
    assert result_of(step("python slow.py", outcome="timeout")) == "timeout"


DIFF = ("diff --git a/pkg/x.py b/pkg/x.py\nindex 1..2 100644\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/reproduce.py b/reproduce.py\nnew file mode 100644\n--- /dev/null\n+++ b/reproduce.py\n"
        "diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n")


def test_diff_files_and_scratch_files():
    files = diff_files(DIFF)
    assert files == {"pkg/x.py": "modified", "reproduce.py": "new", "tests/test_x.py": "new"}
    assert [is_scratch(p, s) for p, s in files.items()] == [False, True, False]
    assert is_scratch("pkg/x.py.bak") and is_scratch("pkg/debug_x.py") and not is_scratch("pkg/new_module.py")


def fake_run(steps, submission):
    submit = Step(0, "THOUGHT: All tests pass. Submitting.", "submitted", SUBMIT, None, submission, 0)
    return Run("x__x-1", "m", "1.7.0", "Submitted", True, submission, "newer", None, tuple(steps) + (submit,), 0.25, 8)


def test_check_run_on_a_fake_run():
    fix = "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-return 1\n+return 2\n"
    run = fake_run([
        step("cat pkg/x.py", "def f(): return 1\n"),
        step("cat > repro.py <<'EOF' && python repro.py\nfrom pkg.x import f\nassert f() == 2\nEOF",
             "Traceback (most recent call last):\nAssertionError\n", 1),
        step("sed -i 's/return 1/return 2/' pkg/x.py"),
        step("python repro.py"),
        step("python -m pytest -q tests | tail -2", "1 failed, 9 passed in 0.50s\n",
             note="THOUGHT: The reproduction passes now and the tests pass."),
        step("rm repro.py"),  # removing a scratch file is not a code edit
    ], fix)
    f = check_run(run)
    assert {k: f[k] for k in ("steps", "cost", "api_calls", "code_edits", "checks", "tests", "checks_after_last_edit",
                              "tests_after_last_edit", "passed_after_last_edit", "last_check_result",
                              "last_test_result", "fail_then_pass", "edited_files", "edited_unopened")} == {
        "steps": 7, "cost": 0.25, "api_calls": 8, "code_edits": 1, "checks": 3, "tests": 1,
        "checks_after_last_edit": 2, "tests_after_last_edit": 1, "passed_after_last_edit": True,
        "last_check_result": "failed", "last_test_result": "failed", "fail_then_pass": True,
        "edited_files": 1, "edited_unopened": 0}
    # "the tests pass" after a passing rerun is supported; the submit note's "All tests pass." follows a failed run
    assert (f["claims"], f["claims_supported"], f["claims_contradicted"]) == (2, 1, 1)
    assert f["final_claims_contradicted"] == 1  # both claims come after the last edit (step 3)
    assert (f["submission_empty"], f["diff_files"], f["diff_scratch_files"], f["gave_up"]) == (False, 1, 0, False)

    empty = check_run(fake_run([step("python repro.py")], ""))
    assert (empty["submission_empty"], empty["code_edits"], empty["checks_after_last_edit"]) == (True, 0, 1)
    assert empty["claims"] == 0  # claims only count after a code edit


def test_only_claims_after_the_last_edit_are_final():
    fix = "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    f = check_run(fake_run([
        step("sed -i 's/a/c/' pkg/x.py"),
        step("pytest -q", "1 failed in 0.10s\n", 1),
        step("sed -i 's/c/b/' pkg/x.py", note="THOUGHT: The tests pass."),  # false, but the agent kept working
        step("pytest -q", "2 passed in 0.10s\n"),
    ], fix))
    assert (f["claims"], f["claims_contradicted"], f["claims_supported"], f["final_claims_contradicted"]) == (2, 1, 1, 0)


@needs_data
def test_checks_on_a_day0_run():
    # gpt-5-nano edits the setting with sed -i, then appends a note to the docs with printf >>, after only
    # grepping the whole repository; it never runs a test or a script.
    f = check_run(load_run(DATA / "20250807_mini-v1.7.0_gpt-5-nano/trajs/django__django-10914.traj.json", labels={}))
    assert {k: f[k] for k in ("code_edits", "checks", "edited_files", "edited_unopened", "last_check_result")} == {
        "code_edits": 2, "checks": 0, "edited_files": 2, "edited_unopened": 2, "last_check_result": "none"}


@needs_data
def test_checks_run_on_every_downloaded_run():
    results = [check_run(run) for run in iter_runs(DATA)]
    assert len(results) == 2500
    assert all(f["cost"] is not None for f in results)
