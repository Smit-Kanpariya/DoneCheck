"""Tests for the donecheck command, on small fake run files and the committed model."""

from typer.testing import CliRunner

from donecheck.cli import app
from test_loader import SUBMIT, bash, ran, reply, write_fake_run

runner = CliRunner()


def test_verify_flags_an_empty_change(tmp_path):
    path = write_fake_run(tmp_path, steps=[bash("sed -i 's/a/b/' pkg/x.py"), ran(""),
                                          bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "THOUGHT: Done."), reply("")])
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 1, result.output
    assert result.output.startswith("DoneCheck: FLAG") and "the submitted change is empty" in result.output


def test_verify_accepts_a_fix_with_a_passing_rerun(tmp_path):
    diff = "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    path = write_fake_run(tmp_path, submission=diff, steps=[
        bash("cat pkg/x.py"), ran("a\n"),
        bash("cat > repro.py <<'EOF' && python repro.py\nimport pkg.x\nEOF"), ran("Traceback (most recent call last):\nAssertionError\n", 1),
        bash("sed -i 's/a/b/' pkg/x.py"), ran(""),
        bash("python repro.py"), ran("ok\n"),
        bash("python -m pytest -q tests/test_x.py"), ran("5 passed in 0.20s\n"),
        bash(SUBMIT, "THOUGHT: The reproduction passes now."), reply(diff)])
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("DoneCheck: ACCEPT") and "Towards accept:\n  - " in result.output


def test_verify_has_nothing_to_check_without_a_submission(tmp_path):
    path = write_fake_run(tmp_path, exit_status="LimitsExceeded", steps=[bash("ls"), ran("a.py\n"), reply("")])
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 2 and "No done claim to check" in result.output


def test_verify_cannot_check_a_missing_file_or_a_bad_model(tmp_path):
    assert runner.invoke(app, ["verify", str(tmp_path / "missing.traj.json")]).exit_code == 3
    bad_model = tmp_path / "model.json"
    bad_model.write_text("{", encoding="utf-8")
    result = runner.invoke(app, ["verify", str(write_fake_run(tmp_path)), "--model", str(bad_model)])
    assert result.exit_code == 3 and "Cannot check" in result.output


def test_verify_reads_a_run_file_outside_the_folder_layout(tmp_path):
    loose = tmp_path / "loose.json"
    loose.write_text(write_fake_run(tmp_path, steps=[bash("sed -i 's/a/b/' pkg/x.py"), ran(""),
                                                     bash(SUBMIT, "THOUGHT: Done."), reply("")]).read_text())
    result = runner.invoke(app, ["verify", str(loose)])
    assert result.exit_code == 1 and result.output.startswith("DoneCheck: FLAG"), result.output
