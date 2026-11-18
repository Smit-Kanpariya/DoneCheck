"""The donecheck command.

    donecheck verify RUN.traj.json          accept or flag one submitted run, with its evidence
    donecheck reproduce                     download the data, rebuild every result and compare
    donecheck labels check|report           the hand-labelling sheet; its analysis once complete
    donecheck judges prompts|run|score SPLIT, judges freeze MODEL   the AI-judge baselines
    donecheck fetch building|final DIR      download the public run files
    donecheck evaluate building|final       rebuild eval/results/ (final runs once)
"""

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import typer

from donecheck import evaluate as evaluation
from donecheck import fetch as fetching
from donecheck import judging, labels
from donecheck.checks import check_run
from donecheck.loader import load_run
from donecheck.model import score

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Is an agent's 'done' backed by evidence?")

PLAIN = {
    "steps": lambda v: f"{v} steps",
    "format_errors": lambda v: f"{v} replies held no runnable command",
    "timeouts": lambda v: f"{v} commands timed out",
    "submission_empty": lambda v: "the submitted change is empty" if v else "the submitted change is not empty",
    "diff_files": lambda v: f"the change touches {v} files",
    "diff_test_files": lambda v: f"{v} test files in the change",
    "diff_scratch_files": lambda v: f"{v} leftover scratch or backup files in the change",
    "code_edits": lambda v: f"{v} code edits found",
    "unknown_edits": lambda v: f"{v} edits whose file could not be read off the command",
    "checks": lambda v: f"{v} checks (tests or scripts) in all",
    "tests": lambda v: f"{v} repository test runs in all",
    "checks_after_last_edit": lambda v: f"{v} checks after the last code edit",
    "tests_after_last_edit": lambda v: f"{v} repository test runs after the last code edit",
    "passed_after_last_edit": lambda v: ("a check passed after the last code edit" if v
                                         else "no check passed after the last code edit"),
    "last_check_result": lambda v: f"the last check after the last code edit: {v}",
    "last_test_result": lambda v: f"the last repository test run: {v}",
    "fail_then_pass": lambda v: ("a check that failed before the last edit passed after it" if v
                                 else "no failing check was seen to pass after the fix"),
    "edited_files": lambda v: f"{v} existing files changed",
    "edited_unopened": lambda v: f"{v} changed files were never opened before being edited",
    "claims": lambda v: f"{v} claims in the notes",
    "claims_supported": lambda v: f"{v} claims backed by a passing check",
    "claims_unsupported": lambda v: f"{v} claims with no check after the edit",
    "claims_contradicted": lambda v: f"{v} claims contradicted by the record",
    "final_claims_contradicted": lambda v: f"{v} claims after the last edit contradicted by the record",
    "gave_up": lambda v: "the last note says the agent could not finish" if v else "the last note does not give up",
}


def explain(artifact: dict, facts: dict, top: int = 4) -> dict:
    """The verdict for one run's facts, with the facts that pushed the score most each way."""
    p, push = score(artifact, facts)
    by_fact = defaultdict(float)
    for feature, value in push.items():
        by_fact[feature.split("=")[0]] += value  # the one-hot parts of a fact add up
    ranked = sorted(by_fact.items(), key=lambda kv: -abs(kv[1]))
    return {"flag": p >= artifact["threshold"], "score": p, "threshold": artifact["threshold"],
            "towards_flag": [PLAIN[f](facts[f]) for f, v in ranked if v > 0][:top],
            "towards_accept": [PLAIN[f](facts[f]) for f, v in ranked if v < 0][:top]}


def load_any(run_file: Path):
    """Load a run file from anywhere. The loader reads the model name from the folder layout
    <date>_mini-v<version>_<model>/trajs/, so a file outside it is read through a link in that layout."""
    parent = run_file.resolve().parent
    if parent.name == "trajs" and "_mini-v" in parent.parent.name:
        return load_run(run_file, labels={})
    with tempfile.TemporaryDirectory() as tmp:
        link = Path(tmp, "00000000_mini-v0_unknown", "trajs", run_file.name)
        link.parent.mkdir(parents=True)
        link.symlink_to(run_file.resolve())
        return load_run(link, labels={})


@app.command()
def verify(run_file: Path, model: Path = Path(evaluation.MODEL_PATH), top: int = 4) -> None:
    """Accept or flag one run's 'done'. Exit code: 0 accept, 1 flag, 2 no done claim, 3 cannot check."""
    try:
        run = load_any(run_file)
        artifact = json.loads(model.read_text())
        if not run.submitted:
            typer.echo(f"No done claim to check: the run ended with {run.exit_status}, not a submission.")
            raise typer.Exit(2)
        v = explain(artifact, check_run(run), top)
    except (OSError, ValueError, KeyError) as err:  # missing file, bad JSON, wrong shape, wrong model file
        typer.echo(f"Cannot check {run_file}: {err}", err=True)
        raise typer.Exit(3) from None
    typer.echo(f"DoneCheck: {'FLAG' if v['flag'] else 'ACCEPT'}  score {v['score']:.2f}, threshold {v['threshold']:.2f}")
    typer.echo("The score ranks how likely this 'done' is false. It is a fitted logistic-regression output, not a\n"
               "calibrated probability: on models it was not fitted on, its calibration drifts (docs/report.md).")
    for label, key in (("Towards flag", "towards_flag"), ("Towards accept", "towards_accept")):
        if v[key]:
            typer.echo(f"{label}:\n" + "\n".join(f"  - {line}" for line in v[key]))
    typer.echo("Limits: the checks read commands and outputs, not the code; a script counts as passed when it\n"
               "exits cleanly; claims are found by keywords.")
    raise typer.Exit(1 if v["flag"] else 0)


@app.command()
def reproduce(skip_download: bool = False) -> None:
    """Download the data, recompute every result into eval/reproduced/ and compare with the committed ones.

    The building results and model are rebuilt from scratch; GLM-4.5 is rescored with the committed,
    frozen model (the committed final test itself is never rerun in place)."""
    if not skip_download:
        fetching.main("building", "data")
        fetching.main("final", "data_final")
    committed_building = json.loads(Path(evaluation.RESULTS, "building.json").read_text())
    committed_final = json.loads(Path(evaluation.RESULTS, "final_glm45.json").read_text())
    shutil.rmtree("eval/reproduced", ignore_errors=True)  # its own output from an earlier run
    saved = evaluation.RESULTS, evaluation.MODEL_PATH
    try:
        evaluation.RESULTS, evaluation.MODEL_PATH = "eval/reproduced", "eval/reproduced/donecheck-v1.json"
        building = evaluation.building()
        evaluation.MODEL_PATH = saved[1]  # GLM-4.5 is scored with the committed, frozen model
        final = evaluation.final()
    finally:
        evaluation.RESULTS, evaluation.MODEL_PATH = saved
    pairs = {
        "building: mean leave-one-model-out AUROC": (committed_building["donecheck_leave_one_model_out"]["mean_fold_auroc"],
                                                     building["donecheck_leave_one_model_out"]["mean_fold_auroc"]),
        "building: threshold": (committed_building["final_model"]["threshold"], building["final_model"]["threshold"]),
        "GLM-4.5: catch at the frozen threshold": (committed_final["donecheck"]["fixed_threshold"]["catch"],
                                                   final["donecheck"]["fixed_threshold"]["catch"]),
        "GLM-4.5: false alarms at the frozen threshold": (committed_final["donecheck"]["fixed_threshold"]["false_alarm"],
                                                          final["donecheck"]["fixed_threshold"]["false_alarm"]),
        "GLM-4.5: AUROC": (committed_final["donecheck"]["auroc"], final["donecheck"]["auroc"]),
    }
    same = True
    for name, (old, new) in pairs.items():
        ok = abs(old - new) < 1e-9
        same &= ok
        typer.echo(f"{'same' if ok else 'DIFFERENT'}  {name}: committed {old:.6f}, reproduced {new:.6f}")
    raise typer.Exit(0 if same else 1)


@app.command()
def fetch(which: str, dest: Path) -> None:
    """Download the 'building' models or the 'final' test model into DEST."""
    fetching.main(which, str(dest))


@app.command()
def evaluate(which: str) -> None:
    """Rebuild the 'building' results, 'freeze' them, or run the one-time 'final' test."""
    {"building": evaluation.building, "freeze": evaluation.freeze, "final": evaluation.final}[which]()
    typer.echo(f"wrote {evaluation.RESULTS}/")


labels_app = typer.Typer(no_args_is_help=True, help="The hand-labelling study (human labels only).")
judges_app = typer.Typer(no_args_is_help=True, help="The AI-judge baselines (frozen prompts judges-v1).")
app.add_typer(labels_app, name="labels")
app.add_typer(judges_app, name="judges")


@labels_app.command("check")
def labels_check() -> None:
    """Progress and format of labels/claims.tsv. Reads only the label sheets. Exit 1 on a problem."""
    status = labels.check()
    typer.echo(f"{status['rows']} rows; runs with labels: {status['labelled_runs']}; complete: {status['complete']}")
    for problem in status["problems"]:
        typer.echo(f"  {problem}")
    raise typer.Exit(1 if status["problems"] else 0)


@labels_app.command("report")
def labels_report() -> None:
    """Once all 100 runs are labelled: agreement with the rules, overclaims, self-agreement."""
    typer.echo(json.dumps(labels.report(), indent=1))


@judges_app.command("prompts")
def judges_prompts(split: str) -> None:
    """Write the frozen prompts: 'building' (the 200-run sample) or 'final' (every GLM-4.5 run)."""
    evaluation.judge_prompts() if split == "building" else judging.write_final_prompts()


@judges_app.command("run")
def judges_run(split: str, judge: str, endpoint: str, model: str, local: bool = False,
               usd_per_mtok_in: float | None = None, usd_per_mtok_out: float | None = None,
               max_usd: float | None = None, work_minutes: float | None = None, rest_minutes: float = 0) -> None:
    """Send one judge's prompts to an OpenAI-compatible endpoint (key from JUDGE_API_KEY). Needs --local
    for a free local server, or both prices and --max-usd. Resumes where it stopped. --work-minutes and
    --rest-minutes run it in chunks with cool-down pauses."""
    typer.echo(json.dumps(judging.run(split, judge, endpoint, model, local, usd_per_mtok_in, usd_per_mtok_out, max_usd,
                                      work_minutes=work_minutes, rest_minutes=rest_minutes)))


@judges_app.command("freeze")
def judges_freeze(model: str, endpoint: str = "http://localhost:11434") -> None:
    """Before GLM-4.5: record the local judge model and the thresholds from the building sample."""
    typer.echo(json.dumps(judging.freeze(model, endpoint), indent=1))


@judges_app.command("score")
def judges_score(split: str) -> None:
    """Score each judge that has answers next to DoneCheck on the same runs."""
    typer.echo(json.dumps(judging.score(split), indent=1))
