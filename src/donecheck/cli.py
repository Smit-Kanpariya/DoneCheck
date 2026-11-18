"""The donecheck command.

    donecheck verify RUN.traj.json          accept or flag one submitted run, with its evidence
    donecheck fetch building|final DIR      download the public run files
    donecheck evaluate building|final       rebuild eval/results/ (final runs once)
    donecheck judge-prompts                 write the AI-judge prompts for the 200-run sample
"""

import json
from collections import defaultdict
from pathlib import Path

import typer

from donecheck import evaluate as evaluation
from donecheck import fetch as fetching
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


@app.command()
def verify(run_file: Path, model: Path = Path(evaluation.MODEL_PATH), top: int = 4) -> None:
    """Accept or flag one run's 'done'. Exit code: 0 accept, 1 flag, 2 no done claim to check."""
    run = load_run(run_file, labels={})
    if not run.submitted:
        typer.echo(f"No done claim to check: the run ended with {run.exit_status}, not a submission.")
        raise typer.Exit(2)
    v = explain(json.loads(model.read_text()), check_run(run), top)
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
    frozen_model = evaluation.MODEL_PATH
    evaluation.RESULTS, evaluation.MODEL_PATH = "eval/reproduced", "eval/reproduced/donecheck-v1.json"
    building = evaluation.building()
    evaluation.MODEL_PATH = frozen_model
    final = evaluation.final()
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


@app.command()
def judge_prompts() -> None:
    """Draw the 200-run judge sample and write both judges' prompts to eval/judges/building/."""
    evaluation.judge_prompts()
