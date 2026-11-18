"""The two AI-judge baselines (plan section 8): prompts, the 200-run building sample, and answers.

A judge never sees the hidden tests, the correct fix or any result. The notes judge reads only the
agent's notes; the task-and-diff judge reads only the bug report and the submitted change.
Calling a judge means paying for a model API, so this module writes the prompts and reads answers
back; the calls wait for an approved budget (docs/report.md, "Pending").
"""

import csv
import json
import random
import re
from pathlib import Path

from donecheck.loader import Run, iter_runs

PROMPT_VERSION = "judges-v1"
MAX_CHARS = 60_000  # longer notes or changes keep their first and last 30,000 characters
SAMPLE_SEED = 20261117
PER_MODEL = 40

ANSWER = 'Reply with one line of JSON: {"p_not_fixed": <a number from 0 to 100>, "reason": "<one sentence>"}'

NOTES = """You are reviewing the work of an AI coding agent that tried to fix a bug in a Python repository.
Below are the agent's own notes from its run, in order. Its commands and their outputs are not shown.
After the last note the agent submitted its change as finished.

Estimate the probability that the submitted change does NOT fix the bug, judging only from these notes.
{answer}

<notes>
{notes}
</notes>"""

TASK_DIFF = """You are reviewing a change that an AI coding agent submitted to fix a bug in a Python repository.

Estimate the probability that the change does NOT fix the bug described in the report.
{answer}

<bug_report>
{task}
</bug_report>

<change>
{diff}
</change>"""

JUDGES = ("notes", "task_diff")
P_NOT_FIXED = re.compile(r'"p_not_fixed"\s*:\s*"?(\d+(?:\.\d+)?)')


def cut(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"


def prompt(judge: str, run: Run) -> str:
    if judge == "notes":
        notes = "\n\n".join(f"[{i}] {s.note}" for i, s in enumerate(run.steps, 1) if s.note)
        return NOTES.format(answer=ANSWER, notes=cut(notes or "(the agent wrote no notes)"))
    if judge == "task_diff":
        diff = run.submission if run.submission.strip() else "(empty: the agent submitted no change)"
        return TASK_DIFF.format(answer=ANSWER, task=cut(run.task), diff=cut(diff))
    raise ValueError(f"unknown judge {judge!r}")


def parse_answer(text: str) -> float | None:
    """The judge's chance that the change does not fix the bug, from 0 to 1; None if it gave none."""
    m = P_NOT_FIXED.search(text or "")
    return min(float(m[1]), 100.0) / 100 if m else None


def draw_sample(data_dir: str | Path, exclude: set[tuple[str, str]]) -> list[Run]:
    """40 submitted runs per building model, drawn at random without looking at results, leaving out
    the hand-labelling sample. The draw uses only the run list, so it does not depend on outcomes."""
    by_model: dict[str, list[Run]] = {}
    for run in iter_runs(data_dir):
        if run.submitted and (run.model, run.instance_id) not in exclude:
            by_model.setdefault(run.model, []).append(run)
    rng = random.Random(SAMPLE_SEED)
    return [run for model in sorted(by_model)
            for run in rng.sample(sorted(by_model[model], key=lambda r: r.instance_id), PER_MODEL)]


def write_prompts(runs: list[Run], out_dir: str | Path) -> None:
    """One JSON line per run and judge: the ids, the prompt version and the exact prompt text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for judge in JUDGES:
        with (out_dir / f"prompts_{judge}.jsonl").open("w", encoding="utf-8") as fh:
            for run in runs:
                fh.write(json.dumps({"model": run.model, "instance_id": run.instance_id, "judge": judge,
                                     "prompt_version": PROMPT_VERSION, "prompt": prompt(judge, run)}) + "\n")


def read_answers(path: str | Path) -> dict[tuple[str, str], dict]:
    """Answers saved as JSON lines with model, instance_id, answer (the reply text), judge_model and
    cost_usd. Each gets p, the parsed chance (None when the reply had none)."""
    answers = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            answers[(a["model"], a["instance_id"])] = a | {"p": parse_answer(a.get("answer", ""))}
    return answers


def read_sample_ids(path: str | Path) -> set[tuple[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {(r["model"], r["instance_id"]) for r in csv.DictReader(fh, delimiter="\t")}
