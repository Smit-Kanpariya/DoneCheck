# DoneCheck

**Is a coding agent's "done" backed by evidence in its run record?**

AI coding agents submit a fix and report "done" even when the bug is still there. In 2,413 submitted
SWE-bench Verified runs by five models, 1,321 did not fix the bug. DoneCheck reads the run record
(every command, output, note and the final change) with fixed rules, scores how likely the "done"
is false, and is tested against the benchmark's hidden tests, which it never sees.

## Result

The model never saw GLM-4.5, though GLM-4.5 ran the same 500 bugs. Its 500 runs were scored once, at
a threshold frozen beforehand on the other five models:

| | Caught (of 229 false "done") | False alarms (of 271 fixed) | AUROC | API cost per run |
|---|---|---|---|---|
| Trust the agent | 0.0% | 0.0% | 0.500 | $0 |
| **DoneCheck** | **13.1%** (8.9% to 17.6%) | **0.7%** (0.0% to 1.9%) | **0.665** | **$0** (about 2.6 ms) |
| AI judges (notes; task and diff) | not run yet | | | |

- **The threshold was stricter than intended.** It was set to aim for at most 5% false alarms, and
  on GLM-4.5 it gave 0.7%. Picked on GLM-4.5 itself for 5%, the same scores would have caught 21.0%.
  That is a point on the curve, not a result anyone could have had in advance.
- **Building models.** On the five building models, one model left out at a time, mean AUROC was
  0.704.
- **Intervals.** They are 95%, from 2,000 resamples by bug.
- **The AI-judge baselines** have frozen prompts and a 200-run sample, but calling them needs an API
  budget.
- **Human claim labels** have not been made yet.

Method, every table and the limits: [docs/report.md](docs/report.md).

## Use it

```bash
uv sync
uv run donecheck fetch final data_final       # the public GLM-4.5 runs (about 106 MiB)
uv run donecheck verify data_final/20250822_mini-v1.9.1_glm-4.5/trajs/django__django-13128.traj.json
```

```
DoneCheck: FLAG  score 1.00, threshold 0.79
The score ranks how likely this 'done' is false. It is a fitted logistic-regression output, not a
calibrated probability: on models it was not fitted on, its calibration drifts (docs/report.md).
Towards flag:
  - the submitted change is empty
  - no check passed after the last code edit
  - 0 existing files changed
  - 15 code edits found
Towards accept:
  ...
```

This GLM-4.5 run made 15 code edits and then submitted with a bare `echo` of the submit word, which
hands in an empty change. The hidden tests mark it not fixed.

`verify` exits 0 for accept, 1 for flag, 2 when the run never submitted, and 3 when it cannot read
the run or the model file. The score is a fitted
logistic-regression output, not a calibrated probability.

## Rebuild every number

```bash
uv run donecheck reproduce    # downloads all runs, rebuilds into eval/reproduced/, compares
uv run pytest -q
```

| Path | What it is |
|---|---|
| `src/donecheck/loader.py`, `checks.py` | Read a run file; the evidence checks |
| `src/donecheck/model.py`, `scoring.py` | The DoneCheck score; AUROC, catch rate, error bars |
| `src/donecheck/evaluate.py`, `judges.py` | Building and final evaluation; the AI-judge prompts |
| `src/donecheck/cli.py`, `fetch.py`, `demo.py` | The command, the download, the demo page |
| `src/donecheck/labels.py`, `judging.py` | The hand-labelling checks and analysis; running and scoring the AI judges |
| `models/donecheck-v1.json`, `eval/freeze.json` | The frozen model and what was frozen |
| `eval/results/` | Every reported number, and the predictions |
| `docs/claims.md`, `labels/` | What counts as a claim; the hand-labelling sample |
| `demo/index.html` | Real GLM-4.5 runs that were caught, missed and falsely flagged |

## Data

The run files come from the public
[SWE-bench experiments](https://github.com/SWE-bench/experiments) bucket and were made with
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent). No licence is stated for them, so
this repository links to them and never stores copies. `donecheck fetch` downloads them.
