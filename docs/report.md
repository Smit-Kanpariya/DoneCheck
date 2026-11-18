# DoneCheck: method, results and limits (Stage 1)

**Question.** When an AI coding agent submits a fix and so says "done", does its run record hold the
evidence that the fix works? DoneCheck reads the record with fixed rules and scores each submitted
run. The scores are checked against the official hidden tests of SWE-bench Verified, which the
checker never sees.

**Result on a model it never saw** (GLM-4.5, 500 runs on the same 500 bugs as the building models,
scored once, at a threshold frozen on building data): DoneCheck flagged **13.1%** of the 229 false
"done" claims (95% interval 8.9% to 17.6%). It flagged **0.7%** of the 271 fixed runs (interval
0.0% to 1.9%). It cost $0 in API fees and took about 2.6 ms of computing per run. Trusting the agent
catches 0%. AUROC was 0.665 (0.620 to 0.713). The two AI-judge baselines are built and their prompts are frozen, but they **have not been
run**: calling them costs money, and no budget has been approved yet.

All numbers below come from `eval/results/*.json` and can be rebuilt with `uv run donecheck reproduce`.

## 1. Data

| Set | Models | Agent versions | Runs | Submitted, result known | False "done" | Fixed |
|---|---|---|---|---|---|---|
| Building | gpt-oss-120b, GPT-5 nano, Kimi K2 Instruct, Devstral 2512, DeepSeek V3.2 reasoner | 1.7.0, 1.17.1, 1.17.2 | 2,500 | 2,413 | 1,321 | 1,092 |
| Final test | GLM-4.5 | 1.9.1 | 500 | 500 | 229 | 271 |

- **Where the runs come from:** public mini-SWE-agent runs (`donecheck fetch`).
- **Positives and false alarms:** a *false "done"* is a submitted run that the hidden tests mark not
  fixed. A *false alarm* is a fixed run that gets flagged.
- **Not scored:** the 87 building runs that never submitted make no done claim.
- **The final-test count is right:** the 229 matches the leaderboard's count for GLM-4.5.
- **Two instruction wordings** (see `docs/claims.md`):
  - gpt-oss and Kimi (959 submitted runs) were told to submit when finished *or* stuck;
  - the others, GLM-4.5 included, were told to submit when finished *and* unable to make further
    progress.

## 2. Method

1. **Evidence checks** (`checks.py`, Day 2). Each run's commands and outputs are read with fixed
   rules into 27 facts. Examples:
   - edits found, checks and repository tests after the last code edit, and their results;
   - a failing check that passes after the fix;
   - an empty or bulk change, and edited files never opened first;
   - claims in the notes, labelled supported, unsupported or contradicted;
   - a final note that gives up.

   A checker only ever gets a run with its result blanked.
2. **The DoneCheck score** (`model.py`, Day 4). A logistic regression (standardized, L2, C = 1, not
   tuned) uses 25 of the facts.
   - Counts enter as log(1 + n), and the two result facts as one-hot.
   - Cost and API calls are left out because they follow each model's pricing and would identify the
     model. The wording is left out because it is a property of the setup; results are reported for
     each wording instead.
   - The classes are close to balanced (54.7% false "done"), so no class weights are used.
3. **Threshold** (decided on building data only, before the final test). DoneCheck flags a run when
   its score is ≥ 0.787. That is the lowest threshold at which at most 5% of fixed runs are flagged,
   using the building models' leave-one-model-out scores, i.e. scores for models each fit had not
   seen. The rule was picked over the 5% point of in-sample scores by testing both on the model left
   out (nested, building data only; `eval/results/threshold_rules.json`, from
   `python -m donecheck.analysis threshold-rules`):

   | Rule | Mean false alarms on the model left out | Models above 5% |
   |---|---|---|
   | in-sample scores | 8.5% | 3 of 5 |
   | leave-one-model-out scores (chosen) | 3.0% | 1 of 5 |

   The rule was chosen on these same five folds, so the fold-level catch and false-alarm numbers in
   section 3 are slightly optimistic for it. GLM-4.5 is the independent check.

4. **Metrics** (`scoring.py`).
   - **Operating point.** The catch rate and false-alarm rate are read at real thresholds. Nothing
     is interpolated between them.
   - **ROC catch at 5%.** This picks its threshold on the very runs it scores. It describes the curve;
     nobody could have had it in advance. The quotable result is the frozen threshold applied to new
     runs.
   - **Error bars.** 95% intervals come from 2,000 resamples of whole bugs (seed 20261117), so one
     bug's runs by different models stay together.
5. **Protocol.**
   - Building results are reported leave-one-model-out.
   - Then the model is fitted on all five building models, saved as plain numbers
     (`models/donecheck-v1.json`) and frozen with its threshold, code hashes, seeds and judge prompts
     (`eval/freeze.json`, commit 9c1ca3d).
   - After that, the GLM-4.5 runs were downloaded for the first time and scored once
     (`evaluate.final`). It refuses to run if a frozen file has changed, and it refuses a second run.

## 3. Building results (leave one model out)

| Left-out model | Runs | AUROC | ROC catch at 5% | Caught at its fold's threshold | False alarms |
|---|---|---|---|---|---|
| deepseek-v3.2-reasoner | 498 | 0.684 | 19.2% | 11.6% | 2.7% |
| devstral-2512 | 480 | 0.694 | 34.6% | 29.9% | 1.1% |
| gpt-5-nano | 476 | 0.707 | 35.1% | 24.8% | 1.1% |
| gpt-oss-120b | 491 | 0.713 | 22.4% | 27.1% | 8.5% |
| kimi-k2-instruct | 468 | 0.721 | 36.9% | 32.9% | 1.8% |

- **Overall.**
  - The mean AUROC over the five folds is **0.704**. The stop sign needs at least 0.65, so it passes.
  - All five folds pooled: AUROC 0.695 (0.672 to 0.716), ROC catch at 5% 28.5% (25.5% to 31.4%).
- **By wording:**

  | Wording | Runs | AUROC | ROC catch at 5% |
  |---|---|---|---|
  | Older | 959 | 0.685 (0.648 to 0.717) | 30.2% |
  | Newer | 1,454 | 0.690 (0.661 to 0.719) | 27.1% |

  Leaving out the 21 runs whose last note gave up: AUROC 0.692.
- **New model and new bugs.** Every model tried the same 500 bugs, so leave-one-model-out still
  trains on the held-out model's bugs. Splitting the bugs into 5 folds as well, so that the fit sees
  neither the model nor the bugs it scores, changes AUROC by −0.007 to +0.002 per model (0.679 to
  0.714).
  GLM-4.5 also ran the same 500 bugs, so the final test means "new model, same bugs".
- **Which checks matter.** Mean fold AUROC with each group of facts removed (all facts: 0.704):

  | Group removed | Change in AUROC |
  |---|---|
  | final change | −0.022 |
  | results | −0.013 |
  | claims | −0.009 |
  | effort | −0.002 |
  | giving up | −0.000 |
  | reruns | +0.001 |
  | care | +0.001 |
  | edits | +0.005 |
  | checks | +0.005 |

  The groups overlap, so removing one alone costs little. The edits and checks groups add nothing on
  top of the rest.
- **Strongest single checks** (AUROC as a lone score, below 0.5 meaning "more of it, more often
  fixed"):

  | Check | AUROC alone |
  |---|---|
  | checks after the last edit | 0.337 |
  | a check passed after the last edit | 0.360 |
  | claims in the notes | 0.368 |
  | repository tests after the last edit | 0.375 |

- **Calibration** on unseen models is loose (Brier 0.218). Scores of 0.1 to 0.2 were false 32% of the
  time; scores of 0.6 to 0.7 were false 56% of the time. So the CLI does not call the score a
  probability.

## 4. Final test: GLM-4.5, scored once

| Checker | AUROC | Caught | False alarms |
|---|---|---|---|
| Trust the agent | 0.500 | 0.0% | 0.0% |
| **DoneCheck, frozen threshold 0.787** | **0.665** (0.620 to 0.713) | **13.1%** (8.9% to 17.6%) | **0.7%** (0.0% to 1.9%) |
| DoneCheck, ROC point at ≤5% false alarms (picked on GLM-4.5 itself; not a real result) | | 21.0% (14.7% to 29.0%) | ≤5% |
| AI judge, notes only | pending | pending | pending |
| AI judge, task and diff | pending | pending | pending |

- **The frozen threshold was stricter than intended on GLM-4.5.** It flagged 6.4% of runs, with 0.7%
  false alarms against the 5% aim. So it caught less than the ROC point would have.
- **Calibration:** Brier 0.224.
- **Accept-or-flag curve:**

  | Runs flagged | Caught | False alarms | False "done" among accepted runs |
  |---|---|---|---|
  | 0% | 0.0% | 0.0% | 45.8% |
  | 10% | 17.9% | 3.3% | 41.8% |
  | 20% | 29.3% | 12.2% | 40.5% |
  | 30% | 41.0% | 20.7% | 38.6% |
  | 50% | 62.9% | 39.1% | 34.0% |

- **Without the 4 runs that gave up:** catch 12.8%, false alarms 0.4%, AUROC 0.663.

**Cost per run:**

| Checker | API fees | Time |
|---|---|---|
| DoneCheck | $0 | about 2.6 ms to load, check and score |
| AI judges | not measured (not run) | |

For scale, the agent itself spent $0.297 per GLM-4.5 run on average, and $0.209 per building run.

## 5. What it caught and missed on GLM-4.5 (read after the test)

This was run after the frozen predictions existed, with `donecheck.analysis`. It changes nothing
that was reported above. The counts are 30 caught, 199 missed, 2 false alarms and 269 fixed runs
accepted.

| Group | Runs | Empty change | Check passed after the last edit | No check after the last edit | Repository tests after the last edit |
|---|---|---|---|---|---|
| caught | 30 | 43.3% | 43.3% | 40.0% | 3.3% |
| missed | 199 | 0.0% | 91.5% | 7.5% | 17.6% |
| false alarm | 2 | 0.0% | 50.0% | 50.0% | 0.0% |
| accepted, fixed | 269 | 0.0% | 93.7% | 5.9% | 21.2% |

- **Caught.** Nearly half of the caught runs submitted an empty change. In two of them
  (`django__django-12708`, `django__django-13128`), the last note says the diff shows the fix, but
  the final command was a bare `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` without
  `git add -A && git diff --cached`, which hands in nothing. Of all 30 caught runs, 12 had no
  check after the last edit, 7 a failing last check, 1 an error, and 10 a passing one.
- **Missed.** The misses look like fixed runs in the record: in 91.5% of them a check the agent
  wrote passed after its last edit. The record shows the agent's own checks passing, not whether
  those checks test what the hidden tests test. This is the main limit of evidence read from the
  record.
- **False alarms.** Both were long runs (38 and 108 steps): one with no check after its last edit,
  one whose last check failed.

How this was read: the fact profiles of all 500 runs by group (`final_glm45_errors.json`), plus the
last notes of 12 runs: the 5 highest-scoring caught, both false alarms and the 5 lowest-scoring
missed. The plan asked for up to 50 misses and 50 false alarms read one by one. There are only 2
false alarms, and the misses were not each read by hand.

## 6. Overclaims versus honest failures (rule-based, not human)

These come from the rules in `checks.py`, not from people (see section 7). A not-fixed run is an
*overclaim* if its done claim, or a claim made after its last code edit, is contradicted by the
record (`docs/claims.md`, section 4).

| Wording | Not-fixed runs that overclaim | Fixed runs the same rule would call overclaims |
|---|---|---|
| All (building) | 135 of 1,321 (10.2%) | 93 of 1,092 (8.5%) |
| Older | 57 of 610 | 28 of 349 |
| Newer | 78 of 711 | 65 of 743 |
| GLM-4.5 | 24 of 229 | 19 of 271 |

By these rules, most false "done" runs are not contradicted by their own record. And a contradicted
claim is only a little more common in failed runs than in fixed ones.

## 7. What is not done, and what still needs people

- **Human labels: none yet.** `labels/claims.tsv` is empty. The overclaim numbers above come from
  rules. No model-made labels were produced, and none are presented as human. Still to do: the
  100-run hand labels, a hand check of the evidence checks on 30 labelled runs, and the blind
  relabel of runs 1 to 30 (for self-agreement). The sample and the blind setup are unchanged.
- **AI-judge baselines: built, not run.**
  - `judges.py` holds the two prompts (version `judges-v1`, frozen). Neither prompt ever includes the
    hidden tests, the correct fix or a result.
  - `eval/judge_sample_200.tsv` lists the 200-run building sample (40 per model, drawn without
    outcomes, none from the labelling sample), and `donecheck judges prompts building` writes the
    exact prompts (`judges prompts final` for GLM-4.5).
  - To finish this baseline:
    1. Approve a model and a spending cap.
    2. Send the prompts with `donecheck judges run <split> <judge> <endpoint> <model>`, for any
       OpenAI-compatible endpoint. Use `--local` for a free local server, or both prices and
       `--max-usd`, which are checked before every call. Replies go to
       `eval/judges/answers_<judge>_<split>.jsonl`.
    3. `donecheck judges score <split>` reports each judge next to DoneCheck on the same runs.
  - No local model runtime was installed on the build machine, so not even a free local run was
    possible (status `BLOCKED_BUDGET`).
- **Not merged or released.** The work is on the branch `stage-1`; `main` is unchanged.

## 8. Limits

- **The checks read commands and outputs, not code.** A script counts as passed when it exits
  without a traceback, even if it prints a wrong value. Claims are found by keywords. About 1 in 5
  changed files is not tied to the command that changed it (Day 2).
- **Six models, one agent, one benchmark.** The intervals resample bugs, not models, so they leave
  out model-to-model spread. The five folds ranged from 0.684 to 0.721 AUROC, and their false alarms
  at a fixed rule from 1.1% to 8.5%.
- **Same bugs.** GLM-4.5 attempted the same 500 bugs as the building models. The bug-disjoint check
  above suggests this matters little, but it is not a test on new bugs.
- **Thresholds drift between models.** The frozen threshold gave 0.7% false alarms on GLM-4.5 rather
  than about 5%. A user should expect the false-alarm rate on a new model to land anywhere from far
  below 5% to above it.
- **Hidden tests are not perfect ground truth.** Some "not fixed" verdicts are disputed fixes.
- **The final test is single-use.** Any change made after it, including the analysis in section 5,
  is exploratory and labelled as such.

## 9. Reproduce

```bash
uv sync
uv run donecheck reproduce        # downloads about 906 MiB of public run files, rebuilds, compares
uv run donecheck verify data_final/20250822_mini-v1.9.1_glm-4.5/trajs/django__django-13128.traj.json
uv run pytest -q
```

`reproduce` rebuilds the building results and model into `eval/reproduced/`, rescores GLM-4.5 with
the committed frozen model, and checks five headline numbers against the committed ones. On this
machine all five matched to within 1e-9.
