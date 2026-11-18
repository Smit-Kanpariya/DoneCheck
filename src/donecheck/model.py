"""The DoneCheck score (plan section 8, Day 4): logistic regression on the evidence checks.

The features are check_run facts about the run's own record. Left out on purpose: cost and
api_calls, which follow each model's pricing and speed and so would identify the model rather than
judge the run, and the instruction wording, which is a property of the setup (results are reported
for each wording instead). Counts enter as log(1 + n); the two result facts are one-hot.
The score is the fitted chance that a submitted run is a false "done"; scored runs at or above the
threshold are flagged.
"""

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donecheck.scoring import auroc, catch_at, operating_point, threshold_at

FORMAT = "donecheck-model-1"
RESULTS = ("passed", "failed", "error", "timeout", "none", "unknown")
CATEGORICAL = ("last_check_result", "last_test_result")
GROUPS = {
    "effort": ["steps", "format_errors", "timeouts"],
    "final change": ["submission_empty", "diff_files", "diff_test_files", "diff_scratch_files"],
    "edits": ["code_edits", "unknown_edits"],
    "checks": ["checks", "tests", "checks_after_last_edit", "tests_after_last_edit"],
    "results": ["passed_after_last_edit", "last_check_result", "last_test_result"],
    "reruns": ["fail_then_pass"],
    "care": ["edited_files", "edited_unopened"],
    "claims": ["claims", "claims_supported", "claims_unsupported", "claims_contradicted",
               "final_claims_contradicted"],
    "giving up": ["gave_up"],
}
FACTS = [fact for group in GROUPS.values() for fact in group]


def vectorize(facts: dict, use: list[str] = FACTS) -> dict[str, float]:
    """One run's facts as named numbers."""
    out = {}
    for name in use:
        value = facts[name]
        if name in CATEGORICAL:
            out |= {f"{name}={r}": float(value == r) for r in RESULTS}
        elif isinstance(value, bool):
            out[name] = float(value)
        else:
            out[name] = math.log1p(value)
    return out


def matrix(all_facts: list[dict], use: list[str] = FACTS) -> tuple[list[str], np.ndarray]:
    rows = [vectorize(f, use) for f in all_facts]
    names = list(rows[0])
    return names, np.array([[r[n] for n in names] for r in rows])


def fit(X: np.ndarray, y: np.ndarray):
    """Standardize, then an L2 logistic regression with scikit-learn's default strength (C=1),
    not tuned. The classes are close to balanced in building data, so no class weights."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y)


def leave_one_model_out(X: np.ndarray, y: np.ndarray, models: np.ndarray,
                        thresholds: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Fit on all models but one and score the one left out, for each model in turn.

    With thresholds=True, each fold also picks its threshold the way the final model does, from its
    training models only, and applies it to the model left out."""
    scores, folds = np.zeros(len(y)), []
    for train, test in LeaveOneGroupOut().split(X, y, models):
        model = fit(X[train], y[train])
        s = model.predict_proba(X[test])[:, 1]
        scores[test] = s
        fold = {"held_out": str(models[test][0]), "runs": int(len(test)), "auroc": auroc(y[test], s),
                "roc_catch_at_5": catch_at(y[test], s)}
        if thresholds:
            t = choose_threshold(X[train], y[train], models[train])
            fold |= {"threshold": t, "with_threshold": operating_point(y[test], s, t)}
        folds.append(fold)
    return scores, folds


def choose_threshold(X: np.ndarray, y: np.ndarray, models: np.ndarray) -> float:
    """The 5% false-alarm point of the models' own leave-one-model-out scores, so the threshold is set
    on scores for models each fit had not seen. Chosen over the 5% point of in-sample scores on
    building data alone: on the model left out, that rule averaged 8.5% false alarms (above 5% for
    3 of 5 models), this one 3.0% (above 5% for 1 of 5). See docs/report.md."""
    scores, _ = leave_one_model_out(X, y, models)
    return threshold_at(y, scores)


def leave_model_and_bugs_out(X: np.ndarray, y: np.ndarray, models: np.ndarray, bugs: np.ndarray,
                             folds: int = 5, seed: int = 20261117) -> tuple[np.ndarray, dict[str, float]]:
    """Scores, and AUROC per model, when the fit has seen neither that model nor the bugs it scores.

    Every model in this data attempted the same 500 bugs, so leave-one-model-out still trains on the
    held-out model's bugs, through the other models' runs. Here the bugs are also split into folds:
    a model's runs in one bug fold are scored by a fit on the other models' runs of the other folds."""
    unique = np.unique(bugs)
    fold_of = dict(zip(unique[np.random.default_rng(seed).permutation(len(unique))], np.arange(len(unique)) % folds))
    bug_fold = np.array([fold_of[b] for b in bugs])
    scores = np.zeros(len(y))
    for m in np.unique(models):
        for k in range(folds):
            train, test = (models != m) & (bug_fold != k), (models == m) & (bug_fold == k)
            scores[test] = fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return scores, {str(m): auroc(y[models == m], scores[models == m]) for m in np.unique(models)}


def calibration(y: np.ndarray, score: np.ndarray, bins: int = 10) -> dict:
    """Brier score and, per score band of width 1/bins, the mean score against the false "done" rate."""
    band = np.minimum((score * bins).astype(int), bins - 1)
    table = [{"band": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "runs": int((band == b).sum()),
              "mean_score": float(score[band == b].mean()), "false_done_rate": float(y[band == b].mean())}
             for b in range(bins) if (band == b).any()]
    return {"brier": float(np.mean((score - y) ** 2)), "bands": table}


def export(model, names: list[str], threshold: float, info: dict) -> dict:
    """The fitted model as plain numbers, so it can be read, checked and used without pickle."""
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return {"format": FORMAT, "facts": FACTS, "features": names, "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "threshold": threshold, **info}


def score(artifact: dict, facts: dict) -> tuple[float, dict[str, float]]:
    """The score for one run's facts, and each feature's push on the log-odds (positive: towards flag)."""
    if artifact.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    x = vectorize(facts, artifact["facts"])
    push = {name: c * (x[name] - m) / s for name, c, m, s in
            zip(artifact["features"], artifact["coef"], artifact["mean"], artifact["scale"])}
    logit = artifact["intercept"] + sum(push.values())
    return 1 / (1 + math.exp(-logit)), push
