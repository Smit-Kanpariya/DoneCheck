"""Scoring a checker against the hidden tests (plan section 8).

A checker gives every submitted run a score; a higher score means the run is more likely a false
"done" (submitted but not fixed). Runs with a score at or above a threshold are flagged.
- catch rate: the share of false "done" runs that are flagged;
- false-alarm rate: the share of fixed runs that are flagged.
No interpolation between thresholds: every reported operating point is one a real threshold gives.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

FALSE_ALARM = 0.05  # the false-alarm rate the headline number is read at
RESAMPLES = 2000
SEED = 20261117


def auroc(false_done, score) -> float:
    """The chance that a random false "done" run scores above a random fixed run (ties count half)."""
    return float(roc_auc_score(np.asarray(false_done, bool), np.asarray(score, float)))


def threshold_at(false_done, score, max_false_alarm: float = FALSE_ALARM) -> float:
    """The lowest threshold that flags at most max_false_alarm of the fixed runs; inf flags nothing."""
    y, s = np.asarray(false_done, bool), np.asarray(score, float)
    fixed = np.sort(s[~y])
    allowed = int(np.floor(max_false_alarm * len(fixed) + 1e-9))
    candidates = np.unique(s)
    flagged_fixed = len(fixed) - np.searchsorted(fixed, candidates, side="left")  # fixed runs >= t
    ok = flagged_fixed <= allowed  # true from some threshold upwards, since the count only falls
    return float(candidates[ok].min()) if ok.any() else float("inf")


def operating_point(false_done, score, threshold: float) -> dict:
    """Catch rate, false-alarm rate and flagged share when runs scoring >= threshold are flagged."""
    y, s = np.asarray(false_done, bool), np.asarray(score, float)
    flagged = s >= threshold
    return {"catch": float(flagged[y].mean()), "false_alarm": float(flagged[~y].mean()),
            "flagged": float(flagged.mean())}


def catch_at(false_done, score, max_false_alarm: float = FALSE_ALARM) -> float:
    """Catch rate at the threshold chosen on these same runs (a point on their ROC curve)."""
    return operating_point(false_done, score, threshold_at(false_done, score, max_false_alarm))["catch"]


def accept_flag_curve(false_done, score, shares=(0, .05, .1, .15, .2, .25, .3, .4, .5)) -> list[dict]:
    """For each target share of runs flagged (highest scores first, ties kept together): what is
    flagged, what it catches, and how often an accepted run is still a false "done"."""
    y, s = np.asarray(false_done, bool), np.asarray(score, float)
    ranked = np.sort(s)[::-1]
    rows = []
    for share in shares:
        k = int(round(share * len(s)))
        t = ranked[k - 1] if k else float("inf")
        accepted = s < t
        rows.append({"target_flagged": share, **operating_point(y, s, t),
                     "false_done_among_accepted": float(y[accepted].mean()) if accepted.any() else None})
    return rows


def bootstrap(groups, metric, *arrays, n: int = RESAMPLES, seed: int = SEED) -> tuple[float, float]:
    """The middle 95% of metric(*arrays) over n resamples that draw whole groups (bug ids) with
    replacement, so runs of one bug by different models stay together."""
    arrays = [np.asarray(a) for a in arrays]
    _, group_of = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(group_of == g) for g in range(group_of.max() + 1)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        values.append(metric(*(a[idx] for a in arrays)))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summary(false_done, score, groups, threshold: float | None = None, n: int = RESAMPLES) -> dict:
    """AUROC and the ROC catch rate at 5% false alarms, each with a 95% interval by bug id. The ROC
    catch rate picks its threshold on the very runs it scores, so it describes the curve and is not a
    result anyone could have had in advance. With a threshold fixed in advance, also its actual catch
    and false-alarm rates on these runs, with intervals: that is the result to quote."""
    y, s = np.asarray(false_done, bool), np.asarray(score, float)
    out = {"runs": int(len(y)), "false_done": int(y.sum()), "fixed": int((~y).sum()),
           "auroc": auroc(y, s), "auroc_95": bootstrap(groups, auroc, y, s, n=n),
           "roc_catch_at_5": catch_at(y, s), "roc_catch_at_5_95": bootstrap(groups, catch_at, y, s, n=n)}
    if threshold is not None:
        point = operating_point(y, s, threshold)
        out |= {"threshold": threshold, "fixed_threshold": point,
                "fixed_threshold_catch_95": bootstrap(
                    groups, lambda a, b: operating_point(a, b, threshold)["catch"], y, s, n=n),
                "fixed_threshold_false_alarm_95": bootstrap(
                    groups, lambda a, b: operating_point(a, b, threshold)["false_alarm"], y, s, n=n)}
    return out
