"""Tests for donecheck.scoring, on small cases worked out by hand."""

import math

import numpy as np

from donecheck.scoring import accept_flag_curve, auroc, bootstrap, catch_at, operating_point, threshold_at

# 4 false "done" runs (1) and 20 fixed runs (0). One fixed run outscores every false "done".
Y = [1, 1, 1, 1] + [0] * 20
S = [0.9, 0.8, 0.3, 0.1] + [0.95] + [0.2] * 19


def test_auroc_counts_ties_as_half():
    # pairs won: 0.9 and 0.8 beat 19 of 20; 0.3 beats 19; 0.1 beats none -> 57 of 80
    assert math.isclose(auroc(Y, S), 57 / 80)
    assert auroc([1, 0, 1, 0], [5, 5, 5, 5]) == 0.5


def test_threshold_allows_at_most_five_percent_of_fixed_runs():
    # 5% of 20 fixed runs is 1, so the 0.95 run may be flagged but no 0.2 run may be
    t = threshold_at(Y, S)
    assert t == 0.3
    assert operating_point(Y, S, t) == {"catch": 0.75, "false_alarm": 0.05, "flagged": 4 / 24}
    assert catch_at(Y, S) == 0.75
    # with 19 fixed runs, 5% allows 0 of them: only scores above 0.95 could be flagged -> none
    assert threshold_at(Y[:-1], S[:-1]) == float("inf")
    assert catch_at(Y[:-1], S[:-1]) == 0.0


def test_a_constant_score_catches_nothing():
    y = [1] * 10 + [0] * 10
    assert catch_at(y, [0.0] * 20) == 0.0  # flagging anyone flags every fixed run too


def test_accept_flag_curve_keeps_ties_together():
    rows = accept_flag_curve(Y, S, shares=(0, 0.1, 0.5))
    assert rows[0] == {"target_flagged": 0, "catch": 0.0, "false_alarm": 0.0, "flagged": 0.0,
                       "false_done_among_accepted": 4 / 24}
    # 10% of 24 rounds to 2 runs: the 0.95 fixed run and the 0.9 false "done"
    assert (rows[1]["catch"], rows[1]["false_alarm"], rows[1]["false_done_among_accepted"]) == (0.25, 0.05, 3 / 22)
    # 50% would cut through the nineteen 0.2 runs, so all of them are flagged
    assert rows[2]["flagged"] == 23 / 24 and rows[2]["false_done_among_accepted"] == 1.0


def test_bootstrap_draws_whole_groups():
    # two groups whose means are 0 and 1: every resample mean is 0, 0.5 or 1, never a mix within a group
    values = np.array([0, 0, 0, 1, 1, 1])
    groups = ["a", "a", "a", "b", "b", "b"]
    lo, hi = bootstrap(groups, lambda v: v.mean(), values, n=200)
    assert lo == 0.0 and hi == 1.0
    assert bootstrap(groups, lambda v: v.mean(), values, n=200) == (lo, hi)  # seeded
    assert bootstrap(["a"] * 6, lambda v: v.mean(), values, n=50) == (0.5, 0.5)
