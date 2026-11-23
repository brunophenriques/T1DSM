"""Tests for the recovery metrics on step curves (PROTOCOL.md section 10).

Run with `python tests/test_recovery_metrics.py` or `pytest tests/`.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.metrics import first_sustained
from relevance.recovery import ranking_recovery, accuracy_recovery
from relevance.analysis import analyze


def test_first_sustained():
    condition = [False, True, True, False, True, True, True]
    assert first_sustained(condition, 3) == 4
    assert first_sustained([True, True], 3) is None


def test_ranking_immediate_recovery():
    t = np.arange(50, 501, 50)
    precision = np.ones(len(t))

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 50


def test_ranking_delayed_recovery():
    t = np.arange(50, 601, 50)

    precision = [
        1, 1, 1, 1,
        0, 0.33, 0.67, 1,
        1, 1, 1, 1
    ]

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 200


def test_ranking_never_recovers():
    t = np.arange(50, 501, 50)
    precision = np.full(len(t), 0.67)

    result = ranking_recovery(t, precision, drift_end=200)

    assert not result["recovered"]
    assert np.isnan(result["recovery"])
    assert result["censored_time"] == 300


def test_temporary_recovery():
    """Two good checkpoints followed by a bad one do not count; the sustained run does."""
    t = np.arange(50, 501, 50)

    precision = [
        1, 1, 1, 1,
        1, 1, 0, 1, 1, 1
    ]

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 200


def test_ranking_nan_is_not_recovered_and_empty_segment_is_nan():
    t = np.arange(50, 501, 50)
    precision = np.where(t > 300, np.nan, 1.0)          # NaN after 300 must not count as precision 1
    r = ranking_recovery(t, precision, drift_end=200)
    assert not r["recovered"] and r["censored_time"] == 300
    r = ranking_recovery(t, np.ones(len(t)), drift_end=500)  # no checkpoint after the drift
    assert not r["recovered"] and np.isnan(r["recovery"]) and np.isnan(r["censored_time"])


def test_accuracy_recovery():
    t = np.arange(50, 2001, 50)

    accuracy = np.where(
        t <= 1000,
        0.85,
        np.where(t <= 1250, 0.60, 0.84)
    )

    result = accuracy_recovery(t, accuracy, drift_start=1000)

    assert result["recovered"]
    assert result["recovery"] == 300
    assert np.isclose(result["accuracy_drop"], 0.25)


def test_accuracy_never_recovers():
    t = np.arange(50, 2001, 50)
    accuracy = np.where(t <= 1000, 0.85, 0.60)

    result = accuracy_recovery(t, accuracy, drift_start=1000)

    assert not result["recovered"]
    assert np.isnan(result["recovery"])
    assert result["censored_time"] == 1000


def test_accuracy_recovery_needs_a_pre_drift_level():
    """Without checkpoints before the drift the recovery is undefined: NaN, not censored."""
    t = np.arange(1050, 2001, 50)
    r = accuracy_recovery(t, np.full(len(t), 0.8), drift_start=1000, pre_samples=500)
    assert not r["recovered"] and np.isnan(r["recovery"]) and np.isnan(r["censored_time"])


def test_accuracy_trough_window_and_brief_crossing():
    """The trough is searched only within `trough_samples`; a 2-checkpoint crossing is not recovery."""
    t = np.arange(50, 3001, 50)
    acc = np.full(len(t), 0.80)
    acc[(t > 1000) & (t <= 1500)] = 0.60                 # drop
    acc[(t > 1500) & (t <= 1600)] = 0.80                 # brief crossing (2 checkpoints)
    acc[(t > 1600) & (t <= 2000)] = 0.65
    acc[t > 2000] = 0.79                                 # sustained recovery from 2050
    r = accuracy_recovery(t, acc, drift_start=1000, pre_samples=1000, trough_samples=1000, consecutive=3)
    assert r["recovered"] and r["recovery"] == 1050 and np.isclose(r["accuracy_drop"], 0.20)


def test_backward_compatible_module():
    from wf2_drift_validity import recovery_metrics as old
    assert old.ranking_recovery is ranking_recovery and old.accuracy_recovery is accuracy_recovery
    assert old.first_sustained is first_sustained


ANALYSIS = {"lags_samples": [50, 200], "topk": 2, "recovery_precision": 1.0, "recovery_consecutive": 3,
            "warmup_flush": 200, "acc_tol": 0.02}


def _gradual_run(width, switch_after_dp):
    """G ranks the new set {2,3} first from dp + switch_after_dp on; drift point dp = 3000."""
    every, T = 50, 120
    t = np.arange(1, T + 1) * every
    G = np.where((t < 3000 + switch_after_dp)[:, None], np.array([1.0, 0.9, 0.2, 0.3]),
                 np.array([0.2, 0.3, 1.0, 0.9]))
    arr = dict(t=t, G=G, A=np.full((T, 4), 0.25), acc_win=np.full(T, 0.8), n_rules=np.full(T, 5),
               created=np.arange(T), removed=np.arange(T))
    truth = np.zeros((t[-1], 4), dtype=int)
    truth[:3000, [0, 1]] = 1
    truth[3000:, [2, 3]] = 1
    return analyze(arr, truth, [3000], ANALYSIS, every, drift_widths=[width])


def test_recovery_from_drift_point_removes_the_floor():
    # Gradual drift (transition 2500-3500); the ranking re-orients at 3200, before the transition ends.
    m = _gradual_run(width=1000, switch_after_dp=200)
    assert m["recovery/G/drift1"] == 50.0                 # protocol metric: floor (first checkpoint after 3500)
    assert m["recovery_from_dp/G/drift1"] == 200.0        # same-origin metric sees the real timing
    assert m["recovery_from_dp_censored/G/drift1"] == 200.0
    # Abrupt drift: both metrics are identical.
    m = _gradual_run(width=0, switch_after_dp=200)
    assert m["recovery/G/drift1"] == m["recovery_from_dp/G/drift1"] == 200.0
    # Never re-orients: censored at the end, from the drift point.
    m = _gradual_run(width=1000, switch_after_dp=10 ** 6)
    assert np.isnan(m["recovery_from_dp/G/drift1"]) and m["recovery_from_dp_censored/G/drift1"] == 6000 - 3000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
