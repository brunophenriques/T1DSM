"""Tests for the WF5 window analysis and restabilisation time (PROTOCOL.md §4, §5).

Run with `python tests/test_wf5_windows.py` or `pytest tests/`.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wf5_windows.windows import (clean_mask, window_means, window_analysis, last_pre_drift_window,
                                 restabilisation, lag_wtau)

T = np.arange(50, 20001, 50)                       # protocol checkpoints, n = 20,000
V1 = np.linspace(1.0, 0.2, 10)                     # top-3 = {0, 1, 2}
V2 = V1[[9, 8, 7, 6, 5, 0, 1, 2, 4, 3]]            # top-3 = {5, 6, 7}


def _series(switch_at=None, noise=0.005, seed=0):
    """G = V1 (+ tiny noise that never changes the order); from checkpoint time `switch_at` on, V2."""
    rng = np.random.default_rng(seed)
    G = np.tile(V1, (len(T), 1))
    if switch_at is not None:
        G[T >= switch_at] = V2
    return G + rng.uniform(-noise, noise, G.shape)


def test_clean_mask_warmup_and_gradual():
    seg, clean = clean_mask(T, [10000], [1000], flush=200)
    assert not clean[T < 200].any() and clean[(T >= 200) & (T <= 9500)].all()
    assert not clean[(T > 9500) & (T < 10700)].any()          # buffer touches the transition
    assert clean[T >= 10700].all() and seg[T > 10500].min() == 1


def test_window_means_use_clean_checkpoints_only():
    G = np.tile(np.arange(10.0), (len(T), 1))
    G[T < 200] = 99.0                                          # warm-up values must be ignored
    _, clean = clean_mask(T, flush=200)
    W, n_clean, lo = window_means(T, G, clean, 20000)
    assert W.shape == (20, 10) and lo[-1] == 19000
    assert n_clean[0] == 17 and (n_clean[1:] == 20).all()      # t = 200..1000 in the first window
    assert np.allclose(W, np.arange(10.0))


def test_stationary_first_vs_last_is_identical():
    r = window_analysis(T, _series(), k=3, n_samples=20000)
    assert r["first_last"] == (1.0, 1.0) and r["pre_last"] is None
    assert all(row["jaccard"] == 1.0 and not row["crosses_drift"] for row in r["consecutive"])


def test_swap_is_seen_by_first_last_and_pre_last():
    r = window_analysis(T, _series(switch_at=10050), k=3, n_samples=20000, drift_points=[10000])
    assert r["pre"] == 9 and r["first_last"][0] == 0.0 and r["pre_last"][0] == 0.0
    crossing = [row for row in r["consecutive"] if row["crosses_drift"]]
    assert [(c["win_a"], c["win_b"]) for c in crossing] == [(9, 10)]
    assert all(row["jaccard"] == 1.0 for row in r["consecutive"] if not row["crosses_drift"])


def test_last_pre_drift_window():
    assert last_pre_drift_window([10000]) == 9                 # abrupt: window 9,000-10,000
    assert last_pre_drift_window([10000], [1000]) == 8         # gradual: transition starts at 9,500
    assert last_pre_drift_window([]) is None


def test_lag_wtau_constant_is_nan():
    G = np.ones((10, 4)); G[5:] = [1, 2, 3, 4]
    s = lag_wtau(G, 4)
    assert np.isnan(s[:9]).all() and np.isclose(s[9], 1.0)


def test_restabilisation_exact_time():
    # Order switches at t = 10,150: lag-4 pairs ending at 10,150..10,300 compare V1 with V2, so the
    # first of 3 consecutive pairs back at τ = 1 ends at 10,350 -> 350 samples after the drift.
    r = restabilisation(T, _series(switch_at=10150), [10000])
    assert r["threshold"] == 1.0 and r["dropped"] and 10150.0 <= r["t_trough"] <= 10300.0
    assert r["restab"] == r["restab_from_dp"] == 350.0


def test_restabilisation_gradual_anchors():
    r = restabilisation(T, _series(switch_at=10150), [10000], [1000])
    assert r["restab"] == 10550.0 - 10500.0                    # protocol: after the transition end (floor)
    assert r["restab_from_dp"] == 350.0                        # from the drop, timed from the drift point


def test_restabilisation_no_drop_and_censored():
    r = restabilisation(T, _series(), [10000])
    assert not r["dropped"] and r["restab"] == 0.0
    G = _series(switch_at=10150)
    G[T >= 10150] = np.random.default_rng(1).uniform(0, 1, G[T >= 10150].shape)   # never settles
    r = restabilisation(T, G, [10000])
    assert r["dropped"] and np.isnan(r["restab"]) and r["restab_censored"] == 20000 - 10000


def test_trough_is_searched_near_the_drift_only():
    G = _series()
    G[(T >= 15000) & (T < 15200)] = V2                          # a disturbance 5,000 samples later
    r = restabilisation(T, G, [10000])
    assert not r["dropped"] and r["restab"] == 0.0 and r["t_trough"] <= 11000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
