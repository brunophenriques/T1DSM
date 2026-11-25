"""Tests for the WF5 window analysis and restabilisation time (PROTOCOL.md §4, §5).

Run with `python tests/test_wf5_windows.py` or `pytest tests/`.
"""
import sys
import json
import pytest
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


def test_statuses_and_negative_gradual_offset():
    r = restabilisation(T, _series(), [10000])
    assert r["restab_status"] == "no_destabilisation"
    assert not r["restabilised"] and r["restab_from_dp"] == 0
    r = restabilisation(T, _series(switch_at=9600), [10000], [1000])
    assert r["restab_from_dp_status"] == "recovered"
    assert r["restab_from_dp"] == -200  # lag-200 comparison settles at 9800
    assert r["restab"] == 50
    assert r["restab_status"] == "recovered" and r["restabilised"]
    r = restabilisation(T, _series(switch_at=10150), [10000])
    assert r["restab_status"] == "recovered" and r["restab"] == 350


def test_undefined_threshold_and_insufficient_post_data():
    r = restabilisation(T, _series(), [100])
    assert r["restab_status"] == "undefined"
    assert r["restab_reason"] == "undefined_pre_drift_threshold"
    assert np.isnan(r["restab"]) and not r["restabilised"]
    r = restabilisation(T, _series(), [19950])
    assert r["restab_reason"] == "insufficient_post_drift_data"
    assert np.isnan(r["restab_from_dp"])


def test_explicit_censor_limit():
    G = _series()
    G[T > 10000] = np.random.default_rng(1).uniform(0, 1, G[T > 10000].shape)
    r = restabilisation(T, G, [10000])
    assert r["restab_status"] == r["restab_from_dp_status"] == "censored"
    assert not r["restabilised"] and np.isnan(r["restab"])
    assert r["restab_censor_limit"] == r["restab_from_dp_censor_limit"] == 10000


def test_load_runs_missing_seed_and_file(tmp_path, monkeypatch):
    import pytest
    import wf5_windows.analyze_wf5 as analysis
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    base = tmp_path / analysis.STREAMS["S1"][0]
    for seed in range(9):
        (base / f"seed_{seed}").mkdir(parents=True)
    with pytest.raises(ValueError, match=r"S1: missing seeds \[9\]"):
        analysis.load_runs("S1")
    (base / "seed_9").mkdir()
    with pytest.raises(ValueError, match="S1 seed 0: missing required file.*checkpoints.npz"):
        analysis.load_runs("S1")


@pytest.mark.parametrize("end", [9900, 10000, 10500])
def test_protocol_undefined_preserves_early_midpoint_recovery(end):
    mask = T <= end
    r = restabilisation(T[mask], _series(switch_at=9600)[mask], [10000], [1000])
    assert r["dropped"] and not r["restabilised"]
    assert r["restab_status"] == "undefined"
    assert r["restab_reason"] == "insufficient_post_transition_data"
    assert np.isnan(r["restab"]) and np.isnan(r["restab_censor_limit"])
    assert np.isnan(r["restab_censored"])
    assert r["restab_from_dp_status"] == "recovered"
    assert r["restab_from_dp"] == -200
    assert r["restab_from_dp_reason"] is None
    assert np.isnan(r["restab_from_dp_censor_limit"])


@pytest.mark.parametrize("end", [9750, 10000, 10500, 10600, 11000])
def test_censoring_requires_observations_at_or_after_anchor(end):
    mask = T <= end
    G = _series()
    G[T >= 9600] = np.random.default_rng(1).uniform(0, 1, G[T >= 9600].shape)
    r = restabilisation(T[mask], G[mask], [10000], [1000])
    assert r["dropped"]
    for key, anchor in (("restab", 10500), ("restab_from_dp", 10000)):
        limit = r[f"{key}_censor_limit"]
        assert np.isnan(limit) or limit >= 0
        assert np.isnan(r[key])
        if end < anchor or (key == "restab" and end == anchor):
            assert r[f"{key}_status"] == "undefined"
            assert r[f"{key}_reason"] is not None
            assert np.isnan(limit) and np.isnan(r[f"{key}_censored"])
        else:
            assert r[f"{key}_status"] == "censored"
            assert limit == end - anchor


@pytest.fixture
def saved_input_factory(tmp_path, monkeypatch):
    import wf5_windows.analyze_wf5 as analysis
    monkeypatch.setattr(analysis, "ROOT", tmp_path)

    def create(name):
        n, d = (10000, 7) if name == "Elec2" else (20000, 3 if name == "S4" else 10)
        dps = [10000] if name in ("S1", "S2", "S4") else []
        widths = [1000 if name == "S2" else 0] if dps else []
        base = tmp_path / analysis.STREAMS[name][0]
        for seed in range(1 if name == "Elec2" else 10):
            sd = base / f"seed_{seed}"
            sd.mkdir(parents=True)
            meta = dict(seed=seed, n=n, d=d, drift_points=dps, drift_widths=widths,
                        stream=dict(n=n, d=d))
            if name != "Elec2":
                meta["stream"].update(drift_at=10000 if dps else None,
                                      drift_width=widths[0] if widths else 0)
            (sd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            t = np.arange(50, n + 1, 50)
            np.savez(sd / "checkpoints.npz", t=t, G=np.tile(np.arange(d), (len(t), 1)))
        return base / "seed_0" / "meta.json"
    return create


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S4", "Elec2"])
def test_valid_stream_metadata(saved_input_factory, name):
    from wf5_windows.analyze_wf5 import load_runs
    saved_input_factory(name)
    runs, _, _ = load_runs(name)
    assert len(runs) == (1 if name == "Elec2" else 10)
    assert runs[0]["dps"] == ([10000] if name in ("S1", "S2", "S4") else [])
    assert runs[0]["widths"] == ([1000 if name == "S2" else 0] if runs[0]["dps"] else [])


@pytest.mark.parametrize("name,field,value", [
    ("S1", "drift_points", "missing"), ("S2", "drift_widths", "missing"),
    ("S4", "drift_points", []), ("S3", "drift_points", "missing"),
    ("S3", "drift_points", [10000]), ("Elec2", "drift_points", [5000]),
    ("S1", "drift_points", ["10000"]), ("S1", "drift_points", [True]),
    ("S1", "drift_points", [float("nan")]), ("S1", "drift_points", [float("inf")]),
    ("S1", "drift_points", [0]), ("S1", "drift_points", [20000]),
    ("S1", "drift_points", [10000, 9000]), ("S1", "drift_points", [10000, 10000]),
    ("S1", "drift_points", [9000]), ("S2", "drift_widths", []),
    ("S2", "drift_widths", [-1]), ("S2", "drift_widths", [float("inf")]),
    ("S2", "drift_widths", ["1000"]), ("S2", "drift_widths", [0]),
    ("S2", "drift_widths", [50000]),
])
def test_invalid_drift_metadata(saved_input_factory, name, field, value):
    from wf5_windows.analyze_wf5 import load_runs
    path = saved_input_factory(name)
    meta = json.loads(path.read_text())
    if value == "missing":
        meta.pop(field)
    else:
        meta[field] = value
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{name} seed 0: {field}"):
        load_runs(name)


@pytest.mark.parametrize("meta,field", [([], "meta.json"), ({"stream": []}, "stream"),
                                         ({}, "stream")])
def test_malformed_metadata_structure(saved_input_factory, meta, field):
    from wf5_windows.analyze_wf5 import load_runs
    path = saved_input_factory("S1")
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"S1 seed 0: {field} must be a dictionary"):
        load_runs("S1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and not fn.__code__.co_argcount:
            fn(); print("PASS", name)
