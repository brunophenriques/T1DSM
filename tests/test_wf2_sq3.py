"""Tests for the WF2 SQ3 additions: Agrawal stream (S5), noisy copy (S6), per-segment k, the oracle's
selection rule and R5 permutation importance.

Run with `python tests/test_wf2_sq3.py` or `pytest tests/`.
"""
import copy, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.streams import (agrawal_stream, add_noisy_copy, build_stream, gauss_shift_stream,
                               AGRAWAL_FEATURES, AGRAWAL_RELEVANT, agrawal_relevant_idx)
from relevance.analysis import analyze
from relevance.permutation import permutation_importance, analyze_permutation
from run_relevance import run_one

try:
    import river  # noqa: F401
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


def _skip_without_river():
    if not HAS_RIVER:
        try:
            import pytest
            pytest.skip("needs river")
        except ImportError:
            return True
    return False


def test_agrawal_stream_matches_river_and_has_the_right_truth():
    if _skip_without_river():
        return
    from river.datasets.synth import Agrawal
    st = agrawal_stream(n=1200, function_before=2, function_after=6, drift_at=600, seed=5)
    assert st.X.shape == (1200, 9) and st.X.min() >= 0.0 and st.X.max() <= 1.0
    for f, sl in ((2, slice(0, 600)), (6, slice(600, 1200))):
        ys = np.array([y for (_, y), _i in zip(Agrawal(classification_function=f, seed=5), range(1200))])
        assert np.array_equal(ys[sl], st.y[sl]), f
    before = {AGRAWAL_FEATURES[j] for j in np.where(st.truth[0])[0]}
    after = {AGRAWAL_FEATURES[j] for j in np.where(st.truth[-1])[0]}
    assert before == {"age", "elevel"} and after == {"salary", "commission", "loan"}
    assert st.drift_points == [600] and st.drift_widths == [0]
    # stationary version: no drift, truth constant
    s0 = agrawal_stream(n=300, function_before=0, seed=5)
    assert s0.drift_points == [] and (s0.truth == s0.truth[0]).all() and s0.truth[0].sum() == 1


def test_agrawal_relevant_sets_are_the_generator_arguments():
    assert set(AGRAWAL_RELEVANT) == set(range(10))
    for f, names in AGRAWAL_RELEVANT.items():
        assert all(n in AGRAWAL_FEATURES for n in names)
        assert len(agrawal_relevant_idx(f)) == len(set(names))
    assert set(AGRAWAL_RELEVANT[2]).isdisjoint(AGRAWAL_RELEVANT[6])
    assert set(AGRAWAL_RELEVANT[0]).isdisjoint(AGRAWAL_RELEVANT[6])


def test_noisy_copy_leaves_the_base_stream_untouched():
    base = gauss_shift_stream(n=800, drift_at=400, seed=3)
    st = add_noisy_copy(base, source=0, noise_sd=0.02, seed=3)
    assert st.X.shape == (800, 11) and np.array_equal(st.X[:, :10], base.X) and np.array_equal(st.y, base.y)
    assert not st.truth[:, 10].any() and np.array_equal(st.truth[:, :10], base.truth)
    assert np.corrcoef(st.X[:, 0], st.X[:, 10])[0, 1] > 0.95
    cfg = {"type": "gauss_shift", "n": 800, "d": 10, "delta": 0.3, "rel_before": [0, 1, 2],
           "rel_after": [5, 6, 7], "drift_at": 400, "copy": {"source": 0, "noise_sd": 0.02}}
    assert np.array_equal(build_stream(cfg, 3).X, st.X)


def test_topk_truth_uses_the_number_of_relevant_features_per_segment():
    every, T = 50, 80
    t = np.arange(1, T + 1) * every
    # 5 features; segment 0: {0} relevant (k=1), segment 1 from 2000: {2,3,4} relevant (k=3)
    G = np.where((t <= 2000)[:, None], np.array([1.0, 0.5, 0.2, 0.2, 0.2]), np.array([0.2, 0.3, 1.0, 0.9, 0.8]))
    arr = dict(t=t, G=G, A=np.full((T, 5), 0.2), acc_win=np.full(T, 0.7), n_rules=np.full(T, 5),
               created=np.arange(T), removed=np.arange(T))
    truth = np.zeros((t[-1], 5), dtype=int)
    truth[:2000, 0] = 1
    truth[2000:, [2, 3, 4]] = 1
    a = {"lags_samples": [200], "topk": "truth", "recovery_precision": 1.0, "recovery_consecutive": 3,
         "warmup_flush": 200, "acc_tol": 0.02}
    m = analyze(arr, truth, [2000], a, every)
    assert m["valid/G/prec_at_k/seg0"] == 1.0 and m["valid/G/prec_at_k/seg1"] == 1.0
    m2 = analyze(arr, truth, [2000], dict(a, topk=2), every)      # a fixed k = 2 misjudges segment 0
    assert m2["valid/G/prec_at_k/seg0"] == 0.5
    try:
        analyze(arr, None, [], a, every)
        raise AssertionError("topk=truth without truth must fail")
    except ValueError:
        pass


def test_selection_rule_picks_the_lowest_disjoint_detected_pair():
    from wf2_drift_validity.oracle_check import select_pair
    prec = {f: 0.5 for f in range(10)}
    prec.update({2: 1.0, 6: 1.0, 7: 1.0})                    # 2-7 share elevel; 2-6 disjoint; 6-7 share salary
    assert select_pair(prec)["chosen"] == [2, 6]
    prec[0] = 1.0                                            # 0-6 and 0-7 are disjoint too: lowest index wins
    s = select_pair(prec)
    assert s["chosen"] == [0, 6] and s["valid_pairs"] == [[0, 6], [0, 7], [2, 6]]
    assert select_pair({f: 0.0 for f in range(10)})["chosen"] is None


class _OnlyFeatureZero:
    """Predicts 1[x0 > 0.5]; records nothing, so any change of state would be visible."""
    def predict(self, X):
        return (np.asarray(X)[:, 0] > 0.5).astype(int)


def test_permutation_importance_finds_the_used_feature():
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = (X[:, 0] > 0.5).astype(int)
    imp, base = permutation_importance(_OnlyFeatureZero(), X, y, np.random.default_rng(1), repeats=5)
    assert base == 1.0 and imp[0] > 0.3 and np.all(imp[1:] == 0.0)


def test_permutation_importance_does_not_change_the_run():
    """R5 is prediction-only with its own generator: predictions, weights and rules are identical."""
    cfg = json.load(open(ROOT / "configs" / "protocol" / "s1_gauss_abrupt.json"))
    cfg["stream"].update(n=1200, drift_at=600)
    params = cfg["variants"]["open"]
    plain = run_one(cfg, 0, log=True, model_cfg=params)
    cfg_p = dict(copy.deepcopy(cfg), perm_importance={"every": 300, "window": 200, "repeats": 2})
    withp = run_one(cfg_p, 0, log=True, model_cfg=params)
    assert np.array_equal(plain["y_pred"], withp["y_pred"]) and np.array_equal(plain["n_rules"], withp["n_rules"])
    a, b = plain["logger"].arrays(), withp["logger"].arrays()
    assert np.array_equal(a["G"], b["G"]) and np.array_equal(a["A"], b["A"])
    p = withp["perm"]
    assert p["t"] == [300, 600, 900, 1200] and np.asarray(p["P"]).shape == (4, 10)
    assert plain["perm"] is None


def test_analyze_permutation_clean_windows_and_k():
    truth = np.zeros((2000, 4), dtype=int)
    truth[:1000, [0, 1]] = 1
    truth[1000:, [2, 3]] = 1
    perm = dict(t=[500, 1000, 1100, 1500, 2000], window=200,
                P=[[.3, .2, 0, 0], [.3, .2, 0, 0], [.1, 0, 0, .2], [0, 0, .3, .2], [0, 0, .2, .3]],
                base_acc=[.8, .8, .6, .7, .8])
    G_t = np.array([500, 1000, 1100, 1500, 2000])
    G = np.array([[1, .9, .2, .2], [1, .9, .2, .2], [.5, .2, .9, .8], [.2, .2, 1, .9], [.2, .2, .9, 1]])
    m = analyze_permutation(perm, truth, [1000], [0], G_t, G, {"topk": 2})
    assert m["perm/n_clean/seg0"] == 2 and m["perm/n_clean/seg1"] == 2     # 1100 is not clean (window touches drift)
    assert m["valid/P/prec_at_k/seg0"] == 1.0 and m["valid/P/prec_at_k/seg1"] == 1.0
    assert m["agree/P_vs_G/jaccard/seg1"] == 1.0
    # A constant R5 (model ignores its input): precision is chance, agreement is undefined, not 1.0
    flat = dict(perm, P=[[0.0] * 4] * 5)
    m = analyze_permutation(flat, truth, [1000], [0], G_t, G, {"topk": 2})
    assert m["valid/P/prec_at_k/seg0"] == 0.5 and np.isnan(m["agree/P_vs_G/jaccard/seg0"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
