"""Unit tests for the SQ1 stability metrics and statistics. Run: python tests/test_stability.py  (or pytest tests/)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance.metrics import tau_b, topk_jaccard
from relevance.stability import (weighted_tau, spearman, segments, lagged_series, topk_switch_rate,
                                 per_feature_sd, rolling_nanmean, metric_fns)
from relevance.stats import (bootstrap_ci, block_bootstrap_ci, rank_biserial_paired, wilcoxon_paired,
                             holm, shuffle_rank_pvalue)


def test_stable_topk_with_shuffled_irrelevant_features():
    """Top-k set never changes, the order of the irrelevant features is random: tau-b low, Jaccard = 1."""
    rng = np.random.default_rng(0)
    w1 = np.array([1.0, 0.9] + list(0.20 + 0.01 * np.arange(8)))
    taus, wts = [], []
    for _ in range(300):
        w2 = w1.copy(); w2[2:] = rng.permutation(w2[2:])
        assert topk_jaccard(w1, w2, 2) == 1.0
        taus.append(tau_b(w1, w2)); wts.append(weighted_tau(w1, w2))
    assert np.mean(taus) < 0.5                  # about 0.37: the full ranking looks unstable
    assert np.mean(wts) > np.mean(taus) + 0.1   # weighted tau discounts the noisy tail


def test_identical_and_disjoint():
    w = np.array([5., 4, 3, 2, 1])
    assert abs(tau_b(w, w) - 1) < 1e-9 and abs(spearman(w, w) - 1) < 1e-9 and abs(weighted_tau(w, w) - 1) < 1e-9
    assert topk_jaccard(np.array([3, 2, 0, 0.]), np.array([0, 0, 3, 2.]), 2) == 0.0


def test_lagged_series_masks_and_constants():
    G = np.array([[3, 2, 1.], [3, 2, 1.], [1, 1, 1.], [3, 2, 1.], [3, 2, 1.], [1, 2, 3.]])
    fn = metric_fns(2)["jaccard"]
    out = lagged_series(G, fn, 1)
    assert np.isnan(out[0]) and out[1] == 1.0 and np.isnan(out[2]) and np.isnan(out[3])   # constant vector excluded
    assert out[4] == 1.0 and out[5] == 1 / 3
    t = np.array([50, 100, 150, 200, 250, 300]); seg, clean = segments(t, [150], flush=100)
    out2 = lagged_series(G, fn, 1, valid=clean, seg=seg)
    assert np.isnan(out2[3])                    # pair (150, 200) crosses the drift at 150


def test_switch_rate_and_per_feature_sd():
    A, B = [3, 2, 1, 0.], [3, 1, 2, 0.]         # same top-1, different top-2
    G = np.array([A, A, B, B, A])
    assert topk_switch_rate(G, 2, every=50) == 2 / (4 * 50) * 1000      # 2 switches in 4 transitions
    assert topk_switch_rate(G, 1, every=50) == 0.0
    sd = per_feature_sd(np.array([[1., 0], [3., 0], [5., 0]]))
    assert np.allclose(sd, [2.0, 0.0])


def test_rolling_nanmean():
    assert np.allclose(rolling_nanmean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
    assert np.isnan(rolling_nanmean([np.nan, np.nan], 2)).all()


def test_bootstrap_and_block_bootstrap():
    rng = np.random.default_rng(1)
    x = np.zeros(400); x[0] = 0
    for i in range(1, 400):
        x[i] = 0.95 * x[i - 1] + rng.normal()                           # strongly autocorrelated series
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=0); blo, bhi = block_bootstrap_ci(x, block=40, n_boot=2000, seed=0)
    assert (bhi - blo) > 1.5 * (hi - lo)        # ignoring autocorrelation makes the CI too narrow
    assert bootstrap_ci(x, n_boot=500, seed=3) == bootstrap_ci(x, n_boot=500, seed=3)


def test_wilcoxon_rank_biserial_holm_shuffle():
    a, b = np.arange(10) + 1.0, np.arange(10)
    out = wilcoxon_paired(a, b)
    assert out["r_rb"] == 1.0 and out["p"] < 0.01 and out["n"] == 10
    assert rank_biserial_paired(b, a) == -1.0 and wilcoxon_paired(a, a)["p"] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert shuffle_rank_pvalue(0.1, np.linspace(0.5, 0.9, 10)) == 1 / 11     # smallest possible p with 10 shuffles
    assert shuffle_rank_pvalue(0.95, np.linspace(0.5, 0.9, 10)) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
