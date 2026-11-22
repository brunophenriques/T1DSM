"""SQ1 stability metrics on a sequence of relevance vectors (rows of G, one row per checkpoint)."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, weightedtau
from .metrics import tau_b, topk_jaccard, topk_idx, is_constant


def spearman(a, b) -> float:
    return float(spearmanr(a, b)[0])


def weighted_tau(a, b) -> float:
    """scipy.stats.weightedtau (hyperbolic weights): disagreements at the top of the ranking cost more."""
    return float(weightedtau(a, b)[0])


def metric_fns(k: int) -> dict:
    return {"jaccard": lambda a, b: topk_jaccard(a, b, k), "tau_b": tau_b,
            "weighted_tau": weighted_tau, "spearman": spearman}


def segments(t, drift_points, flush):
    """seg[i]: stationary segment of checkpoint i. clean[i]: buffer holds only that segment's data."""
    starts = np.array([0] + list(drift_points))
    seg = np.searchsorted(starts, np.asarray(t) - 1, side="right") - 1
    return seg, (np.asarray(t) - starts[seg]) >= flush


def lagged_series(G, fn, lag_ck, valid=None, seg=None):
    """out[i] = fn(G[i-lag_ck], G[i]); NaN if undefined (constant vector), invalid, or across segments."""
    G = np.asarray(G); out = np.full(len(G), np.nan)
    for i in range(lag_ck, len(G)):
        j = i - lag_ck
        if valid is not None and not (valid[i] and valid[j]):
            continue
        if seg is not None and seg[i] != seg[j]:
            continue
        if is_constant(G[i]) or is_constant(G[j]):
            continue
        out[i] = fn(G[j], G[i])
    return out


def topk_switch_rate(G, k, every, valid=None, seg=None) -> float:
    """Changes of the top-k SET between consecutive checkpoints, per 1000 samples."""
    n_pairs = n_sw = 0
    for i in range(1, len(G)):
        j = i - 1
        if valid is not None and not (valid[i] and valid[j]):
            continue
        if seg is not None and seg[i] != seg[j]:
            continue
        if is_constant(G[i]) or is_constant(G[j]):
            continue
        n_pairs += 1
        n_sw += set(topk_idx(G[i], k).tolist()) != set(topk_idx(G[j], k).tolist())
    return float("nan") if n_pairs == 0 else n_sw / (n_pairs * every) * 1000.0


def per_feature_sd(G, valid=None, seg=None):
    """SD of each feature's weight over checkpoints, computed inside each segment and averaged."""
    G = np.asarray(G)
    seg = np.zeros(len(G), int) if seg is None else np.asarray(seg)
    valid = np.ones(len(G), bool) if valid is None else np.asarray(valid)
    sds = [G[valid & (seg == s)].std(axis=0, ddof=1) for s in np.unique(seg[valid]) if (valid & (seg == s)).sum() > 1]
    return np.mean(sds, axis=0) if sds else np.full(G.shape[1], np.nan)


def rolling_nanmean(x, w):
    x = np.asarray(x, dtype=float); out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1): i + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out
