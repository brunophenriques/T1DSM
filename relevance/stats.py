"""Statistics for the analysis plan: bootstrap CIs, block bootstrap, paired Wilcoxon + rank-biserial, Holm."""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, rankdata


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    """Percentile CI of `stat` resampling observations independently (use over seeds)."""
    x = _clean(x); rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def block_bootstrap_ci(x, block, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    """Moving-block bootstrap for an autocorrelated series (use within one run, e.g. Elec2)."""
    x = _clean(x); n = len(x); block = int(max(1, min(block, n))); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        vals[b] = stat(np.concatenate([x[s:s + block] for s in starts])[:n])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def rank_biserial_paired(a, b) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1]; positive if a > b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float); d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d)); wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def wilcoxon_paired(a, b) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if np.all(a == b):
        return dict(n=len(a), W=0.0, p=1.0, r_rb=0.0)
    res = wilcoxon(a, b)
    return dict(n=len(a), W=float(res.statistic), p=float(res.pvalue), r_rb=rank_biserial_paired(a, b))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, dtype=float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def shuffle_rank_pvalue(observed, null, alternative="less") -> float:
    """Exact one-sided permutation-style p of an observed value against a null sample (smallest possible: 1/(n+1))."""
    null = _clean(null)
    hits = (null <= observed).sum() if alternative == "less" else (null >= observed).sum()
    return float((1 + hits) / (len(null) + 1))
