"""WF5 core: window analysis and restabilisation time of R1 (PROTOCOL.md §4, §5).

All functions take one run's checkpoint times `t` (samples seen, multiples of 50) and its relevance
matrix `G` (one row per checkpoint). They only read saved results; no model is run here.

Window analysis (§4)
    Non-overlapping windows of `width` samples: window j holds the checkpoints whose newest sample
    index t - 1 lies in [j*width, (j+1)*width). Its relevance vector is the mean of G over the CLEAN
    checkpoints in it (§4: the 200-sample buffer holds one concept only, using the same rule as
    relevance/analysis.py). Windows are compared with top-k Jaccard and weighted τ.

Restabilisation (§5)
    s[i] = weighted τ between G at checkpoint i and at i - lag (lag 200 = 4 checkpoints, disjoint buffers).
    Threshold = 10th percentile of s over clean pre-drift pairs. "Back above" presupposes a drop, so the
    search starts at the trough: the minimum of s within 1,000 samples after the drift (from the
    transition start to 1,000 samples after its end), the same construction as accuracy recovery (§10).
    Restabilised = first checkpoint from the trough on where s >= threshold for 3 checkpoints in a row.
    A trough not below the threshold means the ranking never destabilised (time 0). Times are counted in samples from an anchor:
    the transition end (protocol convention, §10) and the drift point (= transition midpoint, as
    `recovery_from_dp`). For an abrupt drift both are the same.
"""
from __future__ import annotations
import numpy as np
from relevance.metrics import topk_jaccard, is_constant, first_sustained
from relevance.stability import weighted_tau

WIDTH = 1000
LAG = 200
PCT = 10
CONSECUTIVE = 3
TROUGH = 1000


def transitions(drift_points, drift_widths=None):
    """(start, end) of each transition; start == end for an abrupt drift."""
    dps = list(drift_points or [])
    ws = list(drift_widths or [0] * len(dps))
    return [(dp - w / 2.0, dp + w / 2.0) for dp, w in zip(dps, ws)]


def clean_mask(t, drift_points=(), drift_widths=None, flush=200):
    """Same rule as relevance/analysis.py: segment of the newest sample, buffer holds only that segment.
    Returns (seg, clean)."""
    t = np.asarray(t, dtype=float)
    tr = transitions(drift_points, drift_widths)
    starts = np.array([0.0] + [e for _, e in tr])
    next_tr = np.array([s for s, _ in tr] + [np.inf])
    seg = np.searchsorted(starts, t - 1, side="right") - 1
    clean = ((t - starts[seg]) >= flush) & ((t - 1) < next_tr[seg])
    return seg, clean


def window_means(t, G, clean, n_samples, width=WIDTH):
    """Mean relevance per window over clean checkpoints. Returns (W, n_clean, lo) where W has shape
    (n_windows, d); a window without clean checkpoints is NaN."""
    t = np.asarray(t); G = np.asarray(G, dtype=float); clean = np.asarray(clean, bool)
    n_win = int(np.ceil(n_samples / width))
    win = (t - 1) // width
    W = np.full((n_win, G.shape[1]), np.nan); n_clean = np.zeros(n_win, int)
    for j in range(n_win):
        m = (win == j) & clean
        n_clean[j] = int(m.sum())
        if n_clean[j]:
            W[j] = G[m].mean(axis=0)
    return W, n_clean, np.arange(n_win) * width


def compare(a, b, k):
    """(top-k Jaccard, weighted τ) of two relevance vectors; NaN if either is missing or constant."""
    if np.isnan(a).any() or np.isnan(b).any() or is_constant(a) or is_constant(b):
        return np.nan, np.nan
    return topk_jaccard(a, b, k), weighted_tau(a, b)


def last_pre_drift_window(drift_points, drift_widths=None, width=WIDTH):
    """Index of the last window that ends at or before the first transition starts (None if no drift)."""
    tr = transitions(drift_points, drift_widths)
    if not tr:
        return None
    return int(tr[0][0] // width) - 1


def window_analysis(t, G, k, n_samples, drift_points=(), drift_widths=None, flush=200, width=WIDTH):
    """All window comparisons of one run.
    Returns dict(W, n_clean, consecutive=list of rows, first_last=(jac, wtau), pre_last=(jac, wtau)|None)."""
    seg, clean = clean_mask(t, drift_points, drift_widths, flush)
    W, n_clean, lo = window_means(t, G, clean, n_samples, width)
    tr = transitions(drift_points, drift_widths)
    rows = []
    for j in range(1, len(W)):
        a0, b1 = lo[j - 1], lo[j] + width          # samples spanned by the pair
        crosses = any(s < b1 and e > a0 for s, e in tr)   # a window starting at an abrupt dp is post-drift
        jac, wt = compare(W[j - 1], W[j], k)
        rows.append(dict(win_a=j - 1, win_b=j, start_a=int(a0), start_b=int(lo[j]),
                         jaccard=jac, wtau=wt, crosses_drift=bool(crosses)))
    valid = np.flatnonzero(n_clean > 0)
    first, last = int(valid[0]), int(valid[-1])
    out = dict(W=W, n_clean=n_clean, consecutive=rows, first=first, last=last,
               first_last=compare(W[first], W[last], k), pre=None, pre_last=None)
    p = last_pre_drift_window(drift_points, drift_widths, width)
    if p is not None:
        out["pre"] = p
        out["pre_last"] = compare(W[p], W[last], k)
    return out


def lag_wtau(G, lag_ck):
    """s[i] = weighted τ(G[i - lag_ck], G[i]); NaN for i < lag_ck or a constant vector."""
    G = np.asarray(G, dtype=float); s = np.full(len(G), np.nan)
    for i in range(lag_ck, len(G)):
        if not (is_constant(G[i]) or is_constant(G[i - lag_ck])):
            s[i] = weighted_tau(G[i - lag_ck], G[i])
    return s


def restabilisation(t, G, drift_points, drift_widths=None, every=50, lag=LAG, pct=PCT,
                    consecutive=CONSECUTIVE, trough_samples=TROUGH, flush=200):
    """Restabilisation time after the first drift (see module docstring).

    Returns dict with threshold, n_pre, trough (min wτ), t_trough, dropped (trough < threshold),
    restabilised, and the time from the transition end (`restab`, protocol) and from the drift point
    (`restab_from_dp`). When the trough is not below the threshold, both times are descriptive zeros, dropped=False,
    restabilised=False and restab_status="no_destabilisation" (no recovery event).
    Each anchor has its own status: recovered, censored, or undefined. Undefined cases
    include an explicit reason and NaN times. The midpoint search starts at the trough
    and can give a negative signed offset; the protocol search starts after transition end.
    An anchor without eligible observations is undefined, independently of the other anchor.
    Censoring requires an eligible observation at or after the anchor; its limit is never negative.
    When it never comes back, the times are NaN and `*_censored` holds the time to the last checkpoint."""
    t = np.asarray(t, dtype=float)
    lc = lag // every
    s = lag_wtau(G, lc)
    seg, clean = clean_mask(t, drift_points, drift_widths, flush)
    (tr_s, tr_e), dp = transitions(drift_points, drift_widths)[0], float(list(drift_points)[0])
    idx = np.arange(len(t))
    pre_pairs = (idx >= lc) & clean & np.roll(clean, lc) & (seg == 0) & np.roll(seg == 0, lc)
    pre_vals = s[pre_pairs & np.isfinite(s)]
    thr = float(np.percentile(pre_vals, pct)) if pre_vals.size else np.nan
    out = dict(threshold=thr, n_pre=int(pre_vals.size), trough=np.nan, t_trough=np.nan, dropped=False,
               restabilised=False, restab_status="undefined", restab_from_dp_status="undefined",
               restab_reason=None, restab_from_dp_reason=None,
               restab_censor_limit=np.nan, restab_from_dp_censor_limit=np.nan,
               restab=np.nan, restab_from_dp=np.nan,
               restab_censored=np.nan, restab_from_dp_censored=np.nan)
    search = np.flatnonzero((t > tr_s) & (t <= tr_e + trough_samples) & np.isfinite(s))
    if not np.isfinite(thr) or search.size < consecutive:
        reason = "undefined_pre_drift_threshold" if not np.isfinite(thr) else "insufficient_post_drift_data"
        out.update(restab_reason=reason, restab_from_dp_reason=reason)
        return out
    d0 = int(search[np.argmin(s[search])])
    out.update(trough=float(s[d0]), t_trough=float(t[d0]))
    if s[d0] >= thr:                                        # never destabilised
        out.update(restab_status="no_destabilisation", restab_from_dp_status="no_destabilisation",
                   restabilised=False, restab=0.0, restab_from_dp=0.0,
                   restab_censored=0.0, restab_from_dp_censored=0.0)
        if not np.any((t > tr_e) & np.isfinite(s)):
            out.update(restab_status="undefined", restab_reason="insufficient_post_transition_data",
                       restab=np.nan, restab_censored=np.nan)
        return out
    out["dropped"] = True
    above = np.isfinite(s) & (s >= thr)
    # protocol: searched after the transition end, timed from it (a gradual drift has a 50-sample floor);
    # from_dp: searched from the trough, timed from the drift point. Identical for an abrupt drift.
    for key, start, anchor in (("restab", max(d0, int(np.searchsorted(t, tr_e, side="right"))), tr_e),
                               ("restab_from_dp", d0, dp)):
        hit = first_sustained(above[start:], consecutive)
        if hit is None:
            # Midpoint recovery may be observed before dp, but censoring cannot start
            # before its anchor. The protocol slice already excludes transition end.
            eligible = np.isfinite(s[start:]) & (t[start:] >= anchor)
            if not np.any(eligible):
                out[f"{key}_reason"] = ("insufficient_post_transition_data" if key == "restab"
                                        else "insufficient_post_midpoint_data")
                continue
            out[f"{key}_status"] = "censored"
            out[f"{key}_censor_limit"] = float(t[-1] - anchor)
            out[f"{key}_censored"] = float(t[-1] - anchor)
            continue
        out[f"{key}_status"] = "recovered"
        out[key] = out[f"{key}_censored"] = float(t[start + hit]) - anchor
    out["restabilised"] = bool(np.isfinite(out["restab"]))
    return out
