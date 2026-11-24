"""R5: permutation importance (PROTOCOL.md §3, a model-based cross-check of the relevance scores).

Every `every` samples, on the last `window` samples seen: the accuracy drop when one feature's column is
permuted, averaged over `repeats` permutations. Prediction only (model.predict never changes the model),
with its own random generator, so the prequential run is identical with or without R5.
"""
from __future__ import annotations
import numpy as np
from .metrics import topk_precision, auc, topk_idx

DEFAULT = {"every": 500, "window": 200, "repeats": 5}


def permutation_importance(model, X, y, rng, repeats=5):
    """Return (importance per feature, baseline accuracy). importance_j = baseline accuracy - mean
    accuracy with column j permuted. Uses only model.predict."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    base = float(np.mean(np.asarray(model.predict(X)).ravel() == y))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            drops.append(base - float(np.mean(np.asarray(model.predict(Xp)).ravel() == y)))
        imp[j] = float(np.mean(drops))
    return imp, base


def analyze_permutation(perm: dict, truth, drift_points, drift_widths, G_t, G, a: dict) -> dict:
    """Validity of R5 in clean segments, and agreement with R1 at the same checkpoints.

    A permutation checkpoint is clean when its `window` samples all come from one segment (after the end
    of the previous transition and before the next one starts). k follows a["topk"] (a number or "truth")."""
    t, P = np.asarray(perm["t"]), np.asarray(perm["P"])
    out = {}
    if len(t) == 0 or truth is None:
        return out
    window = int(perm["window"])
    dps = list(drift_points)
    widths = list(drift_widths) if drift_widths else [0] * len(dps)
    tr_end = [dp + w / 2.0 for dp, w in zip(dps, widths)]
    tr_start = [dp - w / 2.0 for dp, w in zip(dps, widths)]
    starts = np.array([0] + tr_end, dtype=float)
    seg = np.searchsorted(starts, t - 1, side="right") - 1
    next_tr = np.array(tr_start + [np.inf], dtype=float)
    clean = ((t - window) >= starts[seg]) & ((t - 1) < next_tr[seg])
    tr = truth[t - 1]
    kk = tr.sum(axis=1).astype(int) if a["topk"] == "truth" else np.full(len(t), int(a["topk"]))
    prec = np.array([topk_precision(P[i], tr[i], kk[i]) for i in range(len(t))])
    auc_ = np.array([auc(P[i], tr[i]) for i in range(len(t))])
    gi = {int(v): i for i, v in enumerate(G_t)}
    # Agreement with R1: the share of R5's top k that is also in R1's top k. R5 often has ties (permuting
    # a feature changes no prediction -> importance exactly 0), and a plain top-k would then pick the
    # lowest feature indices and fake an agreement. So the share is tie-aware, like topk_precision:
    # ties at the k-th place count at their expected value (a constant R5 scores chance, k/d).
    def overlap(i):
        ref = np.zeros(P.shape[1])
        ref[topk_idx(G[gi[int(t[i])]], kk[i])] = 1.0
        return topk_precision(P[i], ref, kk[i])
    agree = np.array([overlap(i) if int(t[i]) in gi else np.nan for i in range(len(t))])
    for s in range(len(starts)):
        m = clean & (seg == s)
        nm = lambda x: float(np.nanmean(x[m])) if m.any() and not np.all(np.isnan(x[m])) else float("nan")
        out[f"valid/P/prec_at_k/seg{s}"] = nm(prec)
        out[f"valid/P/auc/seg{s}"] = nm(auc_)
        out[f"agree/P_vs_G/topk_overlap/seg{s}"] = nm(agree)
        out[f"perm/base_acc/seg{s}"] = nm(np.asarray(perm["base_acc"], dtype=float))
        out[f"perm/n_clean/seg{s}"] = int(m.sum())
    return out
