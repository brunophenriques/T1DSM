"""Turns one run's checkpoints into the metrics of SQ1-SQ3 (+ structural). Pure function, no model access.

Backward compatibility: with the defaults (abrupt drift, no `acc_*` keys in the analysis config) every
metric is computed exactly as in the first version of this file; tests/test_regression_saved_results.py
recomputes the saved results of configs/mve.json and configs/stationary_control.json and checks this.

Gradual drift (drift_widths > 0): the transition is [dp - w/2, dp + w/2]. Checkpoints whose 200-sample
buffer touches the transition are not "clean" (excluded from stability/validity), and recovery times are
counted from the END of the transition (PROTOCOL.md, "Drift ground truth").
"""
from __future__ import annotations
import numpy as np
from .metrics import tau_b, topk_jaccard, topk_precision, auc, is_constant
from .recovery import ranking_recovery, accuracy_recovery

NAN = float("nan")


def _nm(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size and not np.all(np.isnan(x)) else NAN


def analyze(arr: dict, truth, drift_points, a: dict, every: int, drift_widths=None) -> dict:
    t, G, A, acc = arr["t"], arr["G"], arr["A"], arr["acc_win"]
    T, flush = len(t), a["warmup_flush"]
    # k = a fixed number, or "truth": the number of truly relevant features at each checkpoint, for
    # streams whose concepts have relevant sets of different sizes (S5 Agrawal). PROTOCOL.md §5.
    if a["topk"] == "truth":
        if truth is None:
            raise ValueError('topk="truth" needs a ground-truth mask')
        kk = truth[t - 1].sum(axis=1).astype(int)
    else:
        kk = np.full(T, int(a["topk"]))
    dps = list(drift_points)
    widths = list(drift_widths) if drift_widths else [0] * len(dps)
    tr_start = [dp - w / 2.0 for dp, w in zip(dps, widths)]   # transition start (= dp when abrupt)
    tr_end = [dp + w / 2.0 for dp, w in zip(dps, widths)]     # transition end   (= dp when abrupt)

    starts = np.array([0] + tr_end, dtype=float)
    seg = np.searchsorted(starts, t - 1, side="right") - 1      # segment of the newest sample seen
    next_tr = np.array(tr_start + [np.inf], dtype=float)        # where the following transition begins
    clean = ((t - starts[seg]) >= flush) & ((t - 1) < next_tr[seg])   # buffer holds only this segment's data
    out = {"n_checkpoints": T, "struct/rules_mean": _nm(arr["n_rules"]),
           "struct/created_total": float(arr["created"][-1]), "struct/removed_total": float(arr["removed"][-1])}

    for name, W in (("G", G), ("A", A)):
        # SQ1: stability within stationary segments, at several lags (consecutive snapshots overlap 75%)
        for L in a["lags_samples"]:
            lc = max(1, L // every)
            taus, jacs = [], []
            for i in range(lc, T):
                j = i - lc
                if clean[i] and clean[j] and seg[i] == seg[j] and not is_constant(W[i]) and not is_constant(W[j]):
                    taus.append(tau_b(W[j], W[i])); jacs.append(topk_jaccard(W[j], W[i], kk[i]))
            out[f"stab/{name}/tau_b/lag{L}"] = _nm(taus)
            out[f"stab/{name}/jaccard/lag{L}"] = _nm(jacs)
        out[f"frac_constant/{name}"] = float(np.mean([is_constant(W[i]) for i in range(T)]))
        for s in range(len(starts)):
            m = clean & (seg == s)
            out[f"var/{name}/seg{s}"] = float(W[m].var(axis=0).mean()) if m.sum() > 1 else NAN

    if truth is not None:
        tr = truth[t - 1]
        for name, W in (("G", G), ("A", A)):
            prec = np.array([topk_precision(W[i], tr[i], kk[i]) for i in range(T)])
            auc_ = np.array([auc(W[i], tr[i]) for i in range(T)])
            for s in range(len(starts)):                         # SQ3: validity
                m = clean & (seg == s)
                out[f"valid/{name}/prec_at_k/seg{s}"] = _nm(prec[m])
                out[f"valid/{name}/auc/seg{s}"] = _nm(auc_[m])
            for n in range(1, len(dps) + 1):
                # SQ2: ranking recovery time, counted from the END of the transition (protocol metric).
                # t > base: the first sample after the drift is index base, seen at checkpoint t > base.
                # Censoring: a run that never recovers keeps the time to the last checkpoint of the
                # segment as a lower bound (recovery_censored), and recovered = 0.
                base, end = tr_end[n - 1], next_tr[n]
                idx = np.where((t > base) & (t < end))[0]
                r = ranking_recovery(t[idx], prec[idx], drift_end=base, threshold=a["recovery_precision"],
                                     consecutive=a["recovery_consecutive"])
                out[f"recovery/{name}/drift{n}"] = r["recovery"]
                out[f"recovered/{name}/drift{n}"] = float(r["recovered"])
                out[f"recovery_censored/{name}/drift{n}"] = r["censored_time"]
                # Same quantity counted from the drift point itself (the transition midpoint), so that
                # abrupt and gradual drifts are timed from the same moment. For a gradual drift the
                # protocol metric above has a floor of one checkpoint and cannot show a ranking that
                # re-oriented before the transition ended. Identical to the protocol metric when abrupt.
                dp = float(dps[n - 1])
                idx = np.where((t > dp) & (t < end))[0]
                r = ranking_recovery(t[idx], prec[idx], drift_end=dp, threshold=a["recovery_precision"],
                                     consecutive=a["recovery_consecutive"])
                out[f"recovery_from_dp/{name}/drift{n}"] = r["recovery"]
                out[f"recovery_from_dp_censored/{name}/drift{n}"] = r["censored_time"]

        pre_samples = a.get("acc_pre_samples", 5 * every)       # default = first version (5 checkpoints)
        acc_consec = a.get("acc_consecutive", 1)                 # default = first version (no persistence)
        trough_samples = a.get("acc_trough_samples")             # default = first version (whole segment)
        for n in range(1, len(dps) + 1):                         # accuracy recovery (separate from ranking recovery)
            base, end, ts = tr_end[n - 1], next_tr[n], tr_start[n - 1]
            mask = t < end
            r = accuracy_recovery(t[mask], acc[mask], drift_start=ts, drift_end=base, pre_samples=pre_samples,
                                  trough_samples=trough_samples, tolerance=a["acc_tol"], consecutive=acc_consec)
            out[f"acc/drop/drift{n}"] = r["accuracy_drop"]
            out[f"acc/recovery/drift{n}"] = r["recovery"]
            out[f"acc/recovered/drift{n}"] = float(r["recovered"])
            out[f"acc/recovery_censored/drift{n}"] = r["censored_time"]

    return out
