"""Recovery after drift: ranking recovery (SQ2) and accuracy recovery, with right-censoring.

PROTOCOL.md section 10. Both functions take the checkpoint times `t` and one value per checkpoint.
A run that never recovers is right-censored: `recovery` is NaN, `recovered` is False and
`censored_time` is the time from the start point to the last checkpoint (a lower bound).

Moved here from wf2_drift_validity/recovery_metrics.py so that the core package does not depend on
a workflow folder; that module re-exports these functions.
"""
from __future__ import annotations
import numpy as np
from .metrics import first_sustained


def ranking_recovery(t, precision, drift_end, threshold=1.0, consecutive=3) -> dict:
    """First checkpoint after `drift_end` where `precision >= threshold` holds for `consecutive`
    checkpoints in a row, counted in samples from `drift_end`. NaN precision counts as not recovered."""
    t = np.asarray(t, dtype=float)
    precision = np.asarray(precision, dtype=float)
    post = t > drift_end
    times, values = t[post], precision[post]
    index = first_sustained(values >= threshold, consecutive) if times.size else None
    if index is None:
        return {"recovery": np.nan, "recovered": False,
                "censored_time": float(times[-1] - drift_end) if times.size else np.nan}
    recovery = float(times[index] - drift_end)
    return {"recovery": recovery, "recovered": True, "censored_time": recovery}


def accuracy_recovery(t, accuracy, drift_start, drift_end=None, pre_samples=1000, trough_samples=1000,
                      tolerance=0.02, consecutive=3) -> dict:
    """Accuracy recovery (predictive streams only).

    Pre-drift level = mean accuracy over checkpoints in (drift_start - pre_samples, drift_start].
    Trough = minimum within `trough_samples` after `drift_end` (None = the whole post-drift part).
    Recovered = first checkpoint from the trough on where accuracy >= pre-drift - tolerance for
    `consecutive` checkpoints in a row, counted in samples from `drift_end`.
    Without a pre-drift level the recovery is undefined: everything stays NaN (not censored)."""
    t = np.asarray(t, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)
    if drift_end is None:
        drift_end = drift_start
    pre = (t <= drift_start) & (t > drift_start - pre_samples)
    post = t > drift_end
    times, values = t[post], accuracy[post]
    result = {"recovery": np.nan, "recovered": False, "censored_time": np.nan, "accuracy_drop": np.nan}
    if not pre.any() or not times.size or np.all(np.isnan(accuracy[pre])):
        return result
    result["censored_time"] = float(times[-1] - drift_end)
    pre_mean = float(np.nanmean(accuracy[pre]))
    m = len(values) if trough_samples is None else max(1, int(np.sum(times <= drift_end + trough_samples)))
    trough_idx = int(np.nanargmin(values[:m])) if not np.all(np.isnan(values[:m])) else 0
    result["accuracy_drop"] = float(pre_mean - values[trough_idx])
    index = first_sustained(values[trough_idx:] >= pre_mean - tolerance, consecutive)
    if index is not None:
        recovery = float(times[trough_idx + index] - drift_end)
        result.update(recovery=recovery, recovered=True, censored_time=recovery)
    return result
