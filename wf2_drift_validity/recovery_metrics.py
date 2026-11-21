import numpy as np


def first_sustained(condition, consecutive=3):
    """Return the first index of a sustained True sequence."""
    count = 0

    for i, value in enumerate(condition):
        if value:
            count += 1
            if count == consecutive:
                return i - consecutive + 1
        else:
            count = 0

    return None


def ranking_recovery(
    t,
    precision,
    drift_end,
    threshold=1.0,
    consecutive=3,
):
    """Measure ranking recovery after concept drift."""
    t = np.asarray(t)
    precision = np.asarray(precision)

    post = t > drift_end
    times = t[post]
    values = precision[post]

    index = first_sustained(
        values >= threshold,
        consecutive,
    )

    if index is None:
        return {
            "recovery": np.nan,
            "recovered": False,
            "censored_time": (
                float(times[-1] - drift_end)
                if times.size else np.nan
            ),
        }

    recovery = float(times[index] - drift_end)

    return {
        "recovery": recovery,
        "recovered": True,
        "censored_time": recovery,
    }


def accuracy_recovery(
    t,
    accuracy,
    drift_start,
    drift_end=None,
    pre_samples=1000,
    trough_samples=1000,
    tolerance=0.02,
    consecutive=3,
):
    """Measure sustained accuracy recovery after drift."""
    t = np.asarray(t)
    accuracy = np.asarray(accuracy)

    if drift_end is None:
        drift_end = drift_start

    pre = (
        (t <= drift_start)
        & (t > drift_start - pre_samples)
    )

    post = t > drift_end

    times = t[post]
    values = accuracy[post]

    result = {
        "recovery": np.nan,
        "recovered": False,
        "censored_time": (
            float(times[-1] - drift_end)
            if times.size else np.nan
        ),
        "accuracy_drop": np.nan,
    }

    if not pre.any() or not times.size:
        return result

    pre_mean = np.nanmean(accuracy[pre])

    if trough_samples is None:
        m = len(values)
    else:
        m = max(
            1,
            int(np.sum(times <= drift_end + trough_samples)),
        )

    trough_idx = int(np.argmin(values[:m]))

    result["accuracy_drop"] = float(
        pre_mean - values[trough_idx]
    )

    threshold = pre_mean - tolerance

    index = first_sustained(
        values[trough_idx:] >= threshold,
        consecutive,
    )

    if index is not None:
        recovery = float(
            times[trough_idx + index] - drift_end
        )

        result.update(
            recovery=recovery,
            recovered=True,
            censored_time=recovery,
        )

    return result