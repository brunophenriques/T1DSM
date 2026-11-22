"""Streams with known feature relevance (synthetic) and a leakage-free loader (real).

River's Agrawal/SEA drift changes the decision rule, not necessarily WHICH features
matter, so we use our own generator where the relevant set is explicit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import warnings
import numpy as np


@dataclass
class Stream:
    X: np.ndarray                      # (n, d), already in [0, 1] without future information
    y: np.ndarray                      # (n,) int labels
    truth: Optional[np.ndarray]        # (n, d) binary relevance at each time step, or None (real data)
    drift_points: List[int] = field(default_factory=list)   # sample indices where relevance changes
    feature_names: Optional[List[str]] = None


def relevance_drift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2),
                           rel_after=(5, 6, 7), label_noise=0.0, seed=0) -> Stream:
    """y = 1[ sum of the relevant features > |R|/2 ]. At `drift_at` the relevant set R switches
    abruptly from rel_before to rel_after. drift_at=None gives a stationary control stream."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))

    def label(rel):
        rel = list(rel)
        return (X[:, rel].sum(axis=1) > len(rel) / 2.0).astype(int)

    y = label(rel_before)
    truth = np.zeros((n, d), dtype=int)
    truth[:, list(rel_before)] = 1
    drift_points: List[int] = []
    if drift_at is not None and drift_at < n:
        y = np.where(np.arange(n) < drift_at, y, label(rel_after))
        truth[drift_at:, :] = 0
        truth[drift_at:, list(rel_after)] = 1
        drift_points = [int(drift_at)]
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points,
                  feature_names=[f"x{j}" for j in range(d)])


def expanding_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max scaling using only samples seen up to and including t (no future leakage).
    The baseline's min_max_scale uses the whole stream, which leaks future information."""
    lo = np.minimum.accumulate(X, axis=0)
    hi = np.maximum.accumulate(X, axis=0)
    den = hi - lo
    den[den == 0.0] = 1.0
    return (X - lo) / den


def load_river_stream(name: str, n_max: int, scaling: str = "expanding") -> Stream:
    from river import datasets  # lazy import
    ds = getattr(datasets, name)()
    it = iter(ds)
    x0, y0 = next(it)
    names = list(x0.keys())
    rows, labels = [[x0[f] for f in names]], [y0]
    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        rows.append([x[f] for f in names]); labels.append(y)
    uniq = list(dict.fromkeys(labels))
    y_int = np.array([uniq.index(v) for v in labels], dtype=int)
    X = np.asarray(rows, dtype=float)
    if scaling == "expanding":
        mono = [names[j] for j in range(X.shape[1])
                if np.ptp(X[:, j]) > 0 and (np.all(np.diff(X[:, j]) >= 0) or np.all(np.diff(X[:, j]) <= 0))]
        if mono:
            warnings.warn(f"monotone features {mono} collapse to a constant under expanding min-max; "
                          f"use scaling='none' if the data is already in [0,1]")
        X = expanding_minmax(X)
    elif scaling == "none":      # data already in [0,1]: nothing to leak, nothing to distort
        if X.min() < -1e-9 or X.max() > 1 + 1e-9:
            raise ValueError("scaling='none' but features are outside [0,1]: "
                             f"min={X.min():.3g}, max={X.max():.3g}; use scaling='expanding'")
    else:
        raise ValueError(f"unknown scaling {scaling}")
    return Stream(X=X, y=y_int, truth=None, drift_points=[], feature_names=names)


def shuffle_control(st: Stream, seed: int) -> Stream:
    """Shuffled copy of a stream: destroys temporal structure, gives a noise floor."""
    p = np.random.default_rng(seed).permutation(len(st.y))
    return Stream(X=st.X[p], y=st.y[p], truth=None, drift_points=[], feature_names=st.feature_names)


# ---------------------------------------------------------------------------
# Agrawal-based drift stream with FUNCTIONAL ground truth.
# Re-implements the feature distributions and the 10 classification functions of
# river.datasets.synth.Agrawal (source read on 19 Nov 2026), vectorised, so that the
# label function and therefore the set of attributes it depends on is known exactly.
# Not seed-identical to river. tests/test_nonintrusive.py checks the labels against river.
# ---------------------------------------------------------------------------
AGRAWAL_FEATURES = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
# fixed, documented value ranges used for scaling (domain knowledge, no future information)
_AG_LO = np.array([20000, 0, 20, 0, 1, 0, 0, 1, 0.0])
# commission goes up to 85k in river's code (10000 + 75000*rand), although its docstring says 75k
_AG_HI = np.array([150000, 85000, 80, 4, 20, 8, 1.2e6, 30, 500000.0])
# attributes each classification function reads (from the source); indices into AGRAWAL_FEATURES
AGRAWAL_RELEVANT = {0: (2,), 1: (0, 2), 2: (2, 3), 3: (0, 2, 3), 4: (0, 2, 8), 5: (0, 1, 2),
                    6: (0, 1, 8), 7: (0, 1, 3), 8: (0, 1, 3, 8), 9: (0, 1, 3, 6, 7)}


def agrawal_raw_features(n: int, rng) -> np.ndarray:
    salary = 20000 + 130000 * rng.random(n)
    commission = np.where(salary >= 75000, 0.0, 10000 + 75000 * rng.random(n))
    age = rng.integers(20, 81, n).astype(float)
    elevel = rng.integers(0, 5, n).astype(float)
    car = rng.integers(1, 21, n).astype(float)
    zipc = rng.integers(0, 9, n).astype(float)
    hvalue = (8 - zipc) * 100000 * (0.5 + rng.random(n))
    hyears = rng.integers(1, 31, n).astype(float)
    loan = rng.random(n) * 500000
    return np.column_stack([salary, commission, age, elevel, car, zipc, hvalue, hyears, loan])


def agrawal_labels(f: int, X: np.ndarray) -> np.ndarray:
    s, c, a, e, _car, _z, hv, hy, l = X.T
    if f == 0:
        return ((a < 40) | (a >= 60)).astype(int)
    if f == 1:
        return np.where(a < 40, (50000 <= s) & (s <= 100000),
                        np.where(a < 60, (75000 <= s) & (s <= 125000), (25000 <= s) & (s <= 75000))).astype(int)
    if f == 2:
        return np.where(a < 40, (e == 0) | (e == 1),
                        np.where(a < 60, (e >= 1) & (e <= 3), (e >= 2) & (e <= 4))).astype(int)
    if f == 3:
        return np.where(a < 40, np.where((e == 0) | (e == 1), (25000 <= s) & (s <= 75000), (50000 <= s) & (s <= 100000)),
               np.where(a < 60, np.where((e >= 1) & (e <= 3), (50000 <= s) & (s <= 100000), (75000 <= s) & (s <= 125000)),
                        np.where((e >= 2) & (e <= 4), (50000 <= s) & (s <= 100000), (25000 <= s) & (s <= 75000)))).astype(int)
    if f == 4:
        return np.where(a < 40, np.where((50000 <= s) & (s <= 100000), (100000 <= l) & (l <= 300000), (200000 <= s) & (s <= 400000)),
               np.where(a < 60, np.where((75000 <= s) & (s <= 125000), (200000 <= s) & (l <= 400000), (300000 <= s) & (s <= 500000)),
                        np.where((25000 <= s) & (s <= 75000), (300000 <= l) & (l <= 500000), (75000 <= l) & (l <= 300000)))).astype(int)
    if f == 5:
        t = s + c
        return np.where(a < 40, (50000 <= t) & (t <= 100000),
                        np.where(a < 60, (75000 <= t) & (t <= 125000), (25000 <= t) & (t <= 75000))).astype(int)
    if f == 6:
        return np.where(2 * (s + c) / 3 - l / 5 - 20000 > 1, 0, 1)
    if f == 7:
        return np.where(2 * (s + c) / 3 - 5000 * e - 20000 > 1, 0, 1)
    if f == 8:
        return np.where(2 * (s + c) / 3 - 5000 * e - l / 5 - 10000 > 1, 0, 1)
    if f == 9:
        eq = np.where(hy >= 20, hv * (hy - 20) / 10, 0.0)
        return np.where(2 * (s + c) / 3 - 5000 * e + eq / 5 - 10000 > 1, 0, 1)
    raise ValueError("classification function must be 0..9")


def agrawal_drift_stream(n=20000, drift_at=10000, f_before=6, f_after=2, seed=0) -> Stream:
    """Abrupt switch between two Agrawal classification functions. truth = attributes the function reads.
    drift_at=None -> stationary control with f_before only."""
    rng = np.random.default_rng(seed)
    Xr = agrawal_raw_features(n, rng)
    y = agrawal_labels(f_before, Xr)
    truth = np.zeros((n, 9), dtype=int)
    truth[:, list(AGRAWAL_RELEVANT[f_before])] = 1
    drift_points: List[int] = []
    if drift_at is not None and drift_at < n:
        y = np.where(np.arange(n) < drift_at, y, agrawal_labels(f_after, Xr))
        truth[drift_at:, :] = 0
        truth[drift_at:, list(AGRAWAL_RELEVANT[f_after])] = 1
        drift_points = [int(drift_at)]
    X = (Xr - _AG_LO) / (_AG_HI - _AG_LO)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points, feature_names=AGRAWAL_FEATURES)


def build_stream(scfg: dict, seed: int) -> Stream:
    if scfg["type"] == "relevance_drift":
        return relevance_drift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "agrawal_drift":
        return agrawal_drift_stream(n=scfg["n"], drift_at=scfg.get("drift_at"),
                                    f_before=scfg["f_before"], f_after=scfg["f_after"], seed=seed)
    if scfg["type"] == "river":
        st = load_river_stream(scfg["name"], scfg["n"], scfg.get("scaling", "expanding"))
        return shuffle_control(st, seed) if scfg.get("shuffle") else st
    raise ValueError(f"unknown stream type {scfg['type']}")
