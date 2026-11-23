"""Streams with known feature relevance (synthetic) and a leakage-free loader (real).

River's Agrawal/SEA drift changes the decision rule, not necessarily WHICH features
matter, so we use our own generator where the relevant set is explicit.

Stream families (see PROTOCOL.md, section "Streams"):
  relevance_drift : uniform features, y = 1[sum of relevant > |R|/2]  (Minês's original stream)
  gauss_shift     : Gaussian features whose class means differ only on the relevant set
                    (a concept the model can actually learn; see the learnability pilot)
  sea             : SEA concepts; the threshold drifts but the relevant set {x0, x1} never changes
                    (negative control: accuracy can drop, the ranking should not move)
  agrawal         : river's Agrawal generator (S5); a drift switches the labelling function, and the
                    relevant set is the features that function reads in the generator code
  "copy" option   : appends a noisy, never-relevant copy of one feature (S6)
All synthetic streams use fixed, known bounds, so no data-dependent scaling is needed.
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
    drift_widths: List[int] = field(default_factory=list)   # 0 = abrupt; >0 = gradual transition width
    meta: dict = field(default_factory=dict)                  # generator parameters, for meta.json


def _new_concept_mask(n: int, drift_at, width: int, rng) -> np.ndarray:
    """True where the post-drift concept generates the sample.

    Abrupt (width 0): t >= drift_at, and NO random numbers are drawn, so abrupt streams are
    bit-identical to the generators used before gradual drift existed.
    Gradual: P(new concept at t) = 1 / (1 + exp(-4 (t - drift_at) / width)), the same sigmoid as
    river's ConceptDriftStream; the transition is taken as [drift_at - width/2, drift_at + width/2]."""
    t = np.arange(n)
    if drift_at is None or drift_at >= n:
        return np.zeros(n, dtype=bool)
    if not width:
        return t >= drift_at
    p = 1.0 / (1.0 + np.exp(-4.0 * (t - drift_at) / float(width)))
    return rng.random(n) < p


def _truth(n, d, drift_at, rel_before, rel_after) -> np.ndarray:
    truth = np.zeros((n, d), dtype=int)
    truth[:, list(rel_before)] = 1
    if drift_at is not None and drift_at < n:
        truth[drift_at:, :] = 0
        truth[drift_at:, list(rel_after)] = 1
    return truth


def relevance_drift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2),
                           rel_after=(5, 6, 7), label_noise=0.0, seed=0, drift_width=0) -> Stream:
    """y = 1[ sum of the relevant features > |R|/2 ]. At `drift_at` the relevant set R switches
    from rel_before to rel_after (abruptly, or gradually if drift_width > 0).
    drift_at=None gives a stationary control stream."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))

    def label(rel):
        rel = list(rel)
        return (X[:, rel].sum(axis=1) > len(rel) / 2.0).astype(int)

    y = label(rel_before)
    truth = _truth(n, d, drift_at, rel_before, rel_after)
    drift_points: List[int] = []
    widths: List[int] = []
    if drift_at is not None and drift_at < n:
        new = _new_concept_mask(n, drift_at, drift_width, rng)
        y = np.where(new, label(rel_after), y)
        drift_points, widths = [int(drift_at)], [int(drift_width)]
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points,
                  feature_names=[f"x{j}" for j in range(d)], drift_widths=widths,
                  meta=dict(family="relevance_drift", n=n, d=d, drift_at=drift_at, drift_width=drift_width,
                            rel_before=list(rel_before), rel_after=list(rel_after),
                            label_noise=label_noise, seed=seed))


def gauss_shift_stream(n=20000, d=10, drift_at=10000, rel_before=(0, 1, 2), rel_after=(5, 6, 7),
                       delta=0.3, sd=0.12, drift_width=0, label_noise=0.0, seed=0) -> Stream:
    """Balanced binary classes. Every feature is N(0.5, sd); on the features that are relevant for the
    active concept the class means are 0.5 - delta/2 (class 0) and 0.5 + delta/2 (class 1).
    Values are clipped to [0, 1] (fixed bounds, no data-dependent scaling).
    The relevant set switches from rel_before to rel_after at drift_at."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0.5, sd, (n, d))
    new = _new_concept_mask(n, drift_at, drift_width, rng)
    shift = np.where(y == 1, delta / 2.0, -delta / 2.0)
    for mask, rel in ((~new, rel_before), (new, rel_after)):
        idx = np.where(mask)[0]
        if idx.size:
            X[np.ix_(idx, list(rel))] += shift[idx, None]
    X = np.clip(X, 0.0, 1.0)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    has_drift = drift_at is not None and drift_at < n
    return Stream(X=X, y=y.astype(int), truth=_truth(n, d, drift_at, rel_before, rel_after),
                  drift_points=[int(drift_at)] if has_drift else [],
                  feature_names=[f"x{j}" for j in range(d)],
                  drift_widths=[int(drift_width)] if has_drift else [],
                  meta=dict(family="gauss_shift", n=n, d=d, drift_at=drift_at, drift_width=drift_width,
                            rel_before=list(rel_before), rel_after=list(rel_after), delta=delta, sd=sd,
                            label_noise=label_noise, seed=seed))


SEA_THRESHOLDS = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}   # same variants as river.datasets.synth.SEA


def sea_stream(n=20000, drift_at=10000, variant_before=0, variant_after=2, drift_width=0,
               label_noise=0.0, seed=0) -> Stream:
    """SEA concepts: 3 features U[0, 10], y = 1[x0 + x1 > threshold]; x2 is never relevant.
    The drift changes the threshold, NOT the relevant set, so the ground truth is {x0, x1} throughout.
    Features are divided by 10 (fixed bounds) so they lie in [0, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, (n, 3))
    new = _new_concept_mask(n, drift_at, drift_width, rng)
    thr = np.where(new, SEA_THRESHOLDS[variant_after], SEA_THRESHOLDS[variant_before])
    y = (X[:, 0] + X[:, 1] > thr).astype(int)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    truth = np.zeros((n, 3), dtype=int)
    truth[:, [0, 1]] = 1
    has_drift = drift_at is not None and drift_at < n
    return Stream(X=X / 10.0, y=y, truth=truth,
                  drift_points=[int(drift_at)] if has_drift else [],
                  feature_names=["x0", "x1", "x2"],
                  drift_widths=[int(drift_width)] if has_drift else [],
                  meta=dict(family="sea", n=n, drift_at=drift_at, drift_width=drift_width,
                            variant_before=variant_before, variant_after=variant_after,
                            label_noise=label_noise, seed=seed))


AGRAWAL_FEATURES = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
# Fixed bounds read from river's generator (agrawal_source.txt = river 0.27.0). Note: commission is 0 or
# U[10k, 85k], and hvalue = (8 - zipcode) * 100k * U[0.5, 1.5], so it lies in [0, 1.2M].
AGRAWAL_BOUNDS = {"salary": (20000.0, 150000.0), "commission": (0.0, 85000.0), "age": (20.0, 80.0),
                  "elevel": (0.0, 4.0), "car": (1.0, 20.0), "zipcode": (0.0, 8.0), "hvalue": (0.0, 1200000.0),
                  "hyears": (1.0, 30.0), "loan": (0.0, 500000.0)}
# The features each classification function reads, from the generator code (PROTOCOL.md §6: "relevant
# sets from the generator code"). Function 4 is reproduced as river implements it (it reads salary and loan).
AGRAWAL_RELEVANT = {0: ("age",), 1: ("age", "salary"), 2: ("age", "elevel"), 3: ("age", "elevel", "salary"),
                    4: ("age", "salary", "loan"), 5: ("age", "salary", "commission"),
                    6: ("salary", "commission", "loan"), 7: ("salary", "commission", "elevel"),
                    8: ("salary", "commission", "elevel", "loan"),
                    9: ("salary", "commission", "elevel", "hvalue", "hyears")}


def agrawal_relevant_idx(function: int) -> tuple:
    return tuple(AGRAWAL_FEATURES.index(f) for f in AGRAWAL_RELEVANT[function])


def agrawal_stream(n=20000, function_before=2, function_after=None, drift_at=None, drift_width=0, seed=0) -> Stream:
    """river's Agrawal generator (balance_classes=False, perturbation=0). The features come from
    river.datasets.synth.Agrawal(seed=seed); with these settings the random draws do not depend on the
    classification function, so a drift only changes the labelling function: before `drift_at` the
    labels come from `function_before`, after it from `function_after`.
    Features are scaled with the generator's fixed bounds (AGRAWAL_BOUNDS), so no data-dependent scaling.
    Ground truth = the features the active function reads (AGRAWAL_RELEVANT)."""
    from river.datasets.synth import Agrawal   # lazy import: river is only needed for this stream
    has_drift = function_after is not None and drift_at is not None and drift_at < n
    rows = []
    for i, (x, _) in enumerate(Agrawal(classification_function=function_before, seed=seed)):
        if i >= n:
            break
        rows.append([x[f] for f in AGRAWAL_FEATURES])
    raw = np.asarray(rows, dtype=float)

    def label(fn):
        f = getattr(Agrawal, f"_classification_function_{fn}")
        return np.array([f(*r) for r in raw], dtype=int)

    y = label(function_before)
    rel_before, rel_after = agrawal_relevant_idx(function_before), ()
    if has_drift:
        new = _new_concept_mask(n, drift_at, drift_width, np.random.default_rng(seed))
        y = np.where(new, label(function_after), y)
        rel_after = agrawal_relevant_idx(function_after)
    lo = np.array([AGRAWAL_BOUNDS[f][0] for f in AGRAWAL_FEATURES])
    hi = np.array([AGRAWAL_BOUNDS[f][1] for f in AGRAWAL_FEATURES])
    X = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    d = len(AGRAWAL_FEATURES)
    truth = _truth(n, d, drift_at if has_drift else None, rel_before, rel_after)
    return Stream(X=X, y=y, truth=truth,
                  drift_points=[int(drift_at)] if has_drift else [],
                  feature_names=list(AGRAWAL_FEATURES),
                  drift_widths=[int(drift_width)] if has_drift else [],
                  meta=dict(family="agrawal", n=n, function_before=function_before,
                            function_after=function_after if has_drift else None,
                            drift_at=drift_at if has_drift else None, drift_width=drift_width,
                            rel_before=[AGRAWAL_FEATURES[j] for j in rel_before],
                            rel_after=[AGRAWAL_FEATURES[j] for j in rel_after],
                            class1_share=float(y.mean()), seed=seed, river_generator="Agrawal",
                            balance_classes=False, perturbation=0.0))


def add_noisy_copy(st: Stream, source: int, noise_sd: float, seed: int) -> Stream:
    """Append x_d = clip(x_source + N(0, noise_sd), 0, 1): a non-causal copy that is never relevant
    (S6). The noise uses its own random generator, so the original columns and labels are bit-identical
    to the stream without the copy."""
    rng = np.random.default_rng([seed, 6])
    n, d = st.X.shape
    copy = np.clip(st.X[:, source] + rng.normal(0.0, noise_sd, n), 0.0, 1.0)
    truth = None if st.truth is None else np.column_stack([st.truth, np.zeros(n, dtype=st.truth.dtype)])
    names = (st.feature_names or [f"x{j}" for j in range(d)]) + [f"x{d}"]
    return Stream(X=np.column_stack([st.X, copy]), y=st.y, truth=truth, drift_points=list(st.drift_points),
                  feature_names=names, drift_widths=list(st.drift_widths),
                  meta=dict(st.meta, d=d + 1, copy=dict(column=d, source=source, noise_sd=noise_sd)))


def expanding_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max scaling using only samples seen up to and including t (no future leakage).
    The baseline's min_max_scale uses the whole stream, which leaks future information."""
    lo = np.minimum.accumulate(X, axis=0)
    hi = np.maximum.accumulate(X, axis=0)
    den = hi - lo
    den[den == 0.0] = 1.0
    return (X - lo) / den


def stream_from_river(ds, n_max: int, drop=None, name: str = "river",
                      scaling: str = "expanding", bounds: dict | None = None) -> Stream:
    """Vectorise a River dataset using causal or fixed-domain scaling."""
    drop = set(drop or [])
    it = iter(ds)
    x0, y0 = next(it)
    names = [f for f in x0.keys() if f not in drop]
    rows, labels = [[x0[f] for f in names]], [y0]
    for i, (x, y) in enumerate(it, start=1):
        if i >= n_max:
            break
        rows.append([x[f] for f in names]); labels.append(y)
    uniq = list(dict.fromkeys(labels))
    y_int = np.array([uniq.index(v) for v in labels], dtype=int)
    X = np.asarray(rows, dtype=float)
    if scaling == "domain":
        # fixed, documented per-feature bounds (default [0,1]); order-independent, so the shuffled control sees the same scaling
        lo = np.array([(bounds or {}).get(n, [0.0, 1.0])[0] for n in names], dtype=float)
        hi = np.array([(bounds or {}).get(n, [0.0, 1.0])[1] for n in names], dtype=float)
        if not np.all(np.isfinite(lo) & np.isfinite(hi) & (hi > lo)):
            raise ValueError("feature bounds must be finite with upper > lower")
        X = (X - lo) / (hi - lo)
        bad = [names[j] for j in range(X.shape[1]) if X[:, j].min() < -1e-9 or X[:, j].max() > 1 + 1e-9]
        if bad:
            raise ValueError(f"features outside their declared bounds: {bad}; add them to 'bounds'")
    elif scaling == "expanding":
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
    meta = dict(family="river", name=name, n=n_max, dropped=sorted(drop),
                scaling="expanding_minmax" if scaling == "expanding" else scaling)
    if scaling == "domain":
        meta["bounds"] = dict(bounds or {})
    return Stream(X=X, y=y_int, truth=None, drift_points=[], feature_names=names, meta=meta)


def load_river_stream(name: str, n_max: int, scaling="expanding", bounds: dict | None = None,
                      drop=None) -> Stream:
    """Load River data; also accept the legacy third positional drop argument."""
    if not isinstance(scaling, str):
        if drop is not None:
            raise TypeError("drop was supplied twice")
        drop, scaling = scaling, "expanding"
    from river import datasets  # lazy import
    return stream_from_river(getattr(datasets, name)(), n_max, drop=drop, name=name,
                             scaling=scaling, bounds=bounds)


def shuffle_control(st: Stream, seed: int) -> Stream:
    """Shuffled copy of a stream: destroys temporal structure, gives a noise floor."""
    p = np.random.default_rng(seed).permutation(len(st.y))
    return Stream(X=st.X[p], y=st.y[p], truth=None, drift_points=[], feature_names=st.feature_names,
                  meta=dict(st.meta, shuffled=True, shuffle_seed=seed))


# ---------------------------------------------------------------------------
# Agrawal-based drift stream with FUNCTIONAL ground truth.
# Re-implements the feature distributions and the 10 classification functions of
# river.datasets.synth.Agrawal (source read on 19 Nov 2026), vectorised, so that the
# label function and therefore the set of attributes it depends on is known exactly.
# Not seed-identical to river. tests/test_nonintrusive.py checks the labels against river.
# ---------------------------------------------------------------------------
# fixed, documented value ranges used for scaling (domain knowledge, no future information)
_AG_LO = np.array([20000, 0, 20, 0, 1, 0, 0, 1, 0.0])
# commission goes up to 85k in river's code (10000 + 75000*rand), although its docstring says 75k
_AG_HI = np.array([150000, 85000, 80, 4, 20, 8, 1.2e6, 30, 500000.0])
# attributes each classification function reads (from the source); indices into AGRAWAL_FEATURES
_AGRAWAL_DRIFT_RELEVANT = {0: (2,), 1: (0, 2), 2: (2, 3), 3: (0, 2, 3), 4: (0, 2, 8), 5: (0, 1, 2),
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
    truth[:, list(_AGRAWAL_DRIFT_RELEVANT[f_before])] = 1
    drift_points: List[int] = []
    if drift_at is not None and drift_at < n:
        y = np.where(np.arange(n) < drift_at, y, agrawal_labels(f_after, Xr))
        truth[drift_at:, :] = 0
        truth[drift_at:, list(_AGRAWAL_DRIFT_RELEVANT[f_after])] = 1
        drift_points = [int(drift_at)]
    X = (Xr - _AG_LO) / (_AG_HI - _AG_LO)
    return Stream(X=X, y=y.astype(int), truth=truth, drift_points=drift_points, feature_names=AGRAWAL_FEATURES)


def build_stream(scfg: dict, seed: int) -> Stream:
    """Build the stream of a config's "stream" block. An optional "copy": {"source": j, "noise_sd": s}
    appends a noisy, never-relevant copy of feature j (S6)."""
    st = _build_base(scfg, seed)
    if scfg.get("copy"):
        st = add_noisy_copy(st, int(scfg["copy"]["source"]), float(scfg["copy"]["noise_sd"]), seed)
    return st


def _build_base(scfg: dict, seed: int) -> Stream:
    if scfg["type"] == "agrawal":
        return agrawal_stream(
            n=scfg["n"], function_before=scfg["function_before"], function_after=scfg.get("function_after"),
            drift_at=scfg.get("drift_at"), drift_width=scfg.get("drift_width", 0), seed=seed)
    if scfg["type"] == "relevance_drift":
        return relevance_drift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            label_noise=scfg.get("label_noise", 0.0), seed=seed, drift_width=scfg.get("drift_width", 0))
    if scfg["type"] == "gauss_shift":
        return gauss_shift_stream(
            n=scfg["n"], d=scfg["d"], drift_at=scfg.get("drift_at"),
            rel_before=tuple(scfg["rel_before"]), rel_after=tuple(scfg["rel_after"]),
            delta=scfg["delta"], sd=scfg.get("sd", 0.12), drift_width=scfg.get("drift_width", 0),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "sea":
        return sea_stream(
            n=scfg["n"], drift_at=scfg.get("drift_at"), variant_before=scfg.get("variant_before", 0),
            variant_after=scfg.get("variant_after", 2), drift_width=scfg.get("drift_width", 0),
            label_noise=scfg.get("label_noise", 0.0), seed=seed)
    if scfg["type"] == "agrawal_drift":
        return agrawal_drift_stream(n=scfg["n"], drift_at=scfg.get("drift_at"),
                                    f_before=scfg["f_before"], f_after=scfg["f_after"], seed=seed)
    if scfg["type"] == "river":
        st = load_river_stream(scfg["name"], scfg["n"], scaling=scfg.get("scaling", "expanding"),
                               bounds=scfg.get("bounds"), drop=scfg.get("drop"))
        return shuffle_control(st, seed) if scfg.get("shuffle") else st
    raise ValueError(f"unknown stream type {scfg['type']}")
