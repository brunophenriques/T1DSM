"""Observe the prequential run without changing the model."""

import csv
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np

from ENF_UnimNeuron_Safe import FeatureSeparabilityBuffer
from relevance.extract import RuleTracker, take_snapshot


WINDOW_SIZES = (50, 100, 200, 500, 1000)
FORGETTING = (0.99, 0.995, 1.0)
LOCAL_SIZE = 200
LOCAL_MIN = 20
WARMUP = 500


class RecursiveSeparability:
    """Exponentially weighted class mass, mean and within-class sum of squares."""

    def __init__(self, n_classes, n_features):
        self.decay = np.array(FORGETTING)[:, None]
        self.mass = np.zeros((len(FORGETTING), n_classes))
        self.mean = np.zeros((len(FORGETTING), n_classes, n_features))
        self.scatter = np.zeros_like(self.mean)
        self.counts = np.zeros(n_classes, dtype=int)

    def add(self, x, y):
        # R2: age the evidence for every class, even those absent from this step.
        # Uniform decay leaves a class mean unchanged until the class receives a sample.
        self.mass *= self.decay
        self.scatter *= self.decay[:, :, None]
        self.mass[:, y] += 1
        self.counts[y] += 1
        delta = x - self.mean[:, y]
        self.mean[:, y] += delta / self.mass[:, y, None]
        self.scatter[:, y] += delta * (x - self.mean[:, y])

    def weights(self):
        n_features = self.mean.shape[-1]
        weights = np.full((len(FORGETTING), n_features), 1.0 / n_features)
        if self.counts.sum() < 10 or np.count_nonzero(self.counts) < 2:
            return weights

        total_mass = self.mass.sum(axis=1, keepdims=True)
        overall_mean = (self.mass[:, :, None] * self.mean).sum(axis=1) / total_mass
        keep = self.counts >= 2  # Match the batch statistic's singleton handling.
        dev = self.mean[:, keep] - overall_mean[:, None]
        # Features score higher when class centres are far apart and samples within each class stay close.
        between = self.mass[:, keep, None] * dev**2
        sep = (between / (np.maximum(self.scatter[:, keep], 0.0) + 1e-6)).sum(axis=1)
        maximum = sep.max(axis=1)
        valid = maximum > 0
        weights[valid] = 0.2 + 0.8 * sep[valid] / maximum[valid, None]
        return weights


class LocalWindow:
    def __init__(self, n_features):
        self.buffer = FeatureSeparabilityBuffer(n_features, LOCAL_SIZE)
        self.times = deque(maxlen=LOCAL_SIZE)
        self.weights = np.full(n_features, np.nan)
        self.dirty = False

    def add(self, x, y, t):
        self.buffer.add(x, y)
        self.times.append(t)
        self.dirty = True

    def snapshot(self):
        # R3 needs both labels in the local window to estimate class separation.
        if self.dirty:
            if len(self.times) >= LOCAL_MIN and len(set(self.buffer.y)) > 1:
                self.weights = self.buffer.compute_feature_weights()
            else:
                self.weights = np.full(self.buffer.n_features, np.nan)
            self.dirty = False
        return len(self.times), self.times[0], self.times[-1], self.weights


class RelevanceCollector:
    """Collect metrics indexed by one-based sample count and stable rule ID."""

    def __init__(self, model, feature_names, tracker=None):
        self.model = model
        self.d = model.n_features
        self.feature_names = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(self.d)
        ]
        self.tracker = tracker if tracker is not None else RuleTracker()
        self.owns_tracker = tracker is None
        self.local = {}
        self.windows = {size: FeatureSeparabilityBuffer(self.d, size) for size in WINDOW_SIZES}
        self.recursive = RecursiveSeparability(model.n_classes, self.d)
        self.checkpoints = []
        self.rule_rows = []
        self.elapsed = 0.0

    def observe_prediction(self, x, y, t):
        started = perf_counter()
        if self.model.rules:
            activations, _ = self.model._compute_rule_activations_and_dists(x)
            # The strongest activation assigns the sample to an R3 window.
            # Predictions can still draw on several rules.
            winner = self.model.rules[int(np.argmax(activations))]
            uid = self.tracker.uid(winner)
            if uid not in self.local:
                self.local[uid] = LocalWindow(self.d)
            self.local[uid].add(x, y, t + 1)
        for buffer in self.windows.values():
            buffer.add(x, y)
        self.recursive.add(x, y)
        self.elapsed += perf_counter() - started

    def after_fit(self, t, checkpoint):
        started = perf_counter()
        if self.owns_tracker:
            self.tracker.observe(self.model, t)
        if self.local:
            # Retire deleted windows. A merge keeps the surviving rule's ID and window.
            active = {self.tracker.uid(rule) for rule in self.model.rules}
            for uid in self.local.keys() - active:
                del self.local[uid]
        if checkpoint:
            self._checkpoint(t + 1)
        self.elapsed += perf_counter() - started

    def _checkpoint(self, t):
        snap = take_snapshot(self.model, self.tracker)
        valid = 0
        total_support = 0
        weighted = np.zeros(self.d)
        for uid in snap["uids"]:
            if uid in self.local:
                size, first, last, weights = self.local[uid].snapshot()
            else:
                size, first, last, weights = 0, None, None, np.full(self.d, np.nan)
            self.rule_rows.append([t, uid, size, first, last, *weights])
            if np.isfinite(weights).all():
                valid += 1
                total_support += size
                weighted += size * weights

        n_rules = snap["n_rules"]
        coverage = 100 * valid / n_rules if n_rules else np.nan
        # H6 needs one stream-level ranking. Larger valid local windows contribute more.
        aggregate = weighted / total_support if total_support else np.full(self.d, np.nan)
        window_weights = [window.compute_feature_weights() for window in self.windows.values()]
        recursive_weights = self.recursive.weights()
        self.checkpoints.append([
            t, n_rules, valid, coverage,
            *snap["G"], *snap["A"], *aggregate,
            *recursive_weights.ravel(), *np.concatenate(window_weights),
        ])

    def save(self, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        methods = ["R1", "R4", "R3_mean"]
        methods.extend(f"R2_{decay}" for decay in FORGETTING)
        methods.extend(f"window_{size}" for size in WINDOW_SIZES)
        with (outdir / "wf3_checkpoints.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "t", "n_rules", "r3_valid_rules", "r3_coverage_pct",
                *(f"{method}_{name}" for method in methods for name in self.feature_names),
            ])
            writer.writerows(self.checkpoints)
        with (outdir / "wf3_local.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "t", "uid", "window_n", "first_sample", "last_sample",
                *(f"R3_{name}" for name in self.feature_names),
            ])
            writer.writerows(self.rule_rows)
