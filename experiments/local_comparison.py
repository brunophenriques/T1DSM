import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incremental_local_weights import (
    IncrementalLocalUnimNeuron
)


from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from experiments.dynamic_weights import DynamicUnimNeuron
from experiments.local_weights import LocalUnimNeuron
from experiments.local_stream import local_drift_stream
from relevance.metrics import topk_precision, first_sustained

with open(ROOT / "configs" / "mve.json") as f:
    config = json.load(f)

SEEDS = list(range(10))
DRIFT_AT = 10000
CHECKPOINT = 50
WASHOUT = 1000

VARIANTS = [
    ("Baseline", EvolvingNeuroFuzzyUnimSafe),
    ("Dynamic-Global", DynamicUnimNeuron),
    ("Dynamic-Local", LocalUnimNeuron),
    ("Incremental-Local", IncrementalLocalUnimNeuron)
]
results = []

for name, model_cls in VARIANTS:

    for seed in SEEDS:

        stream, regions = local_drift_stream(seed=seed)

        params = copy.deepcopy(config["model"])
        params["buffer_size_separability"] = 100

        model = model_cls(
            n_features=10,
            n_classes=2,
            **params
        )

        n = len(stream.y)

        correct = np.zeros(n)
        precision = np.full(n, np.nan)
        region_match = np.full(n, np.nan)

        for t in range(n):

            x = stream.X[t]
            y = stream.y[t]

            # Predict before learning
            prediction = model.predict(x.reshape(1, -1))[0]
            correct[t] = prediction == y

            if model.rules:

                activations, _ = (
                    model._compute_rule_activations_and_dists(x)
                )

                # Evaluate the most activated rule
                rule = model.rules[np.argmax(activations)]

                precision[t] = topk_precision(
                    rule.w,
                    stream.truth[t],
                    k=3
                )

                region_match[t] = (
                    int(rule.center[0] >= 0.5) == regions[t]
                )

            # Learn only after prediction and evaluation
            model.partial_fit(
                x.reshape(1, -1),
                np.array([y])
            )

        # Evaluate each region independently
        for region in [0, 1]:

            before = (
                (regions == region)
                & (np.arange(n) >= WASHOUT)
                & (np.arange(n) < DRIFT_AT)
            )

            after = (
                (regions == region)
                & (np.arange(n) >= DRIFT_AT + WASHOUT)
            )

            trajectory = []

            for start in range(DRIFT_AT, n, CHECKPOINT):

                end = min(start + CHECKPOINT, n)

                mask = regions[start:end] == region

                values = precision[start:end][mask]

                trajectory.append(
                    np.nanmean(values)
                    if len(values) else np.nan
                )

            recovery_index = first_sustained(
                np.array(trajectory) >= 0.9,
                consecutive=3
            )

            recovery = (
                np.nan
                if recovery_index is None
                else (recovery_index + 1) * CHECKPOINT
            )

            row = {
                "variant": name,
                "seed": seed,
                "region": region,
                "precision_before": np.nanmean(precision[before]),
                "precision_after": np.nanmean(precision[after]),
                "recovery": recovery,
                "accuracy_after": np.mean(correct[after]),
                "region_match": np.nanmean(region_match[after]),
                "accuracy_total": np.mean(correct)
            }

            results.append(row)

            print(
                f"{name:15s} | Seed={seed} | "
                f"Region={region} | "
                f"Precision={row['precision_after']:.4f} | "
                f"Recovery={recovery} | "
                f"Accuracy={row['accuracy_after']:.4f}"
            )

output = ROOT / "results_q2" / "incremental_comparison"
output.mkdir(parents=True, exist_ok=True)

with open(output / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# Aggregate results
summary = []

for name, _ in VARIANTS:

    for region in [0, 1]:

        rows = [
            r for r in results
            if r["variant"] == name and r["region"] == region
        ]

        recovered = [
            r["recovery"] for r in rows
            if np.isfinite(r["recovery"])
        ]

        summary.append({
            "variant": name,
            "region": region,
            "precision_before": np.mean(
                [r["precision_before"] for r in rows]
            ),
            "precision_after": np.mean(
                [r["precision_after"] for r in rows]
            ),
            "recovery_mean": (
                np.mean(recovered) if recovered else np.nan
            ),
            "recovered_runs": len(recovered),
            "accuracy_after": np.mean(
                [r["accuracy_after"] for r in rows]
            ),
            "region_match": np.mean(
                [r["region_match"] for r in rows]
            )
        })

with open(output / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"\nResults saved to {output}")