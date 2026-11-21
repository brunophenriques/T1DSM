import csv
import json
from pathlib import Path

import numpy as np
from river.drift import ADWIN

from run_relevance import run_one


ROOT = Path(__file__).resolve().parents[1]


def detect_drift(errors, delta=0.002):
    """Detect changes in the prequential error stream."""
    detector = ADWIN(delta=delta)
    drift_points = []

    for t, error in enumerate(errors, start=1):
        detector.update(float(error))

        if detector.drift_detected:
            drift_points.append(t)

    return drift_points


def main():
    config_path = ROOT / "configs/protocol/elec2.json"

    with open(config_path) as f:
        cfg = json.load(f)

    result = run_one(
        cfg,
        seed=0,
        log=True,
        model_cfg=cfg["variants"]["ref"],
    )

    y_true = result["y"]
    y_pred = result["y_pred"]

    errors = (y_true != y_pred).astype(int)

    drift_points = detect_drift(errors, delta=0.002)

    print("Stream: Elec2")
    print("Samples:", len(errors))
    print("Accuracy:", np.mean(1 - errors))
    print("Number of detected changes:", len(drift_points))
    print("Detected drift points:", drift_points)

    outdir = ROOT / "results_wf2/adwin"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "drift_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"])

        for t in drift_points:
            writer.writerow([t])

    with open(outdir / "summary.json", "w") as f:
        json.dump(
            {
                "stream": "Elec2",
                "variant": "ref",
                "seed": 0,
                "delta": 0.002,
                "n_samples": len(errors),
                "accuracy": float(np.mean(1 - errors)),
                "n_detected_changes": len(drift_points),
                "drift_points": drift_points,
                "interpretation": "descriptive only",
            },
            f,
            indent=2,
        )

    print("Results saved to:", outdir)


if __name__ == "__main__":
    main()