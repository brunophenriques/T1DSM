import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, rankdata, binomtest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_wf2"

STREAMS = [
    "s1_gauss_abrupt",
    "s2_gauss_gradual",
]

SEEDS = range(10)


def load_metric(stream, variant, metric):
    values = []

    for seed in SEEDS:
        path = (
            RESULTS / stream / variant
            / f"seed_{seed}" / "metrics.json"
        )

        with open(path) as f:
            data = json.load(f)

        values.append(data[metric])

    return np.asarray(values, dtype=float)


def bootstrap_ci(values, n_boot=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(42)

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return tuple(np.percentile(means, [2.5, 97.5]))


def paired_comparison(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    differences = (a[mask] - b[mask]).round(10)

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return {
            "n_pairs": len(differences),
            "wilcoxon_p": 1.0,
            "rank_biserial": 0.0,
            "sign_test_p": 1.0,
        }

    ranks = rankdata(np.abs(nonzero))

    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()

    effect = (positive - negative) / (
        positive + negative
    )

    return {
        "n_pairs": len(differences),
        "wilcoxon_p": float(
            wilcoxon(differences).pvalue
        ),
        "rank_biserial": float(effect),
        "sign_test_p": float(
            binomtest(
                int(np.sum(nonzero > 0)),
                len(nonzero),
                p=0.5,
            ).pvalue
        ),
    }


def holm_correction(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    n = len(pvalues)

    adjusted = np.maximum.accumulate(
        (n - np.arange(n)) * pvalues[order]
    )

    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1.0)

    return result


def main():
    rows = []
    comparisons = []

    metrics = [
        "recovery_censored/G/drift1",
        "recovery_censored/A/drift1",
        "acc/recovery_censored/drift1",
    ]

    for stream in STREAMS:
        for variant in ["ref", "open"]:
            for metric in metrics:

                values = load_metric(
                    stream, variant, metric
                )

                low, high = bootstrap_ci(values)

                rows.append({
                    "stream": stream,
                    "variant": variant,
                    "metric": metric,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "ci_low": low,
                    "ci_high": high,
                })

    # Compare R1 against R4 using censored times.
    for stream in STREAMS:
        for variant in ["ref", "open"]:

            global_recovery = load_metric(
                stream,
                variant,
                "recovery_censored/G/drift1",
            )

            rule_recovery = load_metric(
                stream,
                variant,
                "recovery_censored/A/drift1",
            )

            result = paired_comparison(
                global_recovery,
                rule_recovery,
            )

            comparisons.append({
                "comparison": f"{stream}/{variant}: R1 vs R4",
                **result,
            })

    # Compare ranking and predictive recovery.
    # Only open is used for predictive claims.
    for stream in STREAMS:

        ranking = load_metric(
            stream,
            "open",
            "recovery_censored/G/drift1",
        )

        accuracy = load_metric(
            stream,
            "open",
            "acc/recovery_censored/drift1",
        )

        comparisons.append({
            "comparison": f"{stream}/open: R1 vs accuracy",
            **paired_comparison(ranking, accuracy),
        })

    # Compare abrupt and gradual ranking recovery.
    s1 = load_metric(
        STREAMS[0], "ref", metrics[0]
    )

    s2 = load_metric(
        STREAMS[1], "ref", metrics[0]
    )

    comparisons.append({
        "comparison": "S1 vs S2: R1 post-drift recovery",
        **paired_comparison(s1, s2),
    })

    adjusted = holm_correction([
        row["wilcoxon_p"] for row in comparisons
    ])

    for row, p in zip(comparisons, adjusted):
        row["holm_p"] = float(p)

    outdir = RESULTS / "statistics"
    outdir.mkdir(parents=True, exist_ok=True)

    for filename, data in [
        ("descriptive.csv", rows),
        ("comparisons.csv", comparisons),
    ]:
        with open(
            outdir / filename, "w", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=data[0].keys()
            )
            writer.writeheader()
            writer.writerows(data)

    for row in comparisons:
        print(
            f"{row['comparison']} | "
            f"p={row['wilcoxon_p']:.6f} | "
            f"Holm={row['holm_p']:.6f} | "
            f"effect={row['rank_biserial']:.4f}"
        )

    print("Results saved to:", outdir)


if __name__ == "__main__":
    main()