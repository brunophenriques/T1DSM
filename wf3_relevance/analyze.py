"""Generate Workflow 3 tables and the window-size curve from saved runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from relevance.metrics import is_constant, topk_jaccard, topk_precision
from wf2_drift_validity.recovery_metrics import ranking_recovery
from wf3_relevance.collect import FORGETTING, WARMUP, WINDOW_SIZES


LAG = 200
K = 3


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector(row, prefix):
    return np.array([float(value) for name, value in row.items() if name.startswith(prefix + "_")])


def mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def stability(vectors):
    scores = []
    for t, current in vectors.items():
        previous = vectors.get(t - LAG)
        if previous is not None and not is_constant(current) and not is_constant(previous):
            scores.append(topk_jaccard(previous, current, K))
    return mean(scores), len(scores)


def local_precision_at_endpoint(first, last, weights, drift, before, after):
    if drift is None or last <= drift:
        return topk_precision(weights, before, K)
    if first > drift:
        return topk_precision(weights, after, K)
    return np.nan


def summarize_run(path):
    folder = path.parent
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    checkpoints = {int(row["t"]): row for row in read_csv(path) if int(row["t"]) > WARMUP}
    local = read_csv(folder / "wf3_local.csv")
    keys = dict(stream=meta["config_name"], variant=meta["variant"], seed=meta["seed"])
    drift = next(iter(meta["drift_points"]), None)
    before = np.isin(np.arange(meta["d"]), meta["stream"]["rel_before"])
    after = np.isin(np.arange(meta["d"]), meta["stream"]["rel_after"])
    methods = ("R1", "R4", "R3_mean", *(f"R2_{decay}" for decay in FORGETTING),
               *(f"window_{size}" for size in WINDOW_SIZES))
    weights = {
        method: {t: vector(row, method) for t, row in checkpoints.items()}
        for method in methods
    }

    valid_local = defaultdict(dict)
    for row in local:
        t = int(row["t"])
        if t in checkpoints and row["first_sample"]:
            w = vector(row, "R3")
            if np.isfinite(w).all():
                valid_local[int(row["uid"])][t] = (int(row["first_sample"]), int(row["last_sample"]), w)

    endpoints = []
    # Compare distinct windows at the two ends of the same rule's observed history.
    for uid, observations in valid_local.items():
        first, last = min(observations), max(observations)
        if observations[first][1] == observations[last][1]:
            continue
        first_sample, first_end, first_w = observations[first]
        last_sample, last_end, last_w = observations[last]
        endpoints.append(dict(
            **keys, uid=uid, first_t=first, last_t=last,
            first_start_sample=first_sample, first_end_sample=first_end,
            last_start_sample=last_sample, last_end_sample=last_end,
            first_precision=local_precision_at_endpoint(first_sample, first_end, first_w, drift, before, after),
            last_precision=local_precision_at_endpoint(last_sample, last_end, last_w, drift, before, after),
            jaccard=topk_jaccard(first_w, last_w, K)
            if not is_constant(first_w) and not is_constant(last_w) else np.nan,
            first_weights=" ".join(f"{w:.8g}" for w in first_w),
            last_weights=" ".join(f"{w:.8g}" for w in last_w),
        ))

    segments = [(0, drift or meta["n"], before)]
    if drift is not None:
        segments.append((drift, meta["n"], after))
    stability_rows, precision_rows = [], []
    for segment, (start, end, truth) in enumerate(segments):
        for method in ("R1", "R4", *(f"window_{size}" for size in WINDOW_SIZES)):
            size = int(method.split("_")[-1]) if method.startswith("window_") else 200
            clean = {
                t: w for t, w in weights[method].items()
                if start + size <= t <= end
            }
            jaccard, pairs = stability(clean)
            stability_rows.append(dict(**keys, segment=segment, method=method, jaccard=jaccard,
                                       pairs=pairs))

        local_scores, local_precision = [], []
        for observations in valid_local.values():
            # H3 pairs the same rule in one concept. Its last 200 wins may span much more stream time.
            clean = {t: w for t, (first, _, w) in observations.items()
                     if start < first and start + 200 <= t <= end}
            score, pairs = stability(clean)
            if pairs:
                local_scores.append((score, pairs))
            local_precision.extend(topk_precision(w, truth, K) for w in clean.values())
        local_pairs = sum(pairs for _, pairs in local_scores)
        local_mean = (
            sum(score * pairs for score, pairs in local_scores) / local_pairs
            if local_pairs else np.nan
        )
        stability_rows.append(dict(**keys, segment=segment, method="R3", jaccard=local_mean,
                                   pairs=local_pairs))
        precision_rows.append(dict(**keys, segment=segment, method="R3",
                                   precision=mean(local_precision), observations=len(local_precision)))
        for method in ("R1", *(f"R2_{decay}" for decay in FORGETTING)):
            precision = [topk_precision(w, truth, K) for t, w in weights[method].items()
                         if start + 200 <= t <= end]
            precision_rows.append(dict(**keys, segment=segment, method=method,
                                       precision=mean(precision), observations=len(precision)))

    recovery = []
    if drift is not None:
        # H6 compares R1, R4 and aggregated R3. R2 recovery is a separate descriptive check.
        for method in methods:
            times = sorted(weights[method])
            precision = [
                topk_precision(weights[method][t], after, K)
                if np.isfinite(weights[method][t]).all() else np.nan for t in times
            ]
            recovery.append(dict(**keys, method=method,
                                 **ranking_recovery(times, precision, drift)))

    run = dict(**keys, accuracy=metrics["acc_final"], rules_final=metrics["rules_final"],
               coverage_pct=mean([float(row["r3_coverage_pct"]) for row in checkpoints.values()]),
               ms_per_sample=metrics["wf3_ms_per_sample"])
    return run, stability_rows, precision_rows, recovery, endpoints


def stability_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["method"] in ("R1", "R3", "R4"):
            key = row["stream"], row["variant"], row["segment"], row["method"]
            groups[key].append(row)
    output = []
    for (stream, variant, segment, method), group in sorted(groups.items()):
        scores = [row["jaccard"] for row in group if row["pairs"]]
        output.append(dict(
            stream=stream, variant=variant, segment=segment, method=method,
            mean_jaccard=mean(scores), sd_jaccard=float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
            valid_seeds=len(scores), total_seeds=len(group), pairs=sum(row["pairs"] for row in group),
        ))
    return output


def h6_summary(recovery):
    by_seed = defaultdict(dict)
    for row in recovery:
        if row["stream"] == "s1_gauss_abrupt" and row["variant"] == "open":
            by_seed[row["seed"]][row["method"]] = row
    output = []
    for seed, rows in sorted(by_seed.items()):
        summary = {"seed": seed}
        for method in ("R1", "R3_mean", "R4"):
            for key in ("recovery", "recovered", "censored_time"):
                summary[f"{method}_{key}"] = rows[method][key]
        for method in ("R1", "R3_mean"):
            summary[f"R4_minus_{method}"] = rows["R4"]["recovery"] - rows[method]["recovery"]
        output.append(summary)
    return output


def make_curve(outdir, stability_rows, recovery):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, (stable_ax, recovery_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for variant in sorted({row["variant"] for row in recovery}):
        variant_rows = []
        for size in WINDOW_SIZES:
            method = f"window_{size}"
            scores = [r["jaccard"] for r in stability_rows if r["stream"] == "s3_gauss_stationary"
                      and r["variant"] == variant and r["method"] == method and r["pairs"]]
            runs = [r for r in recovery if r["stream"] == "s1_gauss_abrupt"
                    and r["variant"] == variant and r["method"] == method]
            recovered = [r["recovery"] for r in runs if r["recovered"]]
            variant_rows.append(dict(
                variant=variant, window=size, s3_jaccard=mean(scores),
                s3_jaccard_sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                s3_seeds=len(scores), s1_recovery=mean(recovered),
                s1_recovery_sd=float(np.std(recovered, ddof=1)) if len(recovered) > 1 else np.nan,
                s1_seeds=len(runs), s1_censored=sum(not r["recovered"] for r in runs),
                predicted_min=size / 2, predicted_max=size + 50,
            ))
        rows.extend(variant_rows)
        marker = "o" if variant == "open" else "s"
        for ax, key in ((stable_ax, "s3_jaccard"), (recovery_ax, "s1_recovery")):
            values = [r[key] for r in variant_rows]
            spreads = np.nan_to_num([r[f"{key}_sd"] for r in variant_rows], nan=0.0)
            ax.errorbar(WINDOW_SIZES, values, yerr=spreads, marker=marker, markerfacecolor="none",
                        markersize=5 if variant == "open" else 8, capsize=3, label=variant)
    if not rows:
        plt.close(fig)
        return
    recovery_ax.fill_between(WINDOW_SIZES, [w / 2 for w in WINDOW_SIZES],
                             [w + 50 for w in WINDOW_SIZES], alpha=0.15, label="Predicted range")
    stable_ax.set(ylabel="Mean top-3 Jaccard", title="S3 · Stability at lag 200", ylim=(0, 1.05))
    recovery_ax.set(ylabel="Samples after drift", title="S1 · Ranking recovery")
    for ax in (stable_ax, recovery_ax):
        ax.set(xscale="log", xlabel="Window size (samples)")
        ax.set_xticks(WINDOW_SIZES, labels=WINDOW_SIZES)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "window_tradeoff.png", dpi=160)
    plt.close(fig)
    write_csv(outdir / "window_tradeoff.csv", rows)


def analyze(root):
    files = sorted(root.rglob("wf3_checkpoints.csv"))
    if not files:
        raise SystemExit(f"No Workflow 3 runs in {root}")
    outdir = root / "analysis"
    outdir.mkdir(exist_ok=True)
    runs, stability_rows, precision_rows, recovery, endpoints = [], [], [], [], []
    for file in files:
        run, stable, precision, recovered, comparisons = summarize_run(file)
        runs.append(run)
        stability_rows.extend(stable)
        precision_rows.extend(precision)
        recovery.extend(recovered)
        endpoints.extend(comparisons)
    for name, rows in (("runs", runs), ("h3_stability", stability_rows),
                       ("h3_summary", stability_summary(stability_rows)),
                       ("h6_open", h6_summary(recovery)),
                       ("validity", precision_rows), ("ranking_recovery", recovery),
                       ("first_last_local", endpoints)):
        if rows:
            write_csv(outdir / f"{name}.csv", rows)
    make_curve(outdir, stability_rows, recovery)
    print(f"Analyzed {len(runs)} runs. Results: {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results_wf3"))
    args = parser.parse_args()
    analyze(args.results)


if __name__ == "__main__":
    main()
