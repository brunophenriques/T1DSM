"""Generate Workflow 3 tables and the window-size curve from saved runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

import numpy as np

from relevance.metrics import is_constant, topk_jaccard, topk_precision
from wf2_drift_validity.recovery_metrics import ranking_recovery
from wf3_relevance.collect import FORGETTING, WARMUP, WINDOW_SIZES


LAG = 200
K = 3
DRIFT_OFFSETS = (0, 50, 100, 150, 200, 500)
EVIDENCE_TABLES = (
    "h3_summary", "h3_paired", "h3_paired_summary", "h6_open", "ranking_recovery",
    "validity", "r2_checkpoints", "window_tradeoff", "first_last_summary", "rule_51", "runs",
)


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


def stability_scores(vectors):
    scores = {}
    for t, current in vectors.items():
        previous = vectors.get(t - LAG)
        if previous is not None and not is_constant(current) and not is_constant(previous):
            scores[t] = topk_jaccard(previous, current, K)
    return scores


def paired_stability(global_scores, local_scores):
    """Use the same time pair and weight for both methods in each rule-level comparison."""
    matched = [(uid, t, global_scores[t], score) for uid, scores in local_scores.items()
               for t, score in scores.items() if t in global_scores]
    r1 = mean([global_score for _, _, global_score, _ in matched])
    r3 = mean([local_score for _, _, _, local_score in matched])
    return dict(
        r1_matched=r1, r3_matched=r3, r1_minus_r3=r1 - r3,
        matched_rule_pairs=len(matched), matched_time_pairs=len({t for _, t, _, _ in matched}),
        eligible_r1_time_pairs=len(global_scores), matched_rules=len({uid for uid, _, _, _ in matched}),
    )


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
    active_by_time = defaultdict(set)
    for row in local:
        t = int(row["t"])
        uid = int(row["uid"])
        if t in checkpoints:
            active_by_time[t].add(uid)
        if t in checkpoints and row["first_sample"]:
            w = vector(row, "R3")
            if np.isfinite(w).all():
                valid_local[uid][t] = (int(row["first_sample"]), int(row["last_sample"]), w)

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
    stability_rows, precision_rows, paired_rows = [], [], []
    precision_methods = ("R1", *(f"R2_{decay}" for decay in FORGETTING))
    for segment, (start, end, truth) in enumerate(segments):
        for method in ("R1", "R4", *(f"window_{size}" for size in WINDOW_SIZES)):
            size = int(method.split("_")[-1]) if method.startswith("window_") else 200
            clean = {
                t: w for t, w in weights[method].items()
                if start + size <= t <= end
            }
            scores = stability_scores(clean)
            if method == "R1":
                global_scores = scores
            stability_rows.append(dict(**keys, segment=segment, method=method,
                                       jaccard=mean(list(scores.values())), pairs=len(scores)))

        local_scores, local_precision = {}, []
        for uid, observations in valid_local.items():
            # H3 pairs the same rule in one concept. Its last 200 wins may span much more stream time.
            clean = {t: w for t, (first, _, w) in observations.items()
                     if start < first and start + 200 <= t <= end}
            local_scores[uid] = stability_scores(clean)
            local_precision.extend(topk_precision(w, truth, K) for w in clean.values())
        scores = [score for rule_scores in local_scores.values() for score in rule_scores.values()]
        stability_rows.append(dict(**keys, segment=segment, method="R3",
                                   jaccard=mean(scores), pairs=len(scores)))
        active_rules = {uid for t, uids in active_by_time.items()
                        if start + 200 <= t <= end for uid in uids}
        paired_rows.append(dict(**keys, segment=segment, active_rules=len(active_rules),
                                **paired_stability(global_scores, local_scores)))
        precision_rows.append(dict(**keys, segment=segment, method="R3",
                                   precision=mean(local_precision), observations=len(local_precision)))
        for method in precision_methods:
            precision = [topk_precision(w, truth, K) for t, w in weights[method].items()
                         if start + 200 <= t <= end]
            precision_rows.append(dict(**keys, segment=segment, method=method,
                                       precision=mean(precision), observations=len(precision)))

    recovery, drift_precision = [], []
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
        if meta["config_name"] == "s1_gauss_abrupt":
            for offset in DRIFT_OFFSETS:
                t = drift + offset
                if t not in checkpoints:
                    continue
                truth = before if offset == 0 else after
                for method in precision_methods:
                    drift_precision.append(dict(
                        **keys, checkpoint=t, method=method,
                        precision=topk_precision(weights[method][t], truth, K),
                    ))

    run = dict(**keys, accuracy=metrics["acc_final"], rules_final=metrics["rules_final"],
               coverage_pct=mean([float(row["r3_coverage_pct"]) for row in checkpoints.values()]),
               ms_per_sample=metrics["wf3_ms_per_sample"])
    return run, stability_rows, precision_rows, recovery, endpoints, paired_rows, drift_precision


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
            result = rows[method]
            summary[f"{method}_recovered"] = result["recovered"]
            summary[f"{method}_recovery"] = result["recovery"] if result["recovered"] else None
            summary[f"{method}_censor_limit"] = None if result["recovered"] else result["censored_time"]
        for method in ("R1", "R3_mean"):
            summary[f"R4_minus_{method}"] = (
                rows["R4"]["recovery"] - rows[method]["recovery"]
                if rows["R4"]["recovered"] and rows[method]["recovered"] else None
            )
        output.append(summary)
    return output


def paired_h3_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stream"], row["variant"], row["segment"])].append(row)
    output = []
    for (stream, variant, segment), group in sorted(groups.items()):
        diffs = [row["r1_minus_r3"] for row in group if row["matched_rule_pairs"]]
        counts = {key: sum(row[key] for row in group) for key in (
            "matched_rule_pairs", "matched_time_pairs", "eligible_r1_time_pairs", "matched_rules", "active_rules",
        )}
        output.append(dict(
            stream=stream, variant=variant, segment=segment,
            mean_r1_minus_r3=mean(diffs), sd_r1_minus_r3=np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan,
            positive_seeds=sum(diff > 0 for diff in diffs), valid_seeds=len(diffs), total_seeds=len(group),
            **counts,
        ))
    return output


def checkpoint_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["checkpoint"], row["method"])].append(row["precision"])
    return [dict(variant=variant, checkpoint=t, method=method, mean_precision=mean(scores), seeds=len(scores))
            for (variant, t, method), scores in sorted(groups.items())]


def endpoint_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stream"], row["variant"])].append(row)
    return [dict(stream=stream, variant=variant, run_rule_pairs=len(group),
                 mean_jaccard=mean([r["jaccard"] for r in group if np.isfinite(r["jaccard"])]))
            for (stream, variant), group in sorted(groups.items())]


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
        return False
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
    return True


def analyze(root, evidence_dir=None):
    files = sorted(root.rglob("wf3_checkpoints.csv"))
    if not files:
        raise SystemExit(f"No Workflow 3 runs in {root}")
    outdir = root / "analysis"
    outdir.mkdir(exist_ok=True)
    runs, stability_rows, precision_rows, recovery, endpoints = [], [], [], [], []
    paired, drift_precision = [], []
    for file in files:
        run, stable, precision, recovered, comparisons, matched, checkpoints = summarize_run(file)
        runs.append(run)
        stability_rows.extend(stable)
        precision_rows.extend(precision)
        recovery.extend(recovered)
        endpoints.extend(comparisons)
        paired.extend(matched)
        drift_precision.extend(checkpoints)
    rule_51 = [row for row in endpoints if (
        row["stream"], row["variant"], row["seed"], row["uid"]
    ) == ("s1_gauss_abrupt", "open", 0, 51)]
    written = set()
    for name, rows in (("runs", runs), ("h3_stability", stability_rows),
                       ("h3_summary", stability_summary(stability_rows)),
                       ("h3_paired", paired), ("h3_paired_summary", paired_h3_summary(paired)),
                       ("h6_open", h6_summary(recovery)),
                       ("validity", precision_rows), ("ranking_recovery", recovery),
                       ("r2_checkpoints", checkpoint_summary(drift_precision)),
                       ("first_last_local", endpoints), ("first_last_summary", endpoint_summary(endpoints)),
                       ("rule_51", rule_51)):
        if rows:
            write_csv(outdir / f"{name}.csv", rows)
            written.add(name)
    if make_curve(outdir, stability_rows, recovery):
        written.add("window_tradeoff")
    if evidence_dir is not None:
        missing = set(EVIDENCE_TABLES) - written
        if missing:
            raise SystemExit(f"Run the full S1/S3 suite before exporting: missing {', '.join(sorted(missing))}")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_TABLES:
            copyfile(outdir / f"{name}.csv", evidence_dir / f"{name}.csv")
    print(f"Analyzed {len(runs)} runs. Results: {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results_wf3"))
    parser.add_argument("--evidence-dir", type=Path, help="export review CSVs after the full S1/S3 suite")
    args = parser.parse_args()
    analyze(args.results, args.evidence_dir)


if __name__ == "__main__":
    main()
