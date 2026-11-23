#!/usr/bin/env python
"""SQ2 statistics (PROTOCOL.md sections 10 and 11) from the saved runs in results_wf2/.

    python -m wf2_drift_validity.statistical_analysis

Writes results_wf2/statistics/:
  descriptive.csv   per stream/variant/metric: n, recovered, % censored, mean, SD (ddof=1), median, 95% CI
  comparisons.csv   paired Wilcoxon signed-rank + matched-pairs rank-biserial + sign test, Holm per family
  window_study.csv  H2 second part: R1 recovery on S1 per window size, plus Page's trend test
  sq2_primary.json  the primary SQ2 result against the H2 interval (100-250 samples)

Conventions:
  * Recovery values are the censored ones (`*_censored`): a run that never recovers counts with the time
    to the end of its segment (a lower bound), as the protocol prescribes; recovered/censored counts are
    always reported next to them.
  * Accuracy recovery is reported for `open` only. `ref` predicts at chance (PROTOCOL.md section 7), so its
    accuracy "recovery" is not interpretable.
  * S2's protocol ranking recovery is counted from the END of the transition and has a floor of one
    checkpoint (50 samples). Abrupt vs gradual is therefore compared with recovery_from_dp/*, which times
    both streams from the drift point (10,000).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, rankdata, binomtest, page_trend_test

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_wf2"
SEEDS = range(10)
H2_INTERVAL = (100, 250)
WINDOWS = (50, 100, 200, 500, 1000)
CHECKPOINT = 50


def load_metric(stream, variant, metric):
    vals = []
    for seed in SEEDS:
        with open(RESULTS / stream / variant / f"seed_{seed}" / "metrics.json") as f:
            vals.append(json.load(f)[metric])
    return np.asarray(vals, dtype=float)


def bootstrap_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def describe(values, recovered=None):
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(values)
    out = dict(n=int(np.isfinite(values).sum()), mean=float(np.nanmean(values)),
               sd=float(np.nanstd(values, ddof=1)), median=float(np.nanmedian(values)), ci_low=lo, ci_high=hi)
    if recovered is not None:
        rec = np.asarray(recovered, dtype=float)
        out.update(recovered=int(np.nansum(rec)), censored_pct=float(100.0 * np.mean(rec == 0)))
    return out


def paired_comparison(a, b):
    """Wilcoxon signed-rank on a - b, matched-pairs rank-biserial (positive = a larger), sign test."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    diff = (a[mask] - b[mask]).round(10)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return dict(n_pairs=int(mask.sum()), n_nonzero=0, wilcoxon_p=1.0, rank_biserial=0.0, sign_test_p=1.0)
    ranks = rankdata(np.abs(nonzero))
    pos, neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return dict(n_pairs=int(mask.sum()), n_nonzero=int(len(nonzero)),
                wilcoxon_p=float(wilcoxon(nonzero).pvalue),
                rank_biserial=float((pos - neg) / (pos + neg)),
                sign_test_p=float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), p=0.5).pvalue))


def holm_correction(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adjusted = np.maximum.accumulate((n - np.arange(n)) * p[order])
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


DESCRIPTIVE = [
    # (label, metric, recovered-flag metric or None, variants)
    ("R1 ranking recovery (protocol: from transition end)", "recovery_censored/G/drift1", "recovered/G/drift1", ("ref", "open")),
    ("R1 ranking recovery from the drift point", "recovery_from_dp_censored/G/drift1", "recovered/G/drift1", ("ref", "open")),
    ("R4 ranking recovery (protocol)", "recovery_censored/A/drift1", "recovered/A/drift1", ("ref", "open")),
    ("Accuracy recovery (open only)", "acc/recovery_censored/drift1", "acc/recovered/drift1", ("open",)),
    ("Accuracy drop", "acc/drop/drift1", None, ("open",)),
    ("Final accuracy", "acc_final", None, ("ref", "open")),
    ("Final macro-F1", "f1_final", None, ("ref", "open")),
    ("Final rule count", "rules_final", None, ("ref", "open")),
]


def descriptive_rows():
    rows = []
    for stream in ("s1_gauss_abrupt", "s2_gauss_gradual"):
        for label, metric, flag, variants in DESCRIPTIVE:
            for variant in variants:
                vals = load_metric(stream, variant, metric)
                rec = load_metric(stream, variant, flag) if flag else None
                if flag == "recovered/G/drift1" and "from_dp" in metric:
                    rec = np.isfinite(load_metric(stream, variant, "recovery_from_dp/G/drift1")).astype(float)
                row = dict(stream=stream, variant=variant, label=label, metric=metric, **describe(vals, rec))
                if metric == "recovery_censored/G/drift1":
                    row["at_floor"] = int(np.sum(vals == CHECKPOINT))    # recovered at the first checkpoint
                rows.append(row)
    return rows


def comparisons():
    rows = []

    def add(family, name, a, b, meaning):
        rows.append(dict(family=family, comparison=name, meaning=meaning, **paired_comparison(a, b)))

    for stream in ("s1_gauss_abrupt", "s2_gauss_gradual"):
        add("SQ2", f"{stream}/open: R1 vs accuracy recovery",
            load_metric(stream, "open", "recovery_censored/G/drift1"),
            load_metric(stream, "open", "acc/recovery_censored/drift1"),
            "negative = the ranking recovers before accuracy (both timed from the transition end)")
    add("SQ2", "S1 vs S2: R1 recovery from the drift point (same origin)",
        load_metric("s1_gauss_abrupt", "ref", "recovery_from_dp_censored/G/drift1"),
        load_metric("s2_gauss_gradual", "ref", "recovery_from_dp_censored/G/drift1"),
        "negative = abrupt recovers sooner than gradual; R1 is identical for ref and open")
    for stream in ("s1_gauss_abrupt", "s2_gauss_gradual"):
        for variant in ("ref", "open"):
            add("H6 (secondary)", f"{stream}/{variant}: R1 vs R4 recovery",
                load_metric(stream, variant, "recovery_censored/G/drift1"),
                load_metric(stream, variant, "recovery_censored/A/drift1"),
                "negative = R1 recovers before R4; R4 never recovered, so this only restates 10/10 vs 0/10")
    for family in dict.fromkeys(r["family"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["family"] == family]
        for i, p in zip(idx, holm_correction([rows[i]["wilcoxon_p"] for i in idx])):
            rows[i]["holm_p"] = float(p)
            rows[i]["holm_family_size"] = len(idx)
    return rows


def window_study():
    base = RESULTS / "s1_window_study"
    if not base.exists():
        return [], None
    data, rows = [], []
    for w in WINDOWS:
        vals = load_metric("s1_window_study", f"w{w}", "recovery_censored/G/drift1")
        rec = load_metric("s1_window_study", f"w{w}", "recovered/G/drift1")
        data.append(vals)
        rows.append(dict(window=w, **describe(vals, rec)))
    data = np.column_stack(data)                       # rows = seeds (blocks), columns = windows (ordered)
    page = page_trend_test(data)                       # H1: recovery increases with the window size
    trend = dict(test="Page's trend test, seeds as blocks, windows ordered 50 < 100 < 200 < 500 < 1000",
                 L=float(page.statistic), p=float(page.pvalue), method=page.method,
                 per_seed_monotone=int(np.sum(np.all(np.diff(data, axis=1) >= 0, axis=1))))
    return rows, trend


def main():
    out = RESULTS / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    desc = descriptive_rows()
    comps = comparisons()
    write_csv(out / "descriptive.csv", desc)
    write_csv(out / "comparisons.csv", comps)

    s1 = load_metric("s1_gauss_abrupt", "ref", "recovery_censored/G/drift1")
    s1_open = load_metric("s1_gauss_abrupt", "open", "recovery_censored/G/drift1")
    primary = dict(metric="R1 ranking recovery on S1 (samples after the drift)", values=s1.tolist(),
                   identical_for_ref_and_open=bool(np.array_equal(s1, s1_open)),
                   recovered=int(load_metric("s1_gauss_abrupt", "ref", "recovered/G/drift1").sum()),
                   **describe(s1), h2_interval=list(H2_INTERVAL))
    primary["ci_inside_h2_interval"] = bool(H2_INTERVAL[0] <= primary["ci_low"] and primary["ci_high"] <= H2_INTERVAL[1])
    wrows, trend = window_study()
    if wrows:
        write_csv(out / "window_study.csv", wrows)
        primary["window_study"] = dict(rows=wrows, trend=trend)
    json.dump(primary, open(out / "sq2_primary.json", "w"), indent=2)

    print(f"Primary SQ2: {primary['mean']:.1f} ± {primary['sd']:.1f} samples (SD, ddof=1), "
          f"95% CI [{primary['ci_low']:.1f}, {primary['ci_high']:.1f}], recovered {primary['recovered']}/10, "
          f"CI inside {H2_INTERVAL}: {primary['ci_inside_h2_interval']}")
    for r in comps:
        print(f"[{r['family']}] {r['comparison']} | p={r['wilcoxon_p']:.6f} | Holm={r['holm_p']:.6f} "
              f"| r_rb={r['rank_biserial']:+.3f} | sign p={r['sign_test_p']:.6f} | nonzero={r['n_nonzero']}")
    for r in wrows:
        print(f"window {r['window']:5d}: {r['mean']:7.1f} ± {r['sd']:6.1f}  median {r['median']:6.1f}  "
              f"CI [{r['ci_low']:.1f}, {r['ci_high']:.1f}]  recovered {r['recovered']}/10")
    if trend:
        print(f"Page trend: L={trend['L']:.1f} p={trend['p']:.2e}; seeds monotone: {trend['per_seed_monotone']}/10")
    print("Results saved to:", out)


if __name__ == "__main__":
    main()
