#!/usr/bin/env python
"""WF5 analysis: window comparisons (PROTOCOL.md §4) and restabilisation time (§5) on saved runs.

    python wf5_windows/analyze_wf5.py                      # all streams
    python wf5_windows/analyze_wf5.py --streams S3 Elec2   # Minês's part
    python wf5_windows/analyze_wf5.py --streams S1 S2 S4   # Bruno's part

Reads only saved checkpoints (R1 = G). R1 depends only on the data, so `ref` and `open` have identical
G (checked below); the `ref` runs are used. Writes CSVs and table_wf5.md to wf5_windows/results/.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.stats import bootstrap_ci, wilcoxon_paired, holm
from wf5_windows.windows import window_analysis, restabilisation, WIDTH, LAG, PCT, CONSECUTIVE

OUT = ROOT / "wf5_windows" / "results"
ELEC2_NAMES = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

# name -> (run folder holding seed_<s>/, k, role). Drift points/widths come from each run's meta.json.
STREAMS = {
    "S3": ("results_wf5/s3_gauss_stationary/ref", 3, "stationary control"),
    "Elec2": ("results_relevance/wf1_elec2_nodate", 3, "real, no ground truth"),
    "S1": ("results_wf2/s1_gauss_abrupt/ref", 3, "abrupt relevance swap"),
    "S2": ("results_wf2/s2_gauss_gradual/ref", 3, "gradual relevance swap"),
    "S4": ("results_wf2/s4_sea_drift/ref", 2, "negative control (relevant set unchanged)"),
}


def load_runs(name):
    folder, k, role = STREAMS[name]
    runs = []
    for sd in sorted((ROOT / folder).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        z = np.load(sd / "checkpoints.npz")
        meta = json.loads((sd / "meta.json").read_text()) if (sd / "meta.json").exists() else {}
        st = meta.get("stream", {})
        n = int(meta.get("n", z["t"][-1]))
        dps = meta.get("drift_points") or []
        widths = meta.get("drift_widths") or [st.get("drift_width", 0)] * len(dps)
        names = ELEC2_NAMES if name == "Elec2" else [f"x{i}" for i in range(z["G"].shape[1])]
        runs.append(dict(seed=int(sd.name.split("_")[1]), t=z["t"], G=z["G"], n=n, dps=dps, widths=widths,
                         names=names, rel_before=st.get("rel_before"), rel_after=st.get("rel_after"),
                         recovery=_wf2_recovery(sd)))
    return runs, k, role


def _wf2_recovery(sd):
    m = sd / "metrics.json"
    if not m.exists():
        return {}
    d = json.loads(m.read_text())
    return {k: d.get(k) for k in ("recovery/G/drift1", "recovery_from_dp/G/drift1")}


def check_variants_identical():
    for name, (folder, _, _) in STREAMS.items():
        if not folder.endswith("/ref"):
            continue
        o = ROOT / folder.replace("/ref", "/open")
        if not o.exists():
            continue
        for sd in (ROOT / folder).glob("seed_*"):
            a = np.load(sd / "checkpoints.npz")["G"]; b = np.load(o / sd.name / "checkpoints.npz")["G"]
            assert np.array_equal(a, b), f"{name} {sd.name}: ref and open G differ"


def fmt(x, digits=3):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0:
        return "–"
    if x.size == 1:
        return f"{x[0]:.{digits}f}"
    lo, hi = bootstrap_ci(x)
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def analyse(names):
    per_run, pairs, means, restab = [], [], [], []
    for name in names:
        runs, k, role = load_runs(name)
        for r in runs:
            wa = window_analysis(r["t"], r["G"], k, r["n"], r["dps"], r["widths"])
            for j, (row, nc) in enumerate(zip(wa["W"], wa["n_clean"])):
                for f, v in zip(r["names"], row):
                    means.append(dict(stream=name, seed=r["seed"], window=j, start=j * WIDTH, n_clean=int(nc),
                                      feature=f, relevance=v))
            for row in wa["consecutive"]:
                pairs.append(dict(stream=name, seed=r["seed"], **row))
            stat = [p for p in wa["consecutive"] if not p["crosses_drift"]]
            cross = [p for p in wa["consecutive"] if p["crosses_drift"]]
            rec = dict(stream=name, seed=r["seed"], k=k, n_windows=len(wa["W"]),
                       first=wa["first"], last=wa["last"],
                       first_last_jaccard=wa["first_last"][0], first_last_wtau=wa["first_last"][1],
                       consec_jaccard_mean=np.nanmean([p["jaccard"] for p in stat]),
                       consec_jaccard_min=np.nanmin([p["jaccard"] for p in stat]),
                       consec_wtau_mean=np.nanmean([p["wtau"] for p in stat]),
                       consec_wtau_min=np.nanmin([p["wtau"] for p in stat]),
                       n_consec_stationary=len(stat),
                       crossing_jaccard_min=np.nanmin([p["jaccard"] for p in cross]) if cross else np.nan,
                       crossing_wtau_min=np.nanmin([p["wtau"] for p in cross]) if cross else np.nan,
                       pre=wa["pre"], pre_last_jaccard=np.nan, pre_last_wtau=np.nan,
                       topk_first="", topk_last="", topk_pre="")
            order = lambda v: "{" + ",".join(r["names"][i] for i in sorted(np.argsort(-v, kind="stable")[:k])) + "}"
            rec["topk_first"], rec["topk_last"] = order(wa["W"][wa["first"]]), order(wa["W"][wa["last"]])
            if wa["pre_last"] is not None:
                rec["pre_last_jaccard"], rec["pre_last_wtau"] = wa["pre_last"]
                rec["topk_pre"] = order(wa["W"][wa["pre"]])
            per_run.append(rec)
            if name == "S3":                       # placebo: a fake drift at 10,000 on the stationary stream
                rs = restabilisation(r["t"], r["G"], [r["n"] // 2])
                restab.append(dict(stream="S3 placebo (no drift; fake dp 10,000)", seed=r["seed"], **rs))
            if r["dps"]:
                rs = restabilisation(r["t"], r["G"], r["dps"], r["widths"])
                restab.append(dict(stream=name, seed=r["seed"], **rs,
                                   wf2_ranking_recovery=r["recovery"].get("recovery/G/drift1"),
                                   wf2_ranking_recovery_from_dp=r["recovery"].get("recovery_from_dp/G/drift1")))
    return (pd.DataFrame(per_run), pd.DataFrame(pairs), pd.DataFrame(means), pd.DataFrame(restab))


def write_table(per_run, restab, path):
    L = ["# WF5: window analysis and restabilisation of R1",
         "",
         f"Non-overlapping {WIDTH:,}-sample windows; window relevance = mean of G over the clean checkpoints "
         "in the window (PROTOCOL.md §4). Mean ± SD over seeds [95% bootstrap CI, 10,000 resamples]; "
         "Elec2 is one run (no CI). Generated by `wf5_windows/analyze_wf5.py`.",
         "",
         "## Window comparisons",
         "",
         "| Stream | k | Top-k first → last | Jaccard first vs last | wτ first vs last | Jaccard consecutive (min) | wτ consecutive (mean) | Jaccard pre-drift vs last | wτ pre-drift vs last |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, g in per_run.groupby("stream", sort=False):
        tk = f"{g.topk_first.mode()[0]} → {g.topk_last.mode()[0]}"
        if g.topk_first.nunique() > 1 or g.topk_last.nunique() > 1:
            tk += " (most common)"
        L.append(f"| {s} | {g.k.iloc[0]} | {tk} | {fmt(g.first_last_jaccard)} | {fmt(g.first_last_wtau)} | "
                 f"{fmt(g.consec_jaccard_min)} | {fmt(g.consec_wtau_mean)} | {fmt(g.pre_last_jaccard)} | "
                 f"{fmt(g.pre_last_wtau)} |")
    L += ["", "Consecutive pairs exclude the pair(s) whose windows overlap a drift or transition; on drift "
          "streams that pair is reported separately in `windows_per_run.csv` (`crossing_*`). "
          "Pre-drift window = last window ending at or before the transition start (9,000–10,000 on S1/S4, "
          "8,000–9,000 on S2).", ""]
    if len(restab):
        L += ["## Restabilisation time (samples)", "",
              f"Weighted τ at lag {LAG} back above the {PCT}th percentile of its clean pre-drift values for "
              f"{CONSECUTIVE} checkpoints in a row, counted from the trough "
              "(minimum within 1,000 samples after the drift, as for accuracy recovery in §10). The S3 placebo row applies the "
              "same rule to the stationary stream with a fake drift at 10,000: it is the noise floor of the time.", "",
              "| Stream | Threshold (wτ) | Dropped | Restabilised | From transition end (protocol) | From drift point | WF2 ranking recovery (from dp) |",
              "|---|---|---|---|---|---|---|"]
        for s, g in restab.groupby("stream", sort=False):
            L.append(f"| {s} | {fmt(g.threshold)} | {int(g.dropped.sum())}/{len(g)} | {int(g.restabilised.sum())}/{len(g)} | "
                     f"{fmt(g.restab, 0)} | {fmt(g.restab_from_dp, 0)} | {fmt(g.wf2_ranking_recovery_from_dp, 0)} |")
        L += ["", "Censored runs (never back above the threshold) are NaN in the means; their lower bound is in "
              "`restab_per_run.csv` (`restab_censored`)."]
        plac = restab[restab.stream.str.startswith("S3 placebo")].sort_values("seed")
        tests = []
        for st in ("S1", "S2"):
            g = restab[restab.stream == st].sort_values("seed")
            if len(g) and len(plac) and list(g.seed) == list(plac.seed):
                for col, lab in (("trough", "trough wτ"), ("restab_from_dp", "restabilisation from dp")):
                    w = wilcoxon_paired(g[col].values, plac[col].values)
                    tests.append((f"{st} vs S3 placebo: {lab}", w))
        if tests:
            adj = holm([w["p"] for _, w in tests])
            L += ["", "### Paired tests against the placebo (same seed = same pre-drift data; Wilcoxon, Holm over these tests)", "",
                  "| Comparison | n | W | p | p (Holm) | rank-biserial |", "|---|---|---|---|---|---|"]
            for (lab, w), pa in zip(tests, adj):
                L.append(f"| {lab} | {w['n']} | {w['W']:.1f} | {w['p']:.4f} | {pa:.4f} | {w['r_rb']:+.2f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="*", default=list(STREAMS), choices=list(STREAMS))
    a = ap.parse_args()
    check_variants_identical()
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, pairs, means, restab = analyse(a.streams)
    tag = "" if a.streams == list(STREAMS) else "_" + "_".join(a.streams)
    per_run.to_csv(OUT / f"windows_per_run{tag}.csv", index=False)
    pairs.to_csv(OUT / f"consecutive_pairs{tag}.csv", index=False)
    means.to_csv(OUT / f"window_means{tag}.csv", index=False)
    if len(restab):
        restab.to_csv(OUT / f"restab_per_run{tag}.csv", index=False)
    write_table(per_run, restab, OUT / f"table_wf5{tag}.md")
    print((OUT / f"table_wf5{tag}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
