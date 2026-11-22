#!/usr/bin/env python
"""SQ1 analysis. Reads results_relevance/<condition>/seed_*/checkpoints.npz (from run_relevance.py) and writes
   wf1_stability/table1_sq1.md      SQ1 row of Table 1 (+ overlap and sanity tables)
   wf1_stability/sq1_per_run.csv    every metric for every run
   wf1_stability/fig2a_jaccard_lag200.png/.pdf
Run from the repo root:  python wf1_stability/analyze_sq1.py
"""
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.stability import (metric_fns, segments, lagged_series, topk_switch_rate, per_feature_sd,
                                 rolling_nanmean)
from relevance.stats import bootstrap_ci, block_bootstrap_ci, shuffle_rank_pvalue

RES, OUT = ROOT / "results_relevance", Path(__file__).resolve().parent
CONDITIONS = {"S1": "mve", "S3": "wf1_s3", "Elec2": "wf1_elec2", "Elec2 shuffled": "wf1_elec2_shuffled"}  # label -> results folder (edit S1)
LAGS, PRIMARY_LAG, BLOCK_SAMPLES, SMOOTH_CK = [50, 100, 150, 200, 500, 2000], 200, 500, 10
METRICS = ["jaccard", "tau_b", "weighted_tau", "spearman"]


def load(folder):
    f = RES / folder / "summary.json"
    if not f.exists():
        return None
    meta = json.load(open(f)); runs = []
    for s in meta["seeds"]:
        z = np.load(RES / folder / f"seed_{s}" / "checkpoints.npz"); runs.append(dict(seed=s, t=z["t"], G=z["G"]))
    return dict(cfg=meta["config"], runs=runs)


def run_metrics(run, cfg):
    every, k, flush = cfg["checkpoint_every"], cfg["analysis"]["topk"], cfg["analysis"]["warmup_flush"]
    da = cfg["stream"].get("drift_at"); dps = [da] if da else []
    seg, clean = segments(run["t"], dps, flush)           # stationary segments only, buffer already flushed
    row, series = {"seed": run["seed"]}, None
    for L in LAGS:
        for name, fn in metric_fns(k).items():
            s = lagged_series(run["G"], fn, L // every, valid=clean, seg=seg)
            row[f"{name}@{L}"] = float(np.nanmean(s)) if np.isfinite(s).any() else np.nan
            if name == "jaccard" and L == PRIMARY_LAG:
                series = s
    row["switch_rate_per_1000"] = topk_switch_rate(run["G"], k, every, clean, seg)
    row["sd_mean"] = float(np.nanmean(per_feature_sd(run["G"], clean, seg)))
    return row, series


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def summarize(c):
    rows, series = zip(*[run_metrics(r, c["cfg"]) for r in c["runs"]])
    every = c["cfg"]["checkpoint_every"]
    out = {"rows": list(rows), "series": list(series), "k": c["cfg"]["analysis"]["topk"], "n": len(rows)}
    for key in [k for k in rows[0] if k != "seed"]:
        v = np.array([r[key] for r in rows], float)
        out[key] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
    p = np.array([r[f"jaccard@{PRIMARY_LAG}"] for r in rows], float)
    if len(rows) > 1:      # between-seed / between-shuffle CI
        out["ci"], out["ci_kind"] = bootstrap_ci(p), "bootstrap over runs"
    else:                  # one deterministic run: block bootstrap over its own checkpoints
        out["ci"] = block_bootstrap_ci(series[0], BLOCK_SAMPLES // every); out["ci_kind"] = f"block bootstrap ({BLOCK_SAMPLES}-sample blocks)"
    out["within_sd"] = float(np.mean([np.nanstd(s, ddof=1) for s in series if np.isfinite(s).sum() > 1]))
    out["between_sd"] = out[f"jaccard@{PRIMARY_LAG}"]["sd"]
    return out


def write_tables(S, sfx=""):
    L = []
    L += ["## SQ1: stability of the global relevance weights G (stationary segments, buffer flushed)", "",
          f"Primary metric: top-k Jaccard at lag {PRIMARY_LAG} samples (no window overlap). Mean ± SD over runs; CI = 95%.", "",
          "| condition | k | runs | **Jaccard@200** | 95% CI | tau-b@200 | weighted tau@200 | Spearman@200 | top-k switches /1000 | mean SD of weights |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, s in S.items():
        g = lambda key, nd=3: f"{fmt(s[key]['mean'], nd)} ± {fmt(s[key]['sd'], nd)}"
        L.append(f"| {lab} | {s['k']} | {s['n']} | **{g('jaccard@200')}** | [{fmt(s['ci'][0])}, {fmt(s['ci'][1])}] ({s['ci_kind']}) | "
                 f"{g('tau_b@200')} | {g('weighted_tau@200')} | {g('spearman@200')} | {g('switch_rate_per_1000', 2)} | {g('sd_mean')} |")
    if "Elec2" in S and "Elec2 shuffled" in S:
        e, sh = S["Elec2"], S["Elec2 shuffled"]
        null = [r[f"jaccard@{PRIMARY_LAG}"] for r in sh["rows"]]
        p = shuffle_rank_pvalue(e[f"jaccard@{PRIMARY_LAG}"]["mean"], null, "less")
        L += ["", "### Primary test: Elec2 against its shuffled distribution", "",
              f"Elec2 Jaccard@200 = {fmt(e['jaccard@200']['mean'])} (95% CI {fmt(e['ci'][0])}-{fmt(e['ci'][1])}, {e['ci_kind']}); "
              f"shuffled mean = {fmt(sh['jaccard@200']['mean'])} ± {fmt(sh['jaccard@200']['sd'])} over {sh['n']} shuffles. "
              f"One-sided exact p (Elec2 lower than shuffled) = {p:.3f}; the smallest possible p with {sh['n']} shuffles is {1 / (sh['n'] + 1):.3f}."]
    if "S3" in S:
        s = S["S3"]
        L += ["", "### Sanity check S3 (stationary synthetic)", "",
              f"Jaccard@200 = {fmt(s['jaccard@200']['mean'])} (expected about 1). Within-run SD over checkpoints = {fmt(s['within_sd'])}; "
              f"between-seed SD of run means = {fmt(s['between_sd'])}."]
    L += ["", "### Effect of window overlap (lags below 200)", "",
          "Mean Jaccard by lag. The model's buffer holds 200 samples, so snapshots `lag` apart share `1 - lag/200` of their data.", "",
          "| condition | " + " | ".join(f"lag {l} ({max(0, 100 - 100 * l // 200)}% overlap)" for l in LAGS) + " |", "|---|" + "---|" * len(LAGS)]
    for lab, s in S.items():
        L.append(f"| {lab} | " + " | ".join(fmt(s[f"jaccard@{l}"]["mean"]) for l in LAGS) + " |")
    (OUT / f"table1_sq1{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / f"sq1_per_run{sfx}.csv", "w", newline="") as f:
        keys = ["condition"] + list(next(iter(S.values()))["rows"][0].keys()); w = csv.DictWriter(f, keys); w.writeheader()
        for lab, s in S.items():
            for r in s["rows"]:
                w.writerow({"condition": lab, **r})


def figure(data, sfx=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for lab, c in data.items():
        cfg = c["cfg"]; every, k, flush = cfg["checkpoint_every"], cfg["analysis"]["topk"], cfg["analysis"]["warmup_flush"]
        curves = [rolling_nanmean(lagged_series(r["G"], metric_fns(k)["jaccard"], PRIMARY_LAG // every, valid=r["t"] >= flush), SMOOTH_CK)
                  for r in c["runs"]]
        M = np.vstack(curves); x = c["runs"][0]["t"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); mean, sd = np.nanmean(M, 0), np.nanstd(M, 0)
        (line,) = ax.plot(x, mean, lw=1.3, ls="--" if lab == "S1" else "-", zorder=3 if lab == "S1" else 2,
                          label=f"{lab} (k={k}, {len(curves)} run{'s' if len(curves) > 1 else ''})")
        if len(curves) > 1:
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.18, color=line.get_color())
        if cfg["stream"].get("drift_at"):
            ax.axvline(cfg["stream"]["drift_at"], color="red", ls="--", lw=0.8)
    ax.set_xlabel("samples"); ax.set_ylabel(f"top-k Jaccard, lag {PRIMARY_LAG}"); ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"Figure 2a: stability of the top-k relevant set (rolling mean over {SMOOTH_CK} checkpoints; band = ±1 SD over runs)", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"fig2a_jaccard_lag200{sfx}.png", dpi=200); fig.savefig(OUT / f"fig2a_jaccard_lag200{sfx}.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="SQ1 analysis")
    ap.add_argument("--suffix", default="", help="e.g. _nodate: use the Elec2 runs made without the `date` feature")
    sfx = ap.parse_args().suffix
    conds = {lab: (folder + sfx if lab.startswith("Elec2") else folder) for lab, folder in CONDITIONS.items()}
    data = {lab: load(folder) for lab, folder in conds.items()}
    missing = [lab for lab, d in data.items() if d is None]
    if missing:
        print("No results yet for:", ", ".join(missing), "(skipped)")
    data = {k: v for k, v in data.items() if v is not None}
    S = {lab: summarize(c) for lab, c in data.items()}
    write_tables(S, sfx); figure(data, sfx)
    print((OUT / f"table1_sq1{sfx}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
