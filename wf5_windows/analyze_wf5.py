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
from numbers import Real
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
    base = ROOT / folder
    paths = list(base.glob("seed_*"))
    expected = {0} if name == "Elec2" else set(range(10))
    found = {}
    for sd in paths:
        try:
            seed = int(sd.name.removeprefix("seed_"))
        except ValueError:
            raise ValueError(f"{name} {sd.name}: invalid seed directory") from None
        if seed in found:
            raise ValueError(f"{name} seed {seed}: duplicate seed directories")
        found[seed] = sd
    if set(found) != expected:
        raise ValueError(f"{name}: missing seeds {sorted(expected - set(found))}; "
                         f"unexpected seeds {sorted(set(found) - expected)} in {base}")
    for seed, sd in sorted(found.items()):
        def fail(message):
            raise ValueError(f"{name} seed {seed}: {message}")
        for filename in ("checkpoints.npz", "meta.json"):
            if not (sd / filename).is_file():
                fail(f"missing required file {sd / filename}")
        try:
            meta = json.loads((sd / "meta.json").read_text())
            with np.load(sd / "checkpoints.npz", allow_pickle=False) as archive:
                z = {key: archive[key] for key in archive.files}
            t, G = z["t"], z["G"]
        except (ValueError, OSError, KeyError) as exc:
            fail(f"cannot read required inputs: {exc}")
        if not isinstance(meta, dict):
            fail("meta.json must be a dictionary")
        if not isinstance(meta.get("stream"), dict):
            fail("stream must be a dictionary")
        n_expected, d_expected = (10000, 7) if name == "Elec2" else (20000, 3 if name == "S4" else 10)
        if meta.get("seed") != seed:
            fail("metadata seed disagrees with directory")
        if meta.get("n") != n_expected or meta.get("d") != d_expected:
            fail(f"protocol requires n={n_expected}, d={d_expected}; check metadata")
        for key in ("n", "d"):
            if key in meta.get("stream", {}) and meta["stream"][key] != meta[key]:
                fail(f"stream {key} disagrees with top-level metadata")
        if t.ndim != 1 or G.shape != (len(t), d_expected):
            fail("checkpoint t/G lengths or feature dimensions disagree")
        if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            fail("checkpoint times must be finite and strictly increasing")
        if not np.array_equal(t, np.arange(50, n_expected + 1, 50)):
            fail("checkpoint schedule must cover samples 50..n in steps of 50")
        if "A" in z and z["A"].shape != G.shape:
            fail("checkpoint A feature dimensions disagree with G")
        for key, values in z.items():
            if values.ndim and values.shape[0] != len(t):
                fail(f"checkpoint array {key} has incompatible length")
        st = meta.get("stream", {})
        n = int(meta.get("n", z["t"][-1]))
        # Explicit empty lists identify stationary/no-ground-truth runs. Missing
        # fields must never silently suppress a drift stream's analysis.
        dps = meta.get("drift_points")
        widths = meta.get("drift_widths")
        def finite_number(value):
            return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
        if not isinstance(dps, list) or not all(finite_number(dp) for dp in dps):
            fail("drift_points must be an explicit list of finite numbers")
        if any(not 0 < dp < n for dp in dps) or any(b <= a for a, b in zip(dps, dps[1:])):
            fail("drift_points must be strictly ordered and inside the stream (0, n)")
        if name in ("S1", "S2", "S4") and not dps:
            fail("drift_points must not be empty for a drift stream")
        if name in ("S3", "Elec2") and dps:
            fail("drift_points must be empty for stationary S3 / no-ground-truth Elec2")
        if (not isinstance(widths, list) or len(widths) != len(dps)
                or not all(finite_number(w) and w >= 0 for w in widths)):
            fail("drift_widths must be an explicit list of finite nonnegative numbers matching drift_points")
        if any(dp - w / 2 < 0 or dp + w / 2 > n for dp, w in zip(dps, widths)):
            fail("drift_widths place a transition outside the stream")
        if "drift_at" in st:
            drift_at = st["drift_at"]
            if drift_at is not None and not finite_number(drift_at):
                fail("stream.drift_at must be a finite number or null")
            if dps != ([] if drift_at is None else [drift_at]):
                fail("drift_points disagree with stream.drift_at")
        if "drift_width" in st:
            width = st["drift_width"]
            if not finite_number(width) or width < 0:
                fail("stream.drift_width must be finite and nonnegative")
            if any(w != width for w in widths):
                fail("drift_widths disagree with stream.drift_width")
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


def check_variants_identical(names=None):
    for name, (folder, _, _) in STREAMS.items():
        if (names is not None and name not in names) or not folder.endswith("/ref"):
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
              "| Stream | Threshold (wτ) | Dropped | Recovered after drop | Right-censored | No destabilisation | Undefined | Descriptive time from transition end | Descriptive time from drift point | Observed recovery from drift point | WF2 ranking recovery (from dp) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, g in restab.groupby("stream", sort=False):
            L.append(f"| {s} | {fmt(g.threshold)} | {int(g.dropped.sum())}/{len(g)} | {int(g.restabilised.sum())}/{len(g)} | "
                     f"{int((g.restab_status == 'censored').sum())}/{len(g)} | "
                     f"{int((g.restab_status == 'no_destabilisation').sum())}/{len(g)} | "
                     f"{int((g.restab_status == 'undefined').sum())}/{len(g)} | "
                     f"{fmt(g.restab, 0)} | {fmt(g.restab_from_dp, 0)} | "
                     f"{fmt(g.loc[g.restab_from_dp_status == 'recovered', 'restab_from_dp'], 0)} | "
                     f"{fmt(g.get('wf2_ranking_recovery_from_dp', []), 0)} |")
        L += ["", "Descriptive times include zero-time no-destabilisation cases; these are not observed recoveries. "
              "Observed recovery means include recovered cases only. Status counts use the transition-end search; "
              "midpoint statuses are recorded separately in the CSV. Midpoint offsets may be negative. "
              "Paired time tests retain the descriptive zero-time convention. "
              "Censored and undefined runs are NaN in means; censoring limits are separately recorded in "
              "`restab_per_run.csv` (`restab_censor_limit` and `restab_from_dp_censor_limit`)."]
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
    # Validate every requested stream before writing any summaries.
    for name in a.streams:
        load_runs(name)
    check_variants_identical(a.streams)
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
