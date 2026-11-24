#!/usr/bin/env python
"""SQ3 validity (PROTOCOL.md §1, §5, §12): do the relevance scores rank the truly relevant features first?

    python -m wf2_drift_validity.sq3_validity

Primary: top-k precision of R1 after warm-up in the clean segments of S1 (both halves), per seed.
Secondary: ROC-AUC; S2; S5 (Agrawal F0 -> F6, frozen rule) and the F2 -> F6 sensitivity pair; the F0
failure case; S6 (does R1 mark a non-causal copy of x0 as relevant?); R4; R5 permutation importance.
Writes results_wf2/sq3/: descriptive.csv, comparisons.csv, s6_copy.csv, sq3_primary.json.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.metrics import topk_idx
from wf2_drift_validity.statistical_analysis import describe, paired_comparison, holm_correction, write_csv

RES = ROOT / "results_wf2"
SEEDS = range(10)


def metric(path, key):
    vals = []
    for s in SEEDS:
        with open(path / f"seed_{s}" / "metrics.json") as f:
            vals.append(json.load(f).get(key, np.nan))
    return np.asarray(vals, dtype=float)


def exists(path):
    return all((path / f"seed_{s}" / "metrics.json").exists() for s in SEEDS)


# (label, results folder, segment labels)
STREAMS = [
    ("S1", RES / "s1_gauss_abrupt", ("first half {x0,x1,x2}", "second half {x5,x6,x7}")),
    ("S2", RES / "s2_gauss_gradual", ("before {x0,x1,x2}", "after {x5,x6,x7}")),
    ("S5 F0->F6", RES / "s5_agrawal_pair", ("F0 {age}", "F6 {salary,commission,loan}")),
    ("S5b F2->F6 (sensitivity)", RES / "s5b_agrawal_f2_f6", ("F2 {age,elevel}", "F6 {salary,commission,loan}")),
    ("F0 stationary", RES / "s5f0_agrawal_f0", ("F0 {age}",)),
    ("S6 copy", RES / "s6_gauss_copy", ("first half {x0,x1,x2}; x10 = noisy x0", "second half {x5,x6,x7}")),
    ("S4 SEA", RES / "s4_sea_drift", ("theta 8 {x0,x1}", "theta 7 {x0,x1}")),
]


def descriptive():
    rows = []
    for label, base, segs in STREAMS:
        for variant in ("ref", "open"):
            path = base / variant
            if not exists(path):
                continue
            for name, rel in (("G", "R1"), ("A", "R4")):
                for s, seg in enumerate(segs):
                    for kind in ("prec_at_k", "auc"):
                        v = metric(path, f"valid/{name}/{kind}/seg{s}")
                        rows.append(dict(stream=label, variant=variant, relevance=rel, metric=kind, segment=seg,
                                         seeds_eq_1=int(np.sum(v == 1.0)), **describe(v)))
    for label, sub in (("S1", "s1_gauss_abrupt"), ("S5 F0->F6", "s5_agrawal_pair")):
        for variant in ("ref", "open"):
            path = RES / "r5" / sub / variant
            if not exists(path):
                continue
            for s, seg in enumerate(SEGMENTS[label]):
                for kind, key in (("prec_at_k", f"valid/P/prec_at_k/seg{s}"), ("auc", f"valid/P/auc/seg{s}"),
                                  ("overlap_with_R1", f"agree/P_vs_G/topk_overlap/seg{s}"),
                                  ("base_acc", f"perm/base_acc/seg{s}")):
                    v = metric(path, key)
                    rows.append(dict(stream=label, variant=variant, relevance="R5", metric=kind, segment=seg,
                                     seeds_eq_1=int(np.sum(v == 1.0)), **describe(v)))
    return rows


SEGMENTS = {label: segs for label, _, segs in STREAMS}


def s6_copy_rows():
    """How often R1 puts the copy x10 in its top 3, per seed, in the clean part of each half of S6."""
    rows = []
    path = RES / "s6_gauss_copy" / "ref"
    if not exists(path):
        return rows
    for s in SEEDS:
        meta = json.load(open(path / f"seed_{s}" / "meta.json"))
        cfg = json.load(open(ROOT / meta["config_file"]))
        dp, flush, k = meta["drift_points"][0], cfg["analysis"]["warmup_flush"], int(cfg["analysis"]["topk"])
        col, src = meta["stream"]["copy"]["column"], meta["stream"]["copy"]["source"]
        with np.load(path / f"seed_{s}" / "checkpoints.npz") as z:
            t, G = z["t"], z["G"]
        # clean checkpoints, as in relevance.analysis: the 200-sample buffer holds one concept only
        for seg, mask in (("first half", (t >= flush) & (t <= dp)), ("second half", t >= dp + flush)):
            Gs = G[mask]
            in_top = np.array([col in topk_idx(g, k) for g in Gs])
            rank = np.array([int(np.where(np.argsort(-g, kind="stable") == col)[0][0]) + 1 for g in Gs])
            rows.append(dict(seed=s, segment=seg, copy_in_top3_share=float(in_top.mean()),
                             copy_mean_rank=float(rank.mean()), copy_weight_mean=float(Gs[:, col].mean()),
                             x0_weight_mean=float(Gs[:, src].mean())))
    return rows


def comparisons():
    rows = []

    def add(name, a, b, meaning):
        rows.append(dict(family="SQ3 (secondary)", comparison=name, meaning=meaning, **paired_comparison(a, b)))

    s1, s6 = RES / "s1_gauss_abrupt" / "ref", RES / "s6_gauss_copy" / "ref"
    if exists(s1) and exists(s6):
        add("S6 vs S1: R1 top-3 precision, first half (same streams plus the copy)",
            metric(s6, "valid/G/prec_at_k/seg0"), metric(s1, "valid/G/prec_at_k/seg0"),
            "negative = the copy pushes a relevant feature out of the top 3")
    r5 = RES / "r5" / "s1_gauss_abrupt" / "open"
    if exists(r5):
        both = lambda p, n: (metric(p, f"valid/{n}/prec_at_k/seg0") + metric(p, f"valid/{n}/prec_at_k/seg1")) / 2
        add("S1 open: R5 vs R1 top-3 precision (mean of both halves)", both(r5, "P"), both(r5, "G"),
            "negative = permutation importance finds the relevant features less often than R1")
    if rows:
        for r, p in zip(rows, holm_correction([r["wilcoxon_p"] for r in rows])):
            r["holm_p"] = float(p)
            r["holm_family_size"] = len(rows)
    return rows


def main():
    out = RES / "sq3"
    out.mkdir(parents=True, exist_ok=True)
    desc = descriptive()
    write_csv(out / "descriptive.csv", desc)
    s6 = s6_copy_rows()
    if s6:
        write_csv(out / "s6_copy.csv", s6)
    comps = comparisons()
    if comps:
        write_csv(out / "comparisons.csv", comps)

    s1 = RES / "s1_gauss_abrupt"
    halves = np.column_stack([metric(s1 / "ref", "valid/G/prec_at_k/seg0"), metric(s1 / "ref", "valid/G/prec_at_k/seg1")])
    same = all(np.array_equal(metric(s1 / "ref", f"valid/G/prec_at_k/seg{i}"),
                              metric(s1 / "open", f"valid/G/prec_at_k/seg{i}")) for i in (0, 1))
    per_seed = halves.mean(axis=1)
    primary = dict(metric="Top-3 precision of R1 in the clean segments of S1 (mean of both halves, per seed)",
                   expected=1.0, values=per_seed.tolist(), identical_for_ref_and_open=bool(same),
                   seeds_eq_1_both_halves=int(np.sum(np.all(halves == 1.0, axis=1))),
                   first_half=describe(halves[:, 0]), second_half=describe(halves[:, 1]), **describe(per_seed),
                   auc_both_halves=describe((metric(s1 / "ref", "valid/G/auc/seg0") + metric(s1 / "ref", "valid/G/auc/seg1")) / 2))
    json.dump(primary, open(out / "sq3_primary.json", "w"), indent=2)

    print(f"Primary SQ3: {primary['mean']:.3f} ± {primary['sd']:.3f}, 95% CI [{primary['ci_low']:.3f}, "
          f"{primary['ci_high']:.3f}], seeds with precision 1 in both halves: {primary['seeds_eq_1_both_halves']}/10")
    for r in desc:
        if r["metric"] in ("prec_at_k", "auc", "overlap_with_R1") and (r["relevance"] != "R4" or r["stream"] in ("S1", "S5 F0->F6")):
            print(f"{r['stream']:26s} {r['variant']:4s} {r['relevance']} {r['metric']:13s} {r['segment']:42s} "
                  f"{r['mean']:.3f} ± {r['sd']:.3f}  (=1 in {r['seeds_eq_1']}/10)")
    if s6:
        for seg in ("first half", "second half"):
            v = [r["copy_in_top3_share"] for r in s6 if r["segment"] == seg]
            print(f"S6 copy x10 in R1 top-3, {seg}: {np.mean(v):.3f} of clean checkpoints (seeds {min(v):.2f}-{max(v):.2f})")
    for r in comps:
        print(f"[{r['family']}] {r['comparison']} | p={r['wilcoxon_p']:.4f} Holm={r['holm_p']:.4f} "
              f"r_rb={r['rank_biserial']:+.2f} nonzero={r['n_nonzero']}")
    print("Results saved to:", out)


if __name__ == "__main__":
    main()
