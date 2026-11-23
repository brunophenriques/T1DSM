#!/usr/bin/env python
"""Recompute metrics.json for saved runs from their checkpoints.npz, without re-running the model.

relevance/analysis.analyze is a pure function of the checkpoints and the ground truth, so new metrics
(for example recovery_from_dp/*) can be added to finished runs. The stream is rebuilt from the config
and seed in meta.json to get the ground truth. Every metric that already exists in metrics.json must
come out identical, otherwise the script stops: this guarantees that only new keys are added.

    python -m wf2_drift_validity.reanalyze                       # every run under results_wf2/
    python -m wf2_drift_validity.reanalyze results_wf2/s2_gauss_gradual
"""
import argparse, datetime, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relevance.analysis import analyze
from relevance.streams import build_stream
from run_relevance import aggregate, _git_info

KEEP = ("acc_final", "f1_final", "rules_final", "proc_time_s", "log_time_s")   # not from analyze()


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def reanalyze_seed(seed_dir: Path) -> tuple:
    meta = json.load(open(seed_dir / "meta.json"))
    cfg = json.load(open(ROOT / meta["config_file"]))
    st = build_stream(cfg["stream"], meta["seed"])
    with np.load(seed_dir / "checkpoints.npz") as z:
        arr = {k: z[k] for k in z.files}
    old = json.load(open(seed_dir / "metrics.json"))
    new = analyze(arr, st.truth, st.drift_points, cfg["analysis"], cfg["checkpoint_every"], st.drift_widths)
    keep = lambda k: k in KEEP or k == "perm_time_s" or k.startswith(("valid/P/", "agree/P_vs_G/", "perm/"))
    changed = [k for k in old if not keep(k) and k in new and not _same(old[k], new[k])]
    missing = [k for k in old if not keep(k) and k not in new]
    if changed or missing:
        raise SystemExit(f"{seed_dir}: existing metrics would change {changed} or disappear {missing}")
    added = sorted(k for k in new if k not in old)
    merged = dict(new)
    merged.update({k: v for k, v in old.items() if keep(k)})     # R5 metrics come from permutation.npz
    json.dump(merged, open(seed_dir / "metrics.json", "w"), indent=2)
    return merged, added, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=[str(ROOT / "results_wf2")])
    args = ap.parse_args()
    code = _git_info()                    # before any metrics.json is rewritten
    n_runs, added_all = 0, set()
    for root in (Path(r).resolve() for r in args.roots):
        for summary in sorted(root.rglob("summary.json")):
            vdir = summary.parent
            seed_dirs = sorted((p for p in vdir.glob("seed_*") if (p / "checkpoints.npz").exists()),
                               key=lambda p: int(p.name.split("_")[1]))
            if not seed_dirs:
                continue
            per_seed = []
            for d in seed_dirs:
                m, added, cfg = reanalyze_seed(d)
                per_seed.append(m); added_all.update(added); n_runs += 1
            old = json.load(open(summary))
            old["summary"] = aggregate(per_seed)
            json.dump(old, open(summary, "w"), indent=2)
            print(f"{vdir.relative_to(ROOT)}: {len(seed_dirs)} runs re-analysed")
    note = dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                analysis_code=code, roots=[str(Path(r)) for r in args.roots], runs=n_runs,
                added_keys=sorted(added_all),
                check="every metric already in metrics.json was recomputed and is identical; "
                      "only the keys listed here were added. checkpoints.npz and meta.json are untouched.")
    out = ROOT / "results_wf2" / "reanalysis.json"
    json.dump(note, open(out, "w"), indent=2)
    print(f"{n_runs} runs; added keys: {sorted(added_all)}")


if __name__ == "__main__":
    main()
