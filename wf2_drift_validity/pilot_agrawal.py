#!/usr/bin/env python
"""Learnability rule (PROTOCOL.md §7) applied to the Agrawal functions used in S5.

Same rule and constants as experiments/pilot_learnability.py, fixed before the S5 runs:
  * pilot seed 99 only, accuracy only (relevance is not looked at);
  * each function on its stationary version, n = 20,000;
  * "predictive" if the `open` variant beats the majority-class rate by >= 5 percentage points.
A drift pair is predictive only if both of its functions are.

    python -m wf2_drift_validity.pilot_agrawal   -> results_wf2/pilot_agrawal.csv, pilot_agrawal.json
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pilot_learnability import PILOT_SEED, N, MARGIN, VARIANTS, CFG
from relevance.streams import agrawal_stream
from run_relevance import run_one, _git_info, _versions

FUNCTIONS = (0, 2, 6)          # S5 = F0 -> F6 (frozen rule), sensitivity pair F2 -> F6, F0 failure case


def one(f, vname):
    st = agrawal_stream(n=N, function_before=f, seed=PILOT_SEED)
    majority = float(max(st.y.mean(), 1 - st.y.mean()))
    t0 = time.perf_counter()
    res = run_one(CFG, PILOT_SEED, log=False, stream=st, model_cfg=VARIANTS[vname])
    return dict(function=f, variant=vname, acc=res["acc_final"], majority=majority,
                margin_vs_majority=res["acc_final"] - majority, rules_final=int(res["n_rules"][-1]),
                seconds=round(time.perf_counter() - t0, 1))


def main():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one, *zip(*[(f, v) for f in FUNCTIONS for v in VARIANTS])))
    out = ROOT / "results_wf2"
    out.mkdir(exist_ok=True)
    with open(out / "pilot_agrawal.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    passes = {r["function"]: r["margin_vs_majority"] >= MARGIN for r in rows if r["variant"] == "open"}
    roles = {f"F{f}": "predictive" if passes[f] else "relevance-only" for f in FUNCTIONS}
    pairs = {"S5 F0->F6": "predictive" if passes[0] and passes[6] else "relevance-only",
             "S5b F2->F6 (sensitivity)": "predictive" if passes[2] and passes[6] else "relevance-only"}
    json.dump(dict(rule="PROTOCOL.md §7 learnability rule (seed 99, open beats majority by >= 5 pp)",
                   pilot_seed=PILOT_SEED, n=N, margin=MARGIN, rows=rows, function_roles=roles, pair_roles=pairs,
                   git=_git_info(), versions=_versions()),
              open(out / "pilot_agrawal.json", "w"), indent=2)
    for r in rows:
        print(f"F{r['function']} {r['variant']:4s} acc={r['acc']:.4f} majority={r['majority']:.4f} "
              f"margin={r['margin_vs_majority']:+.4f} rules={r['rules_final']} ({r['seconds']}s)")
    print(json.dumps(dict(function_roles=roles, pair_roles=pairs), indent=2))


if __name__ == "__main__":
    main()
