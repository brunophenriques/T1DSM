#!/usr/bin/env python
"""Oracle check (PROTOCOL.md §6 and §12): can R1's statistic see each concept's relevant features at all?

For every concept, R1's formula (the model's own FeatureSeparabilityBuffer.compute_feature_weights:
between/within-class variance ratio, floored at 0.2) is computed on 100,000 stationary samples and
compared with the ground truth (top-k precision with k = number of relevant features, and AUC).
It separates two failure modes:
  * the statistic cannot see the feature (oracle precision < 1 even on 100,000 samples);
  * the 200-sample estimate is too noisy (oracle precision 1, but low precision on 200-sample windows).
The second part therefore also scores the 500 non-overlapping 200-sample windows of the same data.

Concepts: both halves of S1, both SEA concepts of S4, all 10 Agrawal functions (and S6's first concept,
for information only). Seed 99, the pilot seed that is never used in the experiments.

Agrawal selection rule (fixed in PROTOCOL.md §6 before this ran): choose the two functions whose relevant
sets are detected with top-k precision 1 and are disjoint; ties go to the lowest function index.

    python -m wf2_drift_validity.oracle_check   -> results_wf2/oracle/oracle.csv, selection.json
"""
import csv, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ENF_UnimNeuron_Safe import FeatureSeparabilityBuffer
from relevance.metrics import topk_precision, auc
from relevance.streams import (gauss_shift_stream, sea_stream, agrawal_stream, add_noisy_copy,
                               AGRAWAL_FEATURES, AGRAWAL_RELEVANT)
from run_relevance import _git_info, _versions

ORACLE_SEED = 99
N = 100_000
WINDOW = 200


def r1_weights(X, y, window=None):
    """R1's formula, using the model's own class (exactly what the model computes)."""
    buf = FeatureSeparabilityBuffer(n_features=X.shape[1], buffer_size=window or len(y))
    for x_t, y_t in zip(X, y):
        buf.add(x_t, int(y_t))
    return buf.compute_feature_weights()


def concepts():
    s1 = dict(n=N, d=10, drift_at=None, delta=0.3, sd=0.12, seed=ORACLE_SEED)
    yield "S1 first half", "gauss", None, gauss_shift_stream(rel_before=(0, 1, 2), **s1)
    yield "S1 second half", "gauss", None, gauss_shift_stream(rel_before=(5, 6, 7), **s1)
    yield "SEA theta 8 (S4 before)", "sea", None, sea_stream(n=N, drift_at=None, variant_before=0, seed=ORACLE_SEED)
    yield "SEA theta 7 (S4 after)", "sea", None, sea_stream(n=N, drift_at=None, variant_before=2, seed=ORACLE_SEED)
    for f in range(10):
        yield f"Agrawal F{f}", "agrawal", f, agrawal_stream(n=N, function_before=f, seed=ORACLE_SEED)
    yield ("S6 first half (info only)", "gauss+copy", None,
           add_noisy_copy(gauss_shift_stream(rel_before=(0, 1, 2), **s1), source=0, noise_sd=0.02, seed=ORACLE_SEED))


def check(st):
    truth = st.truth[0].astype(int)
    k = int(truth.sum())
    w = r1_weights(st.X, st.y)
    windows = [r1_weights(st.X[i:i + WINDOW], st.y[i:i + WINDOW], WINDOW) for i in range(0, len(st.y), WINDOW)]
    wprec = np.array([topk_precision(v, truth, k) for v in windows])
    names = st.feature_names
    order = np.argsort(-w, kind="stable")
    return dict(k=k, relevant=[names[j] for j in np.where(truth)[0]], class1_share=float(st.y.mean()),
                oracle_prec=topk_precision(w, truth, k), oracle_auc=auc(w, truth),
                window_prec_mean=float(wprec.mean()), window_prec_eq1=float(np.mean(wprec == 1.0)),
                n_windows=len(windows), ranking=[names[j] for j in order],
                weights={names[j]: round(float(w[j]), 4) for j in range(len(w))})


def select_pair(results: dict) -> dict:
    """The frozen rule: both detected (oracle precision 1) and disjoint; lowest function indices win."""
    detected = [f for f in range(10) if results.get(f) == 1.0]
    valid = [(i, j) for i, j in combinations(detected, 2)
             if not set(AGRAWAL_RELEVANT[i]) & set(AGRAWAL_RELEVANT[j])]
    return dict(detected=detected, valid_pairs=[list(p) for p in valid],
                chosen=list(valid[0]) if valid else None)   # combinations() is lexicographic: lowest first


def main():
    out = ROOT / "results_wf2" / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    rows, agrawal_prec = [], {}
    for name, family, f, st in concepts():
        r = check(st)
        rows.append(dict(concept=name, family=family, function=f, **{k: v for k, v in r.items() if k != "weights"},
                         weights=json.dumps(r["weights"])))
        if family == "agrawal":
            agrawal_prec[f] = r["oracle_prec"]
        print(f"{name:28s} k={r['k']} oracle P@k={r['oracle_prec']:.3f} AUC={r['oracle_auc']:.3f} "
              f"200-window P@k mean={r['window_prec_mean']:.3f} (=1 in {100 * r['window_prec_eq1']:.0f}%) "
              f"class1={r['class1_share']:.3f} top={r['ranking'][:r['k'] + 1]}", flush=True)
    for r in rows:
        r["relevant"] = " ".join(r["relevant"]); r["ranking"] = " ".join(r["ranking"])
    with open(out / "oracle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sel = select_pair(agrawal_prec)
    sel.update(rule="both relevant sets detected with top-k precision 1 on 100,000 samples, disjoint; "
                    "ties go to the lowest function index", relevant_sets={f: list(AGRAWAL_RELEVANT[f]) for f in range(10)},
               oracle_prec={f: agrawal_prec[f] for f in range(10)}, seed=ORACLE_SEED, n=N,
               f0_failure_case=dict(oracle_prec=agrawal_prec[0],
                                    note="F0 labels by age < 40 or age >= 60: both classes have about the same mean "
                                         "age, so a class-mean (between/within variance) statistic cannot see it."),
               git=_git_info(), versions=_versions())
    json.dump(sel, open(out / "selection.json", "w"), indent=2)
    print("detected:", sel["detected"], "valid pairs:", sel["valid_pairs"], "chosen:", sel["chosen"])


if __name__ == "__main__":
    main()
