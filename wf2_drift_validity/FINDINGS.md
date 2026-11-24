# WF2 — Drift Sensitivity Findings (SQ2)

Author: Bruno Henriques. Revised by Luis Soeiro (23 Nov 2026) after the PR #5 review; the changes
are listed in [Revision notes](#revision-notes).
Research question: SQ2 — How quickly does feature relevance recover after drift?

All results here are **exploratory** until the team tags `protocol-freeze` (PROTOCOL.md §13).

## 1. Implementation

- Ranking recovery and accuracy recovery, with right-censoring for runs that never recover
  (`relevance/recovery.py`; `wf2_drift_validity/recovery_metrics.py` re-exports it).
- Recovery requires three consecutive qualifying checkpoints.
- `recovery_from_dp/*`: the same ranking recovery timed from the drift point instead of the end of
  the transition, so that abrupt and gradual drifts are compared from the same moment.
- Unit tests on step curves, including runs that never recover, NaN values, empty segments and a
  gradual drift (`tests/test_recovery_metrics.py`, runnable as a script or with pytest).
- The new metrics were added to the saved runs with `python -m wf2_drift_validity.reanalyze`, which
  recomputes every metric from `checkpoints.npz` and stops if any existing value would change
  (`results_wf2/reanalysis.json`).

## 2. ADWIN — Elec2

Configuration: 10,000 samples, variant `ref`, δ = 0.002. Accuracy 58.86%.

Detected change points: 1152, 1664, 2304, 2944, 3616, 4512, 5536, 6528.

- These are descriptive only and are not ground-truth drift.
- river's ADWIN checks for change every 32 samples (`clock=32`), so every value is a multiple of 32:
  they are **detection times**, not estimates of when the change happened.
- `ref` predicts at chance on the synthetic streams (PROTOCOL.md §7). On Elec2 its accuracy has not
  been compared with the majority-class rate yet (WF4), so the detections may partly reflect changes
  in the class balance rather than in the concept.

## 3. S1 — Preliminary Experiment (Seed 0)

| Metric | Ref | Open |
|---|---:|---:|
| R1 ranking recovery | 150 | 150 |
| R4 ranking recovery | Not recovered | Not recovered |
| Accuracy recovery | not interpretable (chance level) | Not recovered |
| Final accuracy | 0.4959 | 0.5705 |
| R1 post-drift Precision@3 | 1.00 | 1.00 |
| R4 post-drift Precision@3 | 0.30 | 0.00 |
| Final rule count | 5 | 106 |

## 4. S1 — Abrupt drift (10 seeds)

Abrupt Gaussian feature-relevance drift at sample 10,000; seeds 0–9; `ref` and `open`.
Mean ± SD use the sample SD (ddof = 1). Recovery values include censored runs at their censoring
time, with the recovered count next to them.

| Metric | Ref | Open |
|---|---:|---:|
| R1 recovery time | 145 ± 15.8 (10/10 recovered) | 145 ± 15.8 (10/10) |
| R4 recovery | 0/10 recovered (censored at 10,000) | 0/10 recovered |
| Accuracy recovery | N/A: `ref` predicts at chance | 3/10 recovered, 7/10 censored; median 10,000 |
| Accuracy drop after the drift | — | 0.127 ± 0.025 |
| Final accuracy | 0.504 ± 0.006 | 0.582 ± 0.008 |
| R1 post-drift Precision@3 | 1.00 | 1.00 |
| R4 post-drift Precision@3 | 0.18 | 0.00 |
| Final rule count | 5 | 97.7 ± 8.0 |

### Findings

- R1 recovered within the predicted interval of 100–250 samples in every seed.
- R1 is the same for `ref` and `open`: it depends only on the data and the window.
- R4 never recovered. In `ref` the rule weights are frozen; in `open` the rules created before the
  drift keep the old weights and outweigh the newer ones, so R4 never ranks the new features in its
  top 3 (Precision@3 = 0.00 after the drift).
- `open`'s accuracy does **not** return to its pre-drift level. Before the drift it is about 0.62
  (0.60–0.67 by seed); over the last 2,000 samples it is about 0.57 (0.56–0.59). The three runs
  counted as recovered (at 8,850, 9,450 and 9,600 samples) only crossed the threshold
  (pre-drift − 0.02) briefly near the end of the stream.
- R4 is a case of a ranking that is stable but not valid: after the drift its top-3 set does not
  change, yet none of the new features is in it.

## 5. S2 — Gradual drift (10 seeds)

Gradual Gaussian feature-relevance drift with a transition from 9,500 to 10,500.

| Metric | Ref | Open |
|---|---:|---:|
| R1 recovery, protocol (from the transition end, 10,500) | 50 ± 0 (10/10 at the floor) | 50 ± 0 |
| R1 recovery from the drift point (10,000) | 190 ± 31.6 (10/10) | 190 ± 31.6 |
| R4 recovery | 0/10 recovered | 0/10 recovered |
| Accuracy recovery | N/A: chance level | 3/10 recovered, 7/10 censored; median 9,500 |
| Final accuracy | 0.504 ± 0.006 | 0.578 ± 0.008 |
| Final rule count | 5 | 93.4 ± 13.2 |

### Findings

- The protocol value (50 in every seed) is a **floor**: 50 is the first checkpoint after the
  transition ends, and R1 had already re-oriented before that. It shows that R1 is correct once the
  transition is over, but it cannot measure speed.
- Timed from the drift point, as for S1, R1 needs about 190 samples on S2 against 145 on S1:
  gradual drift is **slower** (see §7). In a gradual drift the 200-sample window fills with
  new-concept samples more slowly, because the samples only gradually come from the new concept.
- Measured from the start of the transition (9,500), R1 needs about 690 samples.
- As on S1, R4 never recovers and `open`'s accuracy does not return to its pre-drift level.

## 6. S4 — Negative control (10 seeds)

SEA threshold drift at sample 10,000. The relevant set stays {x0, x1}.

| Metric | Ref | Open |
|---|---:|---:|
| R1 pre-drift Precision@2 | 1.00 | 1.00 |
| R1 post-drift Precision@2 | 1.00 | 1.00 |
| R4 pre-drift Precision@2 | 0.767 | 1.00 |
| R4 post-drift Precision@2 | 0.767 | 1.00 |
| R1 Jaccard (lag 200) | 1.00 | 1.00 |
| Final accuracy | 0.718 ± 0.003 | 0.718 ± 0.003 |
| Macro-F1 | 0.418 | 0.418 |
| Final rule count | 5 | 147.1 ± 14.3 |

### Findings

- R1 identified the relevant features before and after the threshold drift.
- R4 stayed valid in `open`, unlike on S1 and S2, where the relevant set changes.
- **Both variants almost always predict the majority class**: predicting only the majority class
  gives macro-F1 ≈ 0.42 at this class balance, which is what both reach. S4 is relevance-only
  (learnability pilot), so no accuracy claims are made on it.
- Ranking recovery on S4 is trivial (the relevant set does not change) and is not interpreted. The
  pre-drift versus final window comparison is WF5's.

## 7. Statistics

10 paired seeds; Wilcoxon signed-rank (zero differences dropped), matched-pairs rank-biserial
correlation and a sign test; Holm correction within each family; bootstrap CIs with 10,000
resamples. Output in `results_wf2/statistics/`.

### Primary SQ2 result (H2, first part)

R1 ranking recovery on S1: **145 ± 15.8 samples, 95% CI [135, 150], 10/10 recovered.**
The CI lies inside the predicted 100–250 samples.

### Window size (H2, second part)

R1 ranking recovery on S1 for different R1 window sizes (`configs/wf2/s1_window_study.json`; the
window is the model's `buffer_size_separability`, 200 in the baseline):

| Window | 50 | 100 | 200 | 500 | 1,000 |
|---|---:|---:|---:|---:|---:|
| Recovery (mean ± SD) | 50 ± 0 | 90 ± 21.1 | 145 ± 15.8 | 295 ± 15.8 | 545 ± 15.8 |
| Recovered | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

Recovery grows with the window in every seed (10/10 monotone; Page's trend test p = 6.6 × 10⁻²⁰).
The medians are 50 / 100 / 150 / 300 / 550: half the window plus one refresh interval (50 samples).
The new features win as soon as about half of the window holds new data, and R1 is only recomputed
every 50 samples. This is the mechanism the H2 prediction was derived from. At window 50 the value is
the one-checkpoint floor.

### Paired comparisons

| Family | Comparison | Wilcoxon p | Holm p | Rank-biserial |
|---|---|---:|---:|---:|
| SQ2 | S1 open: R1 vs accuracy recovery | 0.0020 | 0.0059 | −1.0 |
| SQ2 | S2 open: R1 vs accuracy recovery | 0.0020 | 0.0059 | −1.0 |
| SQ2 | S1 vs S2: R1 recovery from the drift point | 0.0156 | 0.0156 | −1.0 |
| H6 (secondary) | S1 ref: R1 vs R4 | 0.0020 | 0.0078 | −1.0 |
| H6 (secondary) | S1 open: R1 vs R4 | 0.0020 | 0.0078 | −1.0 |
| H6 (secondary) | S2 ref: R1 vs R4 | 0.0020 | 0.0078 | −1.0 |
| H6 (secondary) | S2 open: R1 vs R4 | 0.0020 | 0.0078 | −1.0 |

- The ranking recovers before accuracy in every seed, on both streams.
- Abrupt drift recovers sooner than gradual drift when both are timed from the drift point (7 seeds
  differ, all in the same direction; 3 are tied).
- R1 vs R4 only restates that R1 recovered in 10/10 seeds and R4 in 0/10 (R4's value is the
  censoring time). These belong to H6 (WF3) and are reported as secondary.

### Limitations

- Censored values are lower bounds, not recovery times; recovered counts are always reported.
- With 10 seeds the smallest two-sided Wilcoxon p is 0.002.
- Results are exploratory until `protocol-freeze`.

## 8. Implementation Risks and Mitigation

### Risk 1 — Misleading recovery measurements

A brief improvement may be read as recovery, unrecovered runs may bias averages, and the time
origin can create a floor (as on S2).

Mitigation: three consecutive checkpoints; censored values with recovered counts; recovery timed
from the drift point for abrupt-vs-gradual comparisons; step-curve unit tests.

### Risk 2 — Confounding relevance and predictive adaptation

Fast R1 recovery does not mean the classifier adapted.

Mitigation: R1, R4 and accuracy evaluated separately; `ref` and `open` compared; rule counts
monitored; accuracy claims only on predictive streams and only for `open`; S4 as a negative control.

## 9. Table 1 — SQ2 row

| Research question | Primary metric | Result | 95% CI | Recovered |
|---|---|---|---|---|
| SQ2 — Drift sensitivity | R1 ranking recovery on S1 (window 200) | 145 ± 15.8 samples | [135, 150] | 10/10 |

Supporting lines for the report:
- Window size: 50 / 90 / 145 / 295 / 545 samples for windows 50 / 100 / 200 / 500 / 1,000
  (monotone in 10/10 seeds).
- Gradual drift (S2), timed from the drift point: 190 ± 31.6 samples, slower than abrupt
  (Holm p = 0.016).

## 10. Final Interpretation

- Global relevance (R1) re-orients to the new relevant features within 145 ± 15.8 samples after an
  abrupt drift, inside the predicted 100–250 samples, and the delay grows with the window size, as
  H2 predicts.
- Under gradual drift R1 re-orients before the transition ends, but later than under abrupt drift
  when both are timed from the drift point (about 190 against 145 samples).
- The rule weights the model actually uses (R4) never re-orient, in either variant.
- `open`'s accuracy drops by about 0.13 after the drift and does not return to its pre-drift level
  within 10,000 samples.
- Relevance adaptation, rule-weight adaptation and predictive recovery are therefore distinct: a
  correct R1 does not mean the model adapted.

## Revision notes

Changes made after the PR #5 review (Luis Soeiro, 23 Nov 2026):
1. S2: the protocol value (50) is reported as a floor; abrupt vs gradual is compared from the drift
   point (new metric `recovery_from_dp/*`). The earlier conclusion that gradual drift recovers
   faster is withdrawn.
2. Accuracy recovery: `open` is reported as 3/10 recovered, 7/10 censored (median of the censored
   values) instead of the mean over the 3 recovered runs (9,300 ± 324); `ref` is N/A.
3. H2: the window-size part was run on S1 (§7), so both parts of H2 are now tested.
4. Sample SD (ddof = 1) throughout; Holm within families; R1 vs R4 moved to a secondary H6 family.
5. S4 majority-class behaviour and ADWIN detection times stated explicitly.
