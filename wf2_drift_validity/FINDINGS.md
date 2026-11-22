# WF2 — Drift Sensitivity Findings

Author: Bruno Henriques
Research question: SQ2 — How quickly does feature relevance recover after drift?

## 1. Implementation

- Implemented ranking recovery and accuracy recovery.
- Recovery requires three consecutive qualifying checkpoints.
- Implemented right-censoring for runs that never recover.
- All 8 unit tests passed.
- Regression, non-intrusive and no-leakage tests passed.

## 2. ADWIN — Elec2

Configuration:
- Samples: 10,000
- Variant: ref
- Delta: 0.002
- Accuracy: 58.86%

Detected change points:
1152, 1664, 2304, 2944, 3616, 4512, 5536, 6528.

These detections are descriptive and do not represent
ground-truth concept drift.

## 3. S1 — Preliminary Experiment (Seed 0)

| Metric | Ref | Open |
|---|---:|---:|
| R1 ranking recovery | 150 | 150 |
| R4 ranking recovery | Not recovered | Not recovered |
| Accuracy recovery | 950 | Not recovered |
| Final accuracy | 0.4959 | 0.5705 |
| R1 post-drift Precision@3 | 1.00 | 1.00 |
| R4 post-drift Precision@3 | 0.30 | 0.00 |
| Final rule count | 5 | 106 |

### Preliminary findings

- Global relevance recovered within the predicted
  interval of 100–250 samples, consistent with H2.
- Effective rule weights failed to recover after drift.
- Removing the rule cap improved predictive accuracy
  but did not guarantee accuracy recovery.
- Feature relevance recovery does not necessarily
  imply predictive performance recovery.

These observations are preliminary and based on one seed.

## 4. S1 — Full Experiment (10 seeds)

Abrupt Gaussian feature-relevance drift at sample 10,000.
Both ref and open configurations were evaluated using seeds 0–9.

| Metric | Ref | Open |
|---|---:|---:|
| R1 recovery time | 145 ± 15 | 145 ± 15 |
| R1 recovered runs | 10/10 | 10/10 |
| R4 recovered runs | 0/10 | 0/10 |
| Accuracy recovery | 845 ± 799.52 | 9300 ± 324.04 |
| Accuracy recovered runs | 10/10 | 3/10 |
| Accuracy censored runs | 0/10 | 7/10 |
| Final accuracy | 0.5043 ± 0.0057 | 0.5819 ± 0.0080 |
| R1 post-drift Precision@3 | 1.00 | 1.00 |
| R4 post-drift Precision@3 | 0.18 | 0.00 |
| Final rule count | 5 | 97.7 ± 7.59 |

Reported standard deviations use ddof=0.
Accuracy recovery means include recovered runs only.

### Findings

- R1 recovered within the predicted interval of 100–250
  samples, supporting H2.
- Neither configuration achieved R4 ranking recovery.
- Global feature relevance recovery was independent of
  the rule cap in this experiment.
- The open configuration achieved higher final accuracy
  but exhibited slower and less frequent accuracy recovery.
- Stable rankings do not necessarily imply valid rankings.
- Feature relevance recovery and predictive recovery
  represent distinct adaptation mechanisms.

### Limitations

The ref configuration remains near chance-level accuracy.
Its faster accuracy recovery should therefore not be
interpreted as superior predictive adaptation.

## 5. S2 — Gradual Drift (10 seeds)

Gradual Gaussian feature-relevance drift with a transition
between samples 9,500 and 10,500.

Recovery is measured from the end of the transition.

| Metric | Ref | Open |
|---|---:|---:|
| R1 recovery time | 50 ± 0 | 50 ± 0 |
| R1 recovered runs | 10/10 | 10/10 |
| R4 recovered runs | 0/10 | 0/10 |
| Accuracy recovery | 1215 ± 841.15 | 7683.33 ± 1033.87 |
| Accuracy recovered runs | 10/10 | 3/10 |
| Accuracy censored runs | 0/10 | 7/10 |
| Final accuracy | 0.5044 ± 0.0058 | 0.5781 ± 0.0073 |
| R1 post-drift Precision@3 | 1.00 | 1.00 |
| R4 post-drift Precision@3 | 0.18 | 0.00 |
| Final rule count | 5 | 93.4 ± 12.48 |

### Findings

- Global relevance recovered at the first eligible
  checkpoint after the gradual transition in all seeds.
- Post-transition recovery was shorter than in S1
  (50 versus 145 samples).
- Adaptation may already occur during the gradual
  transition, limiting direct comparisons of total
  adaptation time with abrupt drift.
- Effective rule weights failed to recover in both
  configurations.
- Predictive recovery remained substantially slower
  than global relevance recovery in the open variant.

Accuracy recovery means include recovered runs only.

## 6. S4 — Negative Control (10 seeds)

SEA threshold drift at sample 10,000.
The relevant feature set remains {x0, x1} throughout
the stream.

| Metric | Ref | Open |
|---|---:|---:|
| R1 pre-drift Precision@2 | 1.00 | 1.00 |
| R1 post-drift Precision@2 | 1.00 | 1.00 |
| R4 pre-drift Precision@2 | 0.7667 | 1.00 |
| R4 post-drift Precision@2 | 0.7667 | 1.00 |
| R1 Jaccard (lag 200) | 1.00 | 1.00 |
| Final accuracy | 0.7176 ± 0.0026 | 0.7176 ± 0.0026 |
| Macro-F1 | 0.4182 | 0.4182 |
| Final rule count | 5 | 147.1 ± 13.59 |

### Findings

- Global relevance correctly identified the relevant
  features before and after the threshold drift.
- R4 remained valid in the open configuration, unlike
  in S1 and S2, where the relevant feature set changed.
- Removing the rule cap substantially increased the
  number of rules without improving predictive accuracy.
- The low macro-F1 highlights the limitations of
  accuracy on this imbalanced stream.

### Limitations

S4 is a relevance-only negative control.
Ranking recovery time is not interpreted because
the ground-truth relevant feature set does not change.

The direct pre-drift versus final window comparison
will be performed in WF5.

## 7. Statistical Analysis

Statistical analysis was performed on 10 paired seeds
using the Wilcoxon signed-rank test, matched-pairs
rank-biserial correlation and a sign test.

Holm correction was applied across seven comparisons.
Bootstrap confidence intervals used 10,000 resamples.

### Primary SQ2 result

S1 global relevance recovery:

- Mean: 145 samples
- Standard deviation: 15 samples
- 95% bootstrap CI: [135, 150]
- Recovery rate: 100%

This supports the predicted recovery interval of
100–250 samples (H2).

### Paired comparisons

| Comparison | Wilcoxon p | Holm p | Effect size |
|---|---:|---:|---:|
| S1 ref: R1 vs R4 | 0.001953 | 0.013672 | -1.0 |
| S1 open: R1 vs R4 | 0.001953 | 0.013672 | -1.0 |
| S2 ref: R1 vs R4 | 0.001953 | 0.013672 | -1.0 |
| S2 open: R1 vs R4 | 0.001953 | 0.013672 | -1.0 |
| S1 open: R1 vs accuracy | 0.001953 | 0.013672 | -1.0 |
| S2 open: R1 vs accuracy | 0.001953 | 0.013672 | -1.0 |
| S1 vs S2: R1 recovery | 0.001953 | 0.013672 | +1.0 |

All seven comparisons remained significant after
Holm correction.

### Interpretation

- Global relevance recovered consistently faster
  than effective rule weights.
- Global relevance recovery preceded predictive
  recovery in the open configuration.
- Gradual drift exhibited shorter post-transition
  recovery than abrupt drift.
- Sign tests confirmed the consistency of the
  observed directions across seeds.

### Statistical limitations

Censored observations represent observation limits,
not actual recovery times.

The S1 versus S2 comparison concerns post-transition
recovery, not total adaptation time.

Identical R1 results across model variants are expected
because R1 depends on the stream rather than the rule base.

## 8. Implementation Risks and Mitigation

Pending.

## 9. Final Conclusions

Pending.