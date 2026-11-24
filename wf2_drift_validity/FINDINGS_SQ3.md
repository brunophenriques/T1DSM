# WF2 — Validity Findings (SQ3)

Author: Luis Soeiro (23 Nov 2026). Research question: SQ3 — do the relevance scores rank the truly
relevant features first?

All results are **exploratory** until the team tags `protocol-freeze` (PROTOCOL.md §13). Mean ± SD use
the sample SD (ddof = 1); CIs are 95% bootstrap CIs over the 10 seeds (10,000 resamples). Numbers come
from `results_wf2/sq3/` (`python -m wf2_drift_validity.sq3_validity`).

## 1. Table 1 — SQ3 row

| Research question | Primary metric | Result | 95% CI | Seeds at 1.0 |
|---|---|---|---|---|
| SQ3 — Validity | Top-3 precision of R1 after warm-up in the clean segments of S1 (both halves) | 1.00 ± 0.00 | [1.00, 1.00] | 10/10 |

One line each for the report:
- **S5 (Agrawal F0 → F6):** R1 ranks the relevant features above chance but never perfectly: top-k precision 0.45 ± 0.05 on F0 {age} (k = 1, chance 0.11) and 0.71 ± 0.02 on F6 (k = 3, chance 0.33), 0/10 seeds at 1. The 200-sample window is too short for these functions (oracle check). The F2 → F6 sensitivity pair gives the same picture.
- **S6 (non-causal copy of x0):** R1 cannot tell a non-causal copy from a relevant feature: the copy is in R1's top 3 at 66% of the checkpoints, and precision falls from 1.00 (S1) to 0.78 ± 0.02 on the same data (Wilcoxon p = 0.002, Holm 0.004).
- **R5 (permutation importance, S1, `open`):** the model itself uses the relevant features only partly: R5 top-3 precision 0.57 ± 0.05 before and 0.42 ± 0.04 after the drift (chance 0.30), against 1.00 for R1 (p = 0.002, Holm 0.004).

## 2. Primary result: S1

| | First half {x0, x1, x2} | Second half {x5, x6, x7} |
|---|---:|---:|
| R1 top-3 precision | 1.00 (10/10 seeds) | 1.00 (10/10) |
| R1 ROC-AUC | 1.00 (10/10) | 1.00 (10/10) |
| R4 top-3 precision, `ref` | 0.58 ± 0.36 | 0.18 ± 0.16 |
| R4 top-3 precision, `open` | 1.00 | 0.00 |

- R1 ranks exactly the three relevant features first at every clean checkpoint, in every seed, as
  expected for mean-shift relevance (H4). R1 is identical for `ref` and `open`.
- S2 (gradual) gives the same: 1.00 in both clean segments, 10/10 seeds.
- R4, the weights the rules use, is fully valid only before the drift and only in `open`. In `ref` it
  is at chance (0.30, uniform weights) in 6 seeds and 1.00 in 4. After the drift R4 never ranks the new
  features in its top 3 in `open` (see FINDINGS.md, H6).
- S4 (SEA, negative control): R1 precision 1.00 before and after the threshold drift.

## 3. Oracle check and the choice of S5

R1's own formula on 100,000 stationary samples per concept (seed 99), and on the 500 non-overlapping
200-sample windows of the same data (`results_wf2/oracle/`):

| Concept | k | Oracle precision (100,000) | 200-sample windows: mean precision | windows at 1.0 |
|---|---:|---:|---:|---:|
| S1 first half / second half | 3 | 1.00 / 1.00 | 1.00 / 1.00 | 100% / 100% |
| SEA θ = 8 / θ = 7 | 2 | 1.00 / 1.00 | 1.00 / 1.00 | 100% / 100% |
| Agrawal F0 {age} | 1 | 1.00 | 0.40 | 40% |
| Agrawal F1 | 2 | 0.50 | 0.53 | 7% |
| Agrawal F2 {age, elevel} | 2 | 1.00 | 0.49 | 11% |
| Agrawal F3 | 3 | 0.33 | 0.44 | 0% |
| Agrawal F4 | 3 | 1.00 | 0.51 | 4% |
| Agrawal F5 | 3 | 0.67 | 0.72 | 16% |
| Agrawal F6 {salary, commission, loan} | 3 | 1.00 | 0.72 | 15% |
| Agrawal F7 | 3 | 1.00 | 0.63 | 16% |
| Agrawal F8 | 4 | 0.75 | 0.75 | 13% |
| Agrawal F9 | 5 | 0.80 | 0.57 | 0% |

- The two failure modes of PROTOCOL.md §12 both occur. F1, F3, F5, F8 and F9 are not seen even on
  100,000 samples: their labels depend on interactions between features (F1, F3, F5) or give some
  relevant features only a weak effect on the class means (F8, F9). F0, F2,
  F4, F6 and F7 are seen on 100,000 samples but rarely on 200: R1's window is too short for them.
- **F0** was expected to be invisible. It is seen on 100,000 samples only because river draws age as an
  integer from 20 to 80, so the class means differ by 0.74 years. On 200-sample windows age is ranked
  first only 40% of the time.
- **Selection.** Detected functions: F0, F2, F4, F6, F7. Disjoint pairs: (F0, F6), (F0, F7), (F2, F6).
  The frozen rule (lowest index) selects **F0 → F6**. Because the protocol also named F0 the failure case,
  F2 → F6 (the lowest pair without F0) was run as a labelled sensitivity analysis. **The team decides
  which pair the report uses** (PROTOCOL.md §14).
- Learnability (seed 99, PROTOCOL.md §7): `open` beats the majority class by 0.0 pp on F0, −0.9 pp on F2
  and +2.7 pp on F6, all below the 5 pp bar. S5 is relevance-only: no accuracy claims.

## 4. S5 — Agrawal

R1 top-k precision in the clean segments (k = size of the active relevant set). R1 is identical for
`ref` and `open`. Chance level is k/9: 0.11 for k = 1, 0.22 for k = 2, 0.33 for k = 3.

| Stream | First concept | Second concept: F6 {salary, commission, loan}, k = 3 |
|---|---:|---:|
| S5: F0 {age}, k = 1 → F6 (frozen rule) | 0.45 ± 0.05 (0/10 seeds at 1) | 0.71 ± 0.02 (0/10) |
| S5b: F2 {age, elevel}, k = 2 → F6 (sensitivity) | 0.48 ± 0.02 (0/10) | 0.71 ± 0.02 (0/10) |
| F0 alone, stationary (failure case), k = 1 | 0.44 ± 0.03 (0/10) | — |

| ROC-AUC of R1 | First concept | F6 |
|---|---:|---:|
| S5 (F0 → F6) | 0.73 ± 0.03 | 0.83 ± 0.01 |
| S5b (F2 → F6) | 0.70 ± 0.02 | 0.83 ± 0.01 |

- R1 ranks the relevant features above chance on every Agrawal concept, but never perfectly: on 200
  samples the class-mean statistic is too noisy for these functions, as the oracle check predicted
  (§3: the same functions reach precision 1 only on 100,000 samples).
- F0: age is ranked first at 44% of the checkpoints (chance 11%). F0 is a failure case of the 200-sample
  window, not of the statistic itself.
- The two candidate pairs lead to the same conclusion, so the choice between them (§3) does not change
  the SQ3 result.
- Every seed counts as "recovered" after the drift, but at very different times (median 1,350 samples,
  from 50 to 5,100): with precision around 0.71, three checkpoints in a row at precision 1 happen by
  chance. Ranking recovery is therefore not interpreted on S5.
- `ref` and `open` both stay near the majority-class rate (S5 is relevance-only).

## 5. S6 — a non-causal copy of x0

S6 is S1 plus x10 = x0 + N(0, 0.02): the first ten columns and the labels are bit-identical to S1, so the
two streams can be compared seed by seed.

| | First half {x0, x1, x2} | Second half {x5, x6, x7} |
|---|---:|---:|
| R1 top-3 precision, S1 | 1.00 (10/10) | 1.00 (10/10) |
| R1 top-3 precision, S6 | 0.78 ± 0.02 (0/10) | 1.00 (10/10) |
| R1 ROC-AUC, S6 | 0.96 ± 0.01 | 1.00 |
| Copy x10 in R1's top 3 (share of clean checkpoints) | 0.66 (0.55–0.73 by seed) | 0.00 |

- **Yes, the per-feature statistic marks the copy as relevant.** While x0 matters, x10 is in R1's top 3
  at two thirds of the checkpoints and pushes one truly relevant feature out (S6 vs S1, Wilcoxon p = 0.002,
  lower in all 10 seeds).
- After the drift x0 is irrelevant, so its copy is too, and R1 drops both.
- R1 measures how well a feature separates the classes, not whether the model or the label depends on it.
  A feature correlated with a relevant one looks relevant.
- Ranking recovery on S6 is the same as on S1 (145 samples), so the copy does not slow re-orientation.

## 6. R5 — permutation importance

R5 is a model-based check: every 500 samples, the accuracy drop when one feature is permuted in the last
200 samples (5 repeats, prediction only, its own random generator). Run on S1 and S5 with both variants
(`results_wf2/r5/`). Adding R5 changed no metric of the 20 S1 runs (compared with the same runs without R5).

| S1 | `ref` | `open` |
|---|---:|---:|
| Accuracy on the 200-sample windows before permuting, first / second half | 0.53 / 0.53 | 0.65 / 0.58 |
| R5 top-3 precision, first half {x0, x1, x2} | 0.34 ± 0.05 | 0.57 ± 0.05 |
| R5 top-3 precision, second half {x5, x6, x7} | 0.30 ± 0.01 | 0.42 ± 0.04 |
| R5 ROC-AUC, first / second half | 0.53 / 0.50 | 0.69 / 0.59 |
| R5 top 3 shared with R1's top 3, first / second half | 0.34 / 0.30 | 0.57 / 0.42 |

| S5 (F0 → F6) | `ref` | `open` |
|---|---:|---:|
| R5 precision on F0 {age}, k = 1 (chance 0.11) | 0.11 | 0.11 |
| R5 precision on F6, k = 3 (chance 0.33) | 0.33 | 0.35 ± 0.01 |

On S5 neither variant learns anything (relevance-only), so R5 is at chance for both.

- `ref` ignores its input: permuting a feature almost never changes a prediction, so R5 sits at chance
  (precision 0.30, AUC 0.50). This confirms from the model's side that `ref`'s rule base is frozen.
- `open` does use the relevant features before the drift (precision 0.57, chance 0.30), but much less
  sharply than R1 says (R1: 1.00). After the drift the model barely uses the new features (0.42). This
  is the same picture as R4 (FINDINGS.md): the rules keep the old weights.
- R1 describes the data and R5 describes the model, and here they disagree. R1 being valid (SQ3) does
  not mean the model relies on the features R1 ranks first.

## 7. Secondary comparisons

Paired by seed (Wilcoxon signed-rank, rank-biserial, sign test; Holm within this family of two):

| Comparison | Wilcoxon p | Holm p | Rank-biserial | Meaning |
|---|---:|---:|---:|---|
| S6 vs S1: R1 top-3 precision, first half | 0.002 | 0.004 | −1.0 | the copy lowers R1's precision in every seed |
| S1 `open`: R5 vs R1 top-3 precision | 0.002 | 0.004 | −1.0 | the model uses the relevant features less sharply than R1 ranks them, in every seed |

## 8. Interpretation

- **H4 holds for mean-shift relevance:** on S1 (and S2, S4) R1 ranks exactly the relevant features first after warm-up, in every seed (precision 1, AUC 1).
- **It fails when relevance is not a class-mean difference or the window is too short:** on the Agrawal functions R1 is above chance but never perfect, including F0, the protocol's planned failure case.
- **R1 is a property of the data, not of the model:** it marks a non-causal copy as relevant (S6), and the model's own importances (R5) agree with it only partly, and less after a drift.
- So a valid R1 ranking says which features separate the classes in the recent window. It does not say which features the model's prediction depends on (R4, R5), nor which features cause the label (S6).

## 9. Limitations

- Results are exploratory until `protocol-freeze`.
- The S5 pair is pending a team decision (§3).
- S5's precision values are averages over clean checkpoints; with k = 1 (F0) precision is simply the share
  of checkpoints where age is ranked first.
- R5 on `ref` measures a model that predicts at chance, so its importances are close to zero for every
  feature; only `open` is interpreted.
