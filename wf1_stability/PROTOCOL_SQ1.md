# Protocol: SQ1 (stability), Workflow 1

**Question.** In stationary segments, how much does the global relevance ranking G fluctuate over time, and is that more than
sampling noise alone would produce?

**Object measured.** `G = model.current_feature_weights` (recomputed by the model every 50 samples from its last 200 samples),
logged at every checkpoint (50 samples) by `relevance/logger.py`. Rule-level weights are not used for SQ1.

## Primary metric
Top-k Jaccard overlap between G at checkpoint i and at checkpoint i - lag, **lag = 200 samples** (windows do not overlap).
k = 2 on the synthetic stream S3 (two truly relevant features), k = 3 on Elec2.
Pairs are used only inside stationary segments, after the 200-sample buffer has been filled (`warmup_flush`).

## Secondary metrics
Weighted tau (`scipy.stats.weightedtau`), tau-b over all features, Spearman, per-feature SD over checkpoints,
top-k switches per 1,000 samples. Implemented in `relevance/stability.py`, tested in `tests/test_stability.py`.

## Lags
50 / 100 / 150 / 200 / 500 / 2000 samples. Lags below 200 are reported only to quantify the overlap effect
(lag 50 shares 75% of the window, lag 100 50%, lag 150 25%); they are never used for conclusions.

## Streams and runs
| Condition | Stream | Runs | Variability comes from |
|---|---|---|---|
| S1 | synthetic stream with a relevance switch at sample 10,000 (`configs/mve.json`) | 3 seeds | stream generator (model is deterministic) |
| S3 | stationary synthetic, relevant features {0, 1} | 10 seeds | stream generator (model is deterministic) |
| Elec2 | first 10,000 rows, original order | 1 | none between runs (deterministic model, fixed stream) |
| Elec2 shuffled | same rows, random permutation | N = 30 shuffles (seed = permutation) | the permutation |

S3 is a sanity check only: a stationary synthetic stream has no temporal structure, so top-k Jaccard is expected to be about 1.
S1 contains a drift at sample 10,000. Pairs that touch the drift and the 200 samples after it are excluded from the SQ1 statistics
(stationary segments only); the drift itself is analysed under SQ2.

## Preprocessing
`check_elec2_ranges.py` (first 10,000 rows) showed that river's Elec2 is **not** fully normalised: `day` runs 1 to 7, `date` is
monotone with a tiny range (0 to 0.03), and `vicprice`, `vicdemand`, `transfer` are constant over this slice.
* Whole-stream min-max (the baseline's `min_max_scale`) uses future information.
* Expanding min-max turns the monotone `date` into a constant in the original order but not in the shuffled order, which would bias the comparison.
* Used instead: **fixed domain scaling** (`"scaling": "domain"`): `day` -> (day - 1) / 6, all other features keep their [0, 1] scale.
  It is leak-free and independent of row order, so the shuffled control receives exactly the same preprocessing.
* Sensitivity analysis: every Elec2 result is also run **without `date`** (`*_nodate` configs, `analyze_sq1.py --suffix _nodate`), because a time index
  is not a covariate and its tiny within-window variance interacts with the `1e-6` constant in the model's separability formula.
  Conclusions are reported only if they hold in both variants.

## Statistical plan
* Mean ± SD over runs; 95% bootstrap CI (S3, shuffled: over runs; Elec2: moving-block bootstrap with 500-sample blocks over its own checkpoint series).
* Primary test: Elec2 Jaccard@200 against the shuffled distribution, one-sided exact p = (1 + #shuffles at or below Elec2) / (N + 1).
  With N = 10 the smallest possible p is 0.091, so N >= 20 (p_min = 0.048) is needed to be able to reach 5%.
* Wilcoxon signed-rank (paired by seed) with matched-pairs rank-biserial effect size and Holm correction are available in
  `relevance/stats.py` for paired-by-seed comparisons (for example S3 variants). They do not apply to Elec2 vs shuffled, which is not paired.

## Outputs
`wf1_stability/table1_sq1.md` (SQ1 row of Table 1, plus the overlap table and the S3 sanity check), `sq1_per_run.csv`, `fig2a_jaccard_lag200.png/.pdf`.
The same files with the suffix `_nodate` hold the without-`date` sensitivity analysis.

## Known limitations
G is a univariate class-mean separation score inside a 200-sample window, so SQ1 measures the stability of that estimator.
Elec2 has real drift, so it has no truly stationary segment: its Jaccard is an overall fluctuation level, interpretable only next to the shuffled control.
Elec2 is a single stream, and with N = 30 shuffles the smallest possible p is 0.032, so the primary test cannot support a claim about electricity data in general.
Class-mix confound (not tested): Elec2 labels are autocorrelated, so a real 200-sample window can be class-imbalanced while a shuffled one is not, which alone could make G less stable on the real stream. A check is to log the class proportion per window and see whether it explains the Jaccard drops.