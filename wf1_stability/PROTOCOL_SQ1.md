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
| S3 | stationary synthetic, relevant features {0, 1} | 10 seeds | stream generator (model is deterministic) |
| Elec2 | first 10,000 rows, original order | 1 | none between runs (deterministic model, fixed stream) |
| Elec2 shuffled | same rows, random permutation | N shuffles (seed = permutation) | the permutation |
S3 is a sanity check only: a stationary synthetic stream has no temporal structure, so top-k Jaccard is expected to be about 1.

## Preprocessing
Elec2 must be used **without** expanding min-max scaling if its features already lie in [0, 1] (check with
`python wf1_stability/check_elec2_ranges.py`): expanding scaling turns a monotone feature such as `date` into a constant in the
original order but not in the shuffled order, which would bias the comparison. Config: `"scaling": "none"`.

## Statistical plan
* Mean ± SD over runs; 95% bootstrap CI (S3, shuffled: over runs; Elec2: moving-block bootstrap with 500-sample blocks over its own checkpoint series).
* Primary test: Elec2 Jaccard@200 against the shuffled distribution, one-sided exact p = (1 + #shuffles at or below Elec2) / (N + 1).
  With N = 10 the smallest possible p is 0.091, so N >= 20 (p_min = 0.048) is needed to be able to reach 5%.
* Wilcoxon signed-rank (paired by seed) with matched-pairs rank-biserial effect size and Holm correction are available in
  `relevance/stats.py` for paired-by-seed comparisons (for example S3 variants). They do not apply to Elec2 vs shuffled, which is not paired.

## Outputs
`wf1_stability/table1_sq1.md` (SQ1 row of Table 1, plus the overlap table and the S3 sanity check), `sq1_per_run.csv`, `fig2a_jaccard_lag200.png/.pdf`.

## Known limitations
G is a univariate class-mean separation score inside a 200-sample window, so SQ1 measures the stability of that estimator.
Elec2 has real drift, so it has no truly stationary segment: its Jaccard is an overall fluctuation level, interpretable only next to the shuffled control.
