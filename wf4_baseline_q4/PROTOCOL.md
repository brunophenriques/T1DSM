# Experimental Protocol — Workflow 4: SQ4 (Information Content) & Incremental Feature Distance

## 1. Research Objectives
1. **SQ4 (Information Content — Primary Deliverable):** Test whether the learned feature-relevance weights ($w$) in `ENF_UnimNeuron_Safe` carry genuine predictive information beyond a simple scaling effect (`FULL` vs. `SHUFFLED_W` under `max_rules=None`).
2. **Incremental Feature-Distance Estimators (Exploratory Extension):** Evaluate whether replacing the baseline's 200-sample `FeatureSeparabilityBuffer` with $O(C \cdot d)$ online EMA estimators (`INC_FISHER_L2`, `INC_L1_CENTROID`, `INC_RELIEFF`) preserves predictive accuracy while reducing tracked memory footprint. Post-drift rank recovery trajectories (`track_trajectory=True`) are scoped as future exploratory analysis and do not block SQ4 validation.

## 2. Model Variants & Ablation Wrappers
All variants wrap or subclass `ENF_UnimNeuron_Safe` without modifying the baseline core file:

### A. Information-Content & Scale Controls (SQ4)
- **`FULL` (Baseline):** Original `ENF_UnimNeuron_Safe` using `FeatureSeparabilityBuffer` (200-sample sliding window, discrete refresh every `50` observations).
- **`SHUFFLED_W`:** Applies a fixed, seed-dependent non-identity permutation $\pi$ to each rule's weight vector $w$ upon creation, update, and merging. Preserves the exact weight distribution and scale while breaking feature alignment.
- **`NO_W`:** Forces uniform unit weights ($w_j = 1.0$) across all new, existing, and merged rules.

### B. Incremental Feature-Distance Formulations (Exploratory)
To isolate strictly the relevance estimator from structural hyperparameters, all incremental variants replace `self.sep_buffer` with `IncrementalSeparabilityBuffer`, updating per-class EMA statistics at every observation ($\alpha = 0.05$, fixed *a priori*) while keeping the baseline's `50`-observation weight refresh cadence (`K % 50 == 0`) and rule creation/merging policy:
- **`INC_FISHER_L2`:** Online Fisher-style between/within-class variance ratio using per-class EMA means $\mu_{c,j}(t)$ and variances $\sigma^2_{c,j}(t)$.
- **`INC_L1_CENTROID`:** Online Manhattan ($L_1$) pairwise distance between class EMA centroids $\sum_{c_1 < c_2} |\mu_{c_1,j}(t) - \mu_{c_2,j}(t)|$.
- **`INC_RELIEFF`:** Online ReliefF-inspired margin estimator comparing sample-to-centroid distances against the nearest different-class centroid (*miss*) versus the sample's own class centroid (*hit*) — distinct from classical $k$-NN ReliefF.

### C. Trivial Streaming Baselines
- **`MajorityClass`:** Predicts the most frequent class observed in $y_{1 \dots t-1}$.
- **`NoChange`:** Persistence predictor returning $\hat{y}_t = y_{t-1}$ (defaults to class `0` at $t=0$).

## 3. Experimental Design, Leakage Control & Resource Profiling
- **Predictive Streams:** Synthetic drift benchmarks `S1` (abrupt switch), `S2` (gradual switch), `S3` (Agrawal drift), `S4` (SEA drift + noise features), plus the real `Elec2` stream (`N=10,000`). Pure-noise streams are excluded from SQ4 inference.
- **Primary Metric:** Overall prequential accuracy $\text{Acc} = \frac{1}{N}\sum_{t=1}^{N} \mathbb{I}(\hat{y}_t = y_t)$ under strict test-then-train evaluation and causal online MinMax scaling (`FastCausalMinMaxScaler`, using strictly $x_{1 \dots t-1}$).
- **Seeds (`0–9`):** Ten paired seeds per synthetic stream (updated from the initial 5-seed proposal to increase statistical power). On `Elec2`, the real sequence is fixed: deterministic models (`FULL`, `NO_W`, `INC_*`) yield identical predictions across seeds, whereas `SHUFFLED_W` variability reflects the 10 random permutations $\pi$.
- **Clean Latency & Memory Separation:** Per-sample latency (`ms_per_sample`) is measured across seeds `0–9` without `tracemalloc` instrumentation; peak Python traced memory (`peak_mem_kb`) is profiled in a separate dedicated pass on `seed 0`.

## 4. Statistical Analysis Plan (Table 1, Row SQ4)
- **Synthetic Streams (`S1–S4`):** Two-sided paired Wilcoxon signed-rank test (`zero_method="wilcox"`) matched on `(dataset, seed)` across `seeds 0–9` (verified `one_to_one` merge, identical `max_rules`), with matched-pairs rank-biserial correlation $r = \frac{W_+ - W_-}{W_+ + W_-}$.
- **Multiple-Testing Correction:** Holm–Bonferroni applied to full unrounded `p_raw` values separately within each pre-specified hypothesis family:
  1. `SQ4_primary`: `FULL` vs. `SHUFFLED_W` (`max_rules=None`) across `S1–S4` (populates Table 1, Row SQ4).
  2. `SQ4_scale_control`: `FULL_REF` vs. `NO_W_REF` (`max_rules=5`) across `S1–S4`.
  3. `INC_*`: Each incremental estimator vs. `FULL` (`max_rules=None`) across `S1–S4`.
- **Real Stream (`Elec2`):** Reported descriptively (`elec2_descriptive_summary.csv`) without 10-seed inferential $p$-values to avoid pseudo-replication on a single fixed real sequence.