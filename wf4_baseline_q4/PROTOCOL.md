# Experimental Protocol — Workflow 4: SQ4 (Information Content) & Incremental Feature Distance

## 1. Research Objectives
1. **SQ4 (Information Content — Primary Deliverable):** Test whether the learned feature-relevance weights ($w$) in `ENF_UnimNeuron_Safe` carry genuine predictive information beyond a simple scaling effect (`FULL` vs. `SHUFFLED_W` under `max_rules=None`).
2. **Incremental Feature-Distance Estimators & Post-Drift Ranking Recovery (Exploratory Extension):** Evaluate whether replacing the baseline's 200-sample `FeatureSeparabilityBuffer` with $O(C \cdot d)$ online EMA estimators (`INC_FISHER_L2`, `INC_L1_CENTROID`, `INC_RELIEFF`) preserves predictive accuracy, reduces tracked memory footprint, and recovers true feature rankings after concept drift (`--drift-ranking`).

## 2. Model Variants & Ablation Wrappers
All variants wrap or subclass `ENF_UnimNeuron_Safe` without modifying the baseline core file:

### A. Information-Content & Scale Controls (SQ4)
- **`FULL` (Baseline):** Original `ENF_UnimNeuron_Safe` using `FeatureSeparabilityBuffer` (200-sample sliding window, discrete refresh every `50` observations).
- **`SHUFFLED_W`:** Applies a fixed, seed-dependent derangement $\pi$ ($\pi(j) \neq j$ for all $j$) to each rule's weight vector $w$ upon creation, update, and merging. Preserves the exact weight distribution and scale while guaranteeing zero aligned features.
- **`NO_W`:** Forces uniform unit weights ($w_j = 1.0$) across all new, existing, and merged rules.
- **`FULL_REF` / `NO_W_REF`:** Scale-control pair evaluated under the baseline's default rule cap (`max_rules=5`).

### B. Incremental Feature-Distance Formulations (Exploratory)
To isolate strictly the relevance estimator from structural hyperparameters, all incremental variants replace `self.sep_buffer` with `IncrementalSeparabilityBuffer`, updating per-class EMA statistics at every observation ($\alpha = 0.05$, fixed *a priori*) while keeping the baseline's `50`-observation weight refresh cadence (`K % 50 == 0`) and rule creation/merging policy:
- **`INC_FISHER_L2`:** Online Fisher-style between/within-class variance ratio using per-class EMA means $\mu_{c,j}(t)$ and variances $\sigma^2_{c,j}(t)$.
- **`INC_L1_CENTROID`:** Online Manhattan ($L_1$) pairwise distance between class EMA centroids $\sum_{c_1 < c_2} |\mu_{c_1,j}(t) - \mu_{c_2,j}(t)|$.
- **`INC_RELIEFF`:** Online ReliefF-inspired margin estimator comparing sample-to-centroid distances against the nearest different-class centroid (*miss*) versus the sample's own class centroid (*hit*) — distinct from classical $k$-NN ReliefF.
- **`INC_FISHER_L2_MATCHED` (Trajectory Control):** Same estimator as `INC_FISHER_L2` with $\alpha = 0.01$ ($N_{\text{eff}} = \frac{2}{\alpha}-1 = 199$ updates per class), included in `--drift-ranking` to disentangle forgetting-horizon length from estimator formulation.

### C. Trivial Streaming Baselines
- **`MajorityClass`:** Predicts the most frequent class observed in $y_{1 \dots t-1}$.
- **`NoChange`:** Persistence predictor returning $\hat{y}_t = y_{t-1}$ (defaults to class `0` at $t=0$).

## 3. Experimental Design, Leakage Control & Resource Profiling
- **Predictive Streams:** Synthetic drift benchmarks `S1` (abrupt switch), `S2` (gradual switch), `S3` (Agrawal drift), `S4` (SEA drift + noise features), plus the real `Elec2` stream (`N=10,000`). Pure-noise streams are excluded from SQ4 inference.
- **Predictive Eligibility Gate:** Before interpreting `FULL` vs. `SHUFFLED_W` differences as predictive gains, `FULL` must outperform both `MajorityClass` and `NoChange` by at least `+2.0` percentage points.
- **Primary Metric:** Overall prequential accuracy $\text{Acc} = \frac{1}{N}\sum_{t=1}^{N} \mathbb{I}(\hat{y}_t = y_t)$ under strict test-then-train evaluation and causal online MinMax scaling (`FastCausalMinMaxScaler`, using strictly $x_{1 \dots t-1}$).
- **Rule Structural Accounting:** Both non-intrusive `ENFRuleSafe` object-ID differences (`*_id_approx`) and exact instance-wrapped counts (`rules_created_exact`, `rules_evicted_exact`, `rules_merged_exact`) are recorded without altering core behavior.
- **Seeds (`0–9`):** Ten paired seeds per synthetic stream. On `Elec2`, the real sequence is fixed: deterministic models (`FULL`, `NO_W`, `INC_*`) yield identical predictions across seeds, whereas `SHUFFLED_W` variability reflects the 10 random derangements $\pi$.
- **Clean Latency & Memory Separation:** Per-sample latency (`ms_per_sample`) is measured across seeds `0–9` without `tracemalloc` instrumentation; peak Python traced memory (`peak_mem_kb`) is profiled in a separate dedicated pass on `seed 0`.

## 4. Statistical Analysis Plan (Table 1, Row SQ4)
- **Synthetic Streams (`S1–S4`):** Two-sided paired Wilcoxon signed-rank test (`zero_method="wilcox"`) matched on `(dataset, seed)` across `seeds 0–9` (verified `one_to_one` merge, identical `max_rules`), with matched-pairs rank-biserial correlation $r = \frac{W_+ - W_-}{W_+ + W_-}$.
- **Multiple-Testing Correction:** Holm–Bonferroni applied to full unrounded `p_raw` values separately within each pre-specified hypothesis family:
  1. `SQ4_diagnostic_all_synth`: `FULL` vs. `SHUFFLED_W` (`max_rules=None`) across `S1–S4` (populates Table 1, Row SQ4).
  2. `SQ4_no_w_unbounded`: `FULL` vs. `NO_W` (`max_rules=None`) across `S1–S4`.
  3. `SQ4_scale_control_ref`: `FULL_REF` vs. `NO_W_REF` (`max_rules=5`) across `S1–S4`.
  4. `INC_*`: Each incremental estimator vs. `FULL` (`max_rules=None`) across `S1–S4`.
- **Real Stream (`Elec2`):** Reported descriptively (`elec2_descriptive_summary.csv`) without 10-seed inferential $p$-values to avoid pseudo-replication on a single fixed real sequence.

## 5. Post-Drift Feature-Ranking Recovery Protocol (`--drift-ranking`)
Evaluates Top-$k$ precision of `current_feature_weights` across `seeds 0–9` at 50-observation checkpoints, defining recovery as 3 consecutive checkpoints with Top-$k$ precision $= 1.0$ after the end of drift:
- **`S1` (Abrupt Switch, $t = 2000$):** Pre-drift reference at $t = 1950$ (`[0, 1]`); post-drift recovery measured after $t = 2000$ (`[4, 5]`).
- **`S2` (Gradual Ramp, $t \in [1500, 2500]$):** Pre-drift reference measured at $t = 1450$ (`[0, 1]`, prior to ramp onset); post-drift recovery measured after ramp completion at $t = 2500$ (`[2, 3]`).
- **`S3` (Agrawal `fn=0 -> fn=2`, $t = 2500$):** Ground-truth features in alphabetically sorted River keys are `[0]` (`age`) pre-drift and `[0, 3]` (`age`, `elevel`) post-drift. Because `fn=2` involves non-monotonic conditional thresholds where class-conditional means nearly cancel, centroid/variance separability estimators do not recover `[0, 3]` — documented explicitly as a structural limitation of first-order separability rather than post-drift latency.
- **`S4` (SEA `variant 0 -> 2`, $t = 2500$):** Pure threshold shift preserving `[0, 1]` (`sea_att1`, `sea_att2`) across both regimes; reported strictly as a stationary-relevance stability control.