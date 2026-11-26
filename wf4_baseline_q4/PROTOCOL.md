# Experimental Protocol — Workflow 4: SQ4 (Information Content) & Incremental Feature Distance

## 1. Research Objectives & Diagnostic Finding
1. **SQ4 (Information Content — Primary Deliverable):** Test whether the learned feature-relevance weights ($w$) in `ENF_UnimNeuron_Safe` yield genuine predictive gains beyond a permutation control (`FULL` vs. `SHUFFLED_W` under `max_rules=None`) on streams where the classifier beats trivial baselines.
2. **Incremental Feature-Distance Estimators (Exploratory Extension):** Evaluate whether replacing the baseline's 200-sample `FeatureSeparabilityBuffer` with $O(C \cdot d)$ online EMA estimators (`INC_FISHER_L2`, `INC_L1_CENTROID`, `INC_RELIEFF`, $\alpha = 0.05$) preserves predictive behavior and post-drift feature ranking while reducing Python traced peak memory footprint. Post-drift rank trajectories (`track_trajectory=True`) are scoped as future exploratory analysis and do not block SQ4 validation.

## 2. Model Variants & Ablation Wrappers
All variants wrap or subclass `ENF_UnimNeuron_Safe` without modifying the baseline core file:

### A. Information-Content & Scale Controls (SQ4)
- **`FULL` (Baseline, `max_rules=None`):** Original `ENF_UnimNeuron_Safe` using `FeatureSeparabilityBuffer` (200-sample sliding window, discrete refresh every `50` observations).
- **`SHUFFLED_W` (`max_rules=None`):** Applies a fixed, seed-dependent derangement $\pi$ ($\pi(j) \neq j\ \forall j$) to each rule's weight vector $w$ upon creation, update, and merging. Preserves the exact weight distribution and scale while breaking all feature alignments.
- **`NO_W` (`max_rules=None`):** Forces uniform unit weights ($w_j = 1.0$) across all new, existing, and merged rules.
- **`FULL_REF` & `NO_W_REF` (`max_rules=5` + core defaults):** Secondary reference-capacity controls using `max_rules=5` and default core parameters.

### B. Incremental Feature-Distance Formulations (Exploratory)
All incremental variants replace `self.sep_buffer` with `IncrementalSeparabilityBuffer`, updating per-class EMA statistics at every observation ($\alpha = 0.05$, fixed *a priori*, effective window $\approx 2/\alpha - 1 \approx 39$ samples vs. the 200-sample baseline sliding buffer) while preserving the baseline's `50`-observation weight refresh cadence (`K % 50 == 0`) and rule-management policy:
- **`INC_FISHER_L2`:** Online Fisher-style between/within-class variance ratio using per-class EMA means $\mu_{c,j}(t)$ and variances $\sigma^2_{c,j}(t)$.
- **`INC_L1_CENTROID`:** Online Manhattan ($L_1$) pairwise distance between class EMA centroids $\sum_{c_1 < c_2} |\mu_{c_1,j}(t) - \mu_{c_2,j}(t)|$.
- **`INC_RELIEFF`:** Online ReliefF-inspired margin estimator comparing sample-to-centroid distances against the nearest different-class centroid (*miss*) versus the sample's own class centroid (*hit*) — distinct from classical $k$-NN ReliefF.

### C. Trivial Streaming Baselines
- **`MajorityClass`:** Predicts the most frequent class observed in $y_{1 \dots t-1}$.
- **`NoChange`:** Persistence predictor returning $\hat{y}_t = y_{t-1}$ (defaults to class `0` at $t=0$).

## 3. Experimental Design, Leakage Control & Structural Measurement Notes
- **Pre-Specified Predictive Eligibility Criterion:** A stream is eligible for confirmatory SQ4 inference only if `FULL` outperforms both `MajorityClass` and `NoChange` by more than **`+2.0` percentage points** in mean prequential accuracy. On `S1–S4`, `FULL` matches `MajorityClass` within $\pm 0.5$ p.p.; on `Elec2`, `FULL` (`59.69%`) trails `NoChange` (`83.75%`) by `24.06` p.p. Consequently, `0/5` streams meet the predictive eligibility threshold, and `S1–S4` are reported as a diagnostic analysis showing that while $w$ recovers true post-drift feature rankings, the classifier's antecedent aggregation and consequent voting do not translate weight alignment into predictive accuracy gains.
- **Rule Turnover & Merge Behavior:** `rule_turnover_id_approx` tracks object-ID changes in `model.rules` between consecutive samples. This count is exact when `max_rules=None`, but under `max_rules=5` (`*_REF`), newly created rules (`support=1`) evicted on the same step are not captured by between-step ID diffs. Furthermore, `_maybe_merge_last_rule()` requires `min(support) >= 10` immediately after `_create_rule()` (`support=1`), so rule merges never trigger in the unmodified baseline core.
- **Clean Latency & Memory Separation:** Per-sample latency (`ms_per_sample`) is measured across seeds `0–9` without `tracemalloc` instrumentation; peak Python traced memory (`peak_mem_kb`) is profiled in a separate dedicated pass on `seed 0`.

## 4. Statistical Analysis Plan (Table 1, Row SQ4)
- **Synthetic Streams (`S1–S4`):** Two-sided paired Wilcoxon signed-rank test (`zero_method="wilcox"`) matched on `(dataset, seed)` across `seeds 0–9` (verified `one_to_one` merge and explicit `max_rules` checks), reporting `n_nonzero` and matched-pairs rank-biserial correlation $r = \frac{W_+ - W_-}{W_+ + W_-}$ (marked `NaN` when `n_nonzero < 5`, such as `S4` where 9/10 seeds tie exactly).
- **Multiple-Testing Correction:** Holm–Bonferroni applied to full unrounded `p_raw` values separately within each hypothesis family (`SQ4_diagnostic_all_synth`, `SQ4_no_w_unbounded`, `SQ4_scale_control_ref`, and `INC_*`).
- **Real Stream (`Elec2`):** Reported descriptively (`elec2_descriptive_summary.csv`) without 10-seed inferential $p$-values to avoid pseudo-replication on a single fixed real sequence.