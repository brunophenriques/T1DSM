# Experimental Protocol — Workflow 4: SQ4 (Information Content) & Incremental Feature Distance

## 1. Research Objectives
1. **SQ4 (Information Content):** Test whether the learned feature-relevance weights ($w$) in `ENF_UnimNeuron_Safe` carry genuine discriminative information beyond a simple scaling effect (`FULL` vs. `SHUFFLED_W` under `max_rules=None`).
2. **Incremental Feature-Distance Comparison (`INC_W`):** Evaluate whether replacing the baseline's discrete 50-sample `FeatureSeparabilityBuffer` refresh with continuous, sample-by-sample incremental feature-distance formulations improves predictive accuracy, weight stability, drift responsiveness, or memory/runtime efficiency.

## 2. Model Variants & Ablation Wrappers
All variants wrap or subclass `ENF_UnimNeuron_Safe` without editing the baseline source file:

### A. Information-Content & Scale Controls (SQ4)
- **`FULL` (Baseline):** Original `ENF_UnimNeuron_Safe` using `FeatureSeparabilityBuffer` (discrete refresh every 50 observations).
- **`SHUFFLED_W`:** Applies a fixed seed-dependent random permutation $\pi$ to each rule's learned weight vector $w$. Preserves exact weight scale and distribution while breaking feature alignment.
- **`NO_W`:** Replaces all feature weights with uniform values ($w_j = 1.0$).

### B. Incremental Feature-Distance Formulations (Replacing the 50-Sample Buffer)
All incremental variants update class statistics online at every observation $t$ using a fixed smoothing factor $\alpha = 0.05$ (fixed *a priori*, never tuned on test streams) and normalize $w(t) \in [0, 1]$:
- **`INC_FISHER_L2` (Incremental Squared Euclidean / Fisher Separability):**
  Maintains per-class EMA means $\mu_{c,j}(t)$ and variances $\sigma^2_{c,j}(t)$, computing:
  $$w_j(t) \propto \frac{\sum_{c_1 < c_2} \big(\mu_{c_1,j}(t) - \mu_{c_2,j}(t)\big)^2}{\sum_{c} \sigma^2_{c,j}(t) + \epsilon}$$
- **`INC_L1_CENTROID` (Incremental Manhattan / $L_1$ Inter-Class Distance):**
  Replaces squared differences with $L_1$ distances between online class centroids to test robustness against noisy observations:
  $$w_j(t) \propto \sum_{c_1 < c_2} \big|\mu_{c_1,j}(t) - \mu_{c_2,j}(t)\big|$$
- **`INC_RELIEFF` (Incremental Nearest-Hit vs. Nearest-Miss Distance):**
  Updates feature relevance via the sample-wise distance margin between the nearest different-class centroid (*miss*) and the sample's own class centroid (*hit*):
  $$s_j(t) = (1 - \alpha)\,s_j(t-1) + \alpha\,\Big(\big|x_{t,j} - \mu_{\text{miss}, j}(t)\big| - \big|x_{t,j} - \mu_{\text{hit}, j}(t)\big|\Big)$$

### C. Trivial Streaming Baselines
- **`MajorityClass`:** Predicts the most frequent class observed in $y_{1 \dots t-1}$.
- **`NoChange`:** Persistence predictor returning $\hat{y}_t = y_{t-1}$ (defaults to class `0` at $t=0$).

## 3. Experimental Comparisons & Controls
- **Primary SQ4 Test (Information Content):**
  - **Comparison:** `FULL` vs. `SHUFFLED_W` with `max_rules=None` (removes rule-eviction caps so differences reflect weight alignment alone).
  - **Streams:** Predictive streams only (`S1`, `S2`, `S3`, `S4`, `Elec2`), excluding pure-noise streams where no ground-truth relevance exists.
  - **Primary Metric:** Mean prequential accuracy across seeds `0–9`. Expected: `FULL > SHUFFLED_W`.
- **Scale Control (`FULL` vs. `NO_W` on Reference Config):**
  - Reported separately with default `max_rules`, explicitly documented as testing **weight scale** rather than information content.
- **Incremental Distance Evaluation (`INC_FISHER_L2`, `INC_L1_CENTROID`, `INC_RELIEFF` vs. `FULL`):**
  - Evaluated across `S1–S4` and `Elec2` (`seeds 0–9`) on:
    1. **Predictive quality:** Mean prequential accuracy, Macro-F1, and post-drift accuracy recovery time.
    2. **Weight stability & validity:** Multi-lag rank correlation (Kendall $\tau_b$ / Spearman at lags 50, 500, 2000), top-$k$ feature-switch frequency, and alignment with ground-truth relevant features (testing whether removing the 50-sample step delay speeds up reorientation after drift).
    3. **Structural & resource trade-off:** Active rule count, rule turnover ($\text{created} + \text{deleted}$), latency ($\text{ms/sample}$ — checking if per-sample updates cost more CPU than 50-step batch refreshes), and peak memory (`tracemalloc` KB).

## 4. Prequential Protocol, Leakage Control & Seeds
- **Strict Test-Then-Train:** At step $t$, $\hat{y}_t$ is predicted before $y_t$ is revealed to `learn_one`. Online normalization uses strictly past observations $x_{1 \dots t-1}$.
- **Seeds (`0–9`):** 10 paired repetitions per stream. Because `ENF_UnimNeuron_Safe` is deterministic, the seed controls only the stream generator (and the fixed permutation $\pi$ in `SHUFFLED_W`).

## 5. Statistical Analysis Plan (Table 1, Row SQ4 & Incremental Comparison)
- **Hypothesis Testing:** Two-sided paired **Wilcoxon signed-rank test** across seeds `0–9` per dataset for:
  1. `FULL` vs. `SHUFFLED_W` (populates Row SQ4 of Table 1).
  2. Each incremental distance variant (`INC_FISHER_L2`, `INC_L1_CENTROID`, `INC_RELIEFF`) vs. `FULL`.
- **Effect Size & Correction:** Matched-pairs rank-biserial correlation $r = \frac{W_+ - W_-}{W_+ + W_-}$, with **Holm–Bonferroni** adjustment across streams ($\alpha = 0.05$).