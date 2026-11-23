## SQ1: stability of the global relevance weights G (stationary segments, buffer flushed)

Primary metric: top-k Jaccard at lag 200 samples (no window overlap). Mean ± SD over runs; CI = 95%.

| condition | k | runs | **Jaccard@200** | 95% CI | tau-b@200 | weighted tau@200 | Spearman@200 | top-k switches /1000 | mean SD of weights |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 3 | 3 | **1.000 ± 0.000** | [1.000, 1.000] (bootstrap over runs) | 0.465 ± 0.008 | 0.609 ± 0.007 | 0.636 ± 0.007 | 0.00 ± 0.00 | 0.065 ± 0.001 |
| S3 | 2 | 10 | **1.000 ± 0.000** | [1.000, 1.000] (bootstrap over runs) | 0.356 ± 0.011 | 0.565 ± 0.009 | 0.486 ± 0.011 | 0.00 ± 0.00 | 0.037 ± 0.001 |
| Elec2 | 3 | 1 | **0.702 ± n/a** | [0.663, 0.744] (block bootstrap (500-sample blocks)) | 0.874 ± n/a | 0.888 ± n/a | 0.944 ± n/a | 6.73 ± n/a | 0.044 ± n/a |
| Elec2 shuffled | 3 | 30 | **0.883 ± 0.024** | [0.875, 0.892] (bootstrap over runs) | 0.961 ± 0.007 | 0.967 ± 0.006 | 0.985 ± 0.003 | 2.74 ± 0.57 | 0.013 ± 0.001 |

### Primary test: Elec2 against its shuffled distribution

Elec2 Jaccard@200 = 0.702 (95% CI 0.663-0.744, block bootstrap (500-sample blocks)); shuffled mean = 0.883 ± 0.024 over 30 shuffles. One-sided exact p (Elec2 lower than shuffled) = 0.032; the smallest possible p with 30 shuffles is 0.032.

### Sanity check S3 (stationary synthetic)

Jaccard@200 = 1.000 (expected about 1). Within-run SD over checkpoints = 0.000; between-seed SD of run means = 0.000.

### Effect of window overlap (lags below 200)

Mean Jaccard by lag. The model's buffer holds 200 samples, so snapshots `lag` apart share `1 - lag/200` of their data.

| condition | lag 50 (75% overlap) | lag 100 (50% overlap) | lag 150 (25% overlap) | lag 200 (0% overlap) | lag 500 (0% overlap) | lag 2000 (0% overlap) |
|---|---|---|---|---|---|---|
| S1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Elec2 | 0.832 | 0.759 | 0.735 | 0.702 | 0.743 | 0.739 |
| Elec2 shuffled | 0.931 | 0.910 | 0.894 | 0.883 | 0.887 | 0.885 |
