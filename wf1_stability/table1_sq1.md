## SQ1: stability of the global relevance weights G (stationary segments, buffer flushed)

Primary metric: top-k Jaccard at lag 200 samples (no window overlap). Mean ± SD over runs; CI = 95%.

| condition | k | runs | **Jaccard@200** | 95% CI | tau-b@200 | weighted tau@200 | Spearman@200 | top-k switches /1000 | mean SD of weights |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 3 | 3 | **1.000 ± 0.000** | [1.000, 1.000] (bootstrap over runs) | 0.465 ± 0.008 | 0.609 ± 0.007 | 0.636 ± 0.007 | 0.00 ± 0.00 | 0.065 ± 0.001 |
| S3 | 2 | 10 | **1.000 ± 0.000** | [1.000, 1.000] (bootstrap over runs) | 0.356 ± 0.011 | 0.565 ± 0.009 | 0.486 ± 0.011 | 0.00 ± 0.00 | 0.037 ± 0.001 |

### Sanity check S3 (stationary synthetic)

Jaccard@200 = 1.000 (expected about 1). Within-run SD over checkpoints = 0.000; between-seed SD of run means = 0.000.

### Effect of window overlap (lags below 200)

Mean Jaccard by lag. The model's buffer holds 200 samples, so snapshots `lag` apart share `1 - lag/200` of their data.

| condition | lag 50 (75% overlap) | lag 100 (50% overlap) | lag 150 (25% overlap) | lag 200 (0% overlap) | lag 500 (0% overlap) | lag 2000 (0% overlap) |
|---|---|---|---|---|---|---|
| S1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
