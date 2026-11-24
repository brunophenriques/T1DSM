# WF3 — Feature Relevance Findings

Author: João Pinho  
Research questions: H3 — Is local relevance less stable than global relevance? H6 — Do the weights stored in the rules recover more slowly after drift?

## How to read the labels

| Label | What it identifies | Example |
|---|---|---|
| Rule | A fuzzy region learned by the model. Its ID identifies that region in one run. | Rule 51 in S1 `open`, seed 0 |
| R | A way to measure feature relevance. | R3 computes relevance inside each rule's local window. |
| H | A hypothesis about the expected result. | H3 predicts that R3 will be less stable than R1. |
| S | A synthetic stream used to test the model. | S3 keeps the same features relevant throughout the run. |

For example, **H3 on S3 compares R3 with R1**. For each eligible rule, R3 measures how much its top three features change between checkpoints. R1 provides the global comparison on the same stream. The numbers in the labels are independent.

## 1. Implementation and experiment

- **R1** reads the model's global feature weights from its 200-sample sliding window. **R4** reads the support-weighted mean of the feature weights stored in the active rules. Both use the existing extraction code.
- **R2** estimates between-class separation relative to within-class spread from running class statistics. We evaluated forgetting factors of 0.99, 0.995 and 1.0.
- **R3** applies the R1 formula to the last 200 samples on which each rule had the highest activation. The label enters the window after prediction. A rule needs at least 20 wins and two classes to produce a valid score. Coverage is the percentage of active rules that meet those conditions.
- The optional `--wf3` flag records these measures every 50 samples. The analysis skips the first 500 samples. All 31 tests passed, including checks that enabling WF3 preserves predictions, rule counts, centres and weights. The independent 200-sample window matched R1 at all 16,000 checkpoints.

The experiment comprised **40 runs**, with 20,000 samples per run. S1 changes the relevant features from `{x0, x1, x2}` to `{x5, x6, x7}` after sample 10,000. S3 keeps the original features relevant throughout. Both streams used seeds 0–9 with `ref` (five-rule cap) and `open` (no cap). The [README](README.md) contains the run commands and the archived Elec2 and Agrawal rule examples.

## 2. H3 — Local stability

Mean top-3 Jaccard at a lag of 200 stream samples. R3 pairs checkpoints for the same rule ID. Both windows in a pair have to be valid and belong to the same concept.

| S3 variant | R1 | R3 | R4 | Valid R3 pairs |
|---|---:|---:|---:|---:|
| `ref` | 1.000 | 0.999 | 1.000 | 15,763 |
| `open` | 1.000 | 0.948 | 1.000 | 145,027 |

The local top three changed more frequently in `open`, in line with H3. In `ref`, R3 was nearly as stable as R1. The R4 value for `ref` covers four seeds because the other six had uniform stored weights, which yield no meaningful top-three ranking.

### S1 within each concept

S3 is the stationary control. S1 also allows a stability comparison within each concept, using pairs of checkpoints from the same side of the drift.

| S1 `open` segment | R1 | R3 | R4 | Valid R3 pairs |
|---|---:|---:|---:|---:|
| Before drift | 1.000 | 0.933 | 1.000 | 46,288 |
| After drift | 1.000 | 0.980 | 1.000 | 16,894 |

R3 was less stable than R1 in both segments. Its mean Jaccard was higher after the drift among the eligible pairs in that segment. H6 measures adaptation across the change.

## 3. H6 — Recovery after drift

Recovery is the first of three consecutive checkpoints with tie-aware precision@3 of 1 against the new feature set. The R3 ranking here is the mean of valid local scores weighted by the number of samples in each rule's window.

| S1 `open` measure | Mean recovery ± sample SD | Runs recovered |
|---|---:|---:|
| R1 | 145 ± 16 samples | 10/10 |
| R3 aggregate | 2,635 ± 261 samples | 10/10 |
| R4 | Censored at 10,000 samples | 0/10 |

R4 failed to recover within the observation period, in line with H6. The stored rule weights can persist after the global and local separation estimates have changed.

## 4. R2 — Class separation with and without forgetting

For each feature, R2 measures how far apart the class centres are relative to the spread within each class. A feature scores highly when the class centres are far apart and the samples in each class cluster tightly. R2 maintains the weighted counts, means and sums of squared deviations, updating them as each label arrives. Each step multiplies the previous evidence by λ. At λ = 1, the running estimate matches the batch statistic over the full history.

| R2 forgetting factor | Mean precision@3 before drift | Mean precision@3 after drift | Mean S1 recovery | Runs recovered |
|---:|---:|---:|---:|---:|
| 0.99 | 1.000 | 1.000 | 100 samples | 10/10 |
| 0.995 | 1.000 | 1.000 | 180 samples | 10/10 |
| 1.0 | 1.000 | 0.010 | Censored at 10,000 samples | 0/10 |

Precision uses clean S1 `open` checkpoints. The recovery column applies the same three-checkpoint criterion as H6 to R2 as a separate descriptive analysis. R2 uses only the labelled stream, so the recovery figures are identical in both model configurations. On S1, stronger forgetting helped the estimate follow the change in relevant features. With λ = 1, the pre-drift samples continued to influence the estimate throughout the second half.

### How the ranking changed at the drift

Mean precision@3 across the 10 S1 `open` seeds at selected checkpoints. The checkpoint at 10,000 still uses the original target set (`x0`, `x1`, `x2`). Later checkpoints use the new target set (`x5`, `x6`, `x7`), including during the transition in the estimates.

| Checkpoint | R1 | R2 · λ = 0.99 | R2 · λ = 0.995 | R2 · λ = 1.0 |
|---:|---:|---:|---:|---:|
| 10,000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10,050 | 0.000 | 0.067 | 0.000 | 0.000 |
| 10,100 | 0.567 | 1.000 | 0.000 | 0.000 |
| 10,150 | 1.000 | 1.000 | 0.800 | 0.000 |
| 10,200 | 1.000 | 1.000 | 1.000 | 0.000 |
| 10,500 | 1.000 | 1.000 | 1.000 | 0.000 |

The λ = 0.99 estimate found all three new features by checkpoint 10,100 in every seed. The recovery table above reports the first checkpoint in each seed's qualifying sequence, averaged across seeds.

## 5. R3 validity, window size and coverage

On clean S1 `open` checkpoints, R3 mean precision@3 was **0.734 before drift** and **0.806 after drift**. R3 asks which features separate the classes within the region represented by a rule. This region can have a different mix of data from the global stream, and its window advances only when that rule wins again.

| Global window | S3 mean Jaccard | S1 mean recovery | Predicted interval |
|---:|---:|---:|---:|
| 50 | 1.000 | 50 | 25–100 |
| 100 | 1.000 | 90 | 50–150 |
| 200 | 1.000 | 145 | 100–250 |
| 500 | 1.000 | 295 | 250–550 |
| 1,000 | 1.000 | 545 | 500–1,050 |

In these streams, larger global windows made recovery slower and all five windows already reached the maximum S3 Jaccard. The observed mean recovery fell within the predicted interval at each size.

The first/last analysis found **1,361 run-rule pairs** with two distinct valid windows. In `open`, the mean overlap between the first and last top-three sets was **0.273** on S1 (669 pairs) and **0.506** on S3 (608 pairs). Local rankings can therefore shift over a rule's lifetime even on a stationary stream.

### Example from rule 51

Rule 51 in S1 `open`, seed 0, had a valid local window on each side of the drift. The sample ranges show the first and last wins in each window. Each window contains only samples won by that rule.

| R3 window | Checkpoint | Sample range | Top three features and weights |
|---|---:|---:|---|
| First valid | 5,950 | 4,900–5,932 | `x1` (1.000), `x0` (0.467), `x2` (0.386) |
| Last valid | 20,000 | 11,575–19,990 | `x6` (1.000), `x5` (0.915), `x7` (0.871) |

For this rule, all three top features changed, and both windows recovered the correct feature set for their respective concept.

In S1 `open`, mean R3 coverage was **61.35%**. For seed 0, it rose from **44/73** active rules at the drift to **78/106** at the final checkpoint. Collection on that run took **1.50 ms per sample**, excluding file writing.

## 6. Interpretation and limitations

- Local windows may overlap at a lag of 200 stream samples because each rule wins only some of the samples. The stability results are conditional on the windows available for comparison.
- S1 and S3 are synthetic Gaussian streams with known relevant features. The accuracy, stability and runtime figures apply to these streams and configurations.
- `results_wf3/analysis/` contains the generated evidence. The main files are `h3_summary.csv`, `h6_open.csv`, `validity.csv`, `ranking_recovery.csv`, `first_last_local.csv`, `runs.csv` and `window_tradeoff.png`.
