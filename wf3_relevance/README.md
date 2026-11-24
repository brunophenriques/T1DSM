# Workflow 3 · Global and rule-local feature relevance

See [FINDINGS.md](FINDINGS.md) for the experiment results and interpretation.

## Run

```powershell
python run_relevance.py --wf3 --config configs/protocol/s1_gauss_abrupt.json --variants open --seeds 0
python run_relevance.py --wf3 --config configs/protocol/s1_gauss_abrupt.json
python run_relevance.py --wf3 --config configs/protocol/s3_gauss_stationary.json
python -m wf3_relevance.analyze
python -m pytest -q tests/test_wf3_relevance.py
```

Use the first command for a single-seed demo. The next two run the protocol seeds 0–9 for both `ref` and `open`. Use `--outdir` on the runner and `--results` on the analysis command for a custom output folder. The default `results_wf3/` directory contains generated files and is excluded from Git.

The tables linked in [FINDINGS.md](FINDINGS.md) are committed under [`evidence/`](evidence/). After running the full S1/S3 suite, regenerate them with `python -m wf3_relevance.analyze --evidence-dir wf3_relevance/evidence`. Runtime values in `runs.csv` depend on the machine and concurrent load.

Each seed has `wf3_checkpoints.csv` with R1, R4, coverage, the weighted mean of R3, the three R2 variants and the global window sweep. `wf3_local.csv` stores R3 by rule ID with window size and first/last sample. Times are one-based counts. `metrics.json` reports `wf3_ms_per_sample`, the wall time spent in WF3 callbacks per input sample, including activation recomputation and checkpoint collection. This excludes file writing.

## Where relevance lives

- **R1** reads `current_feature_weights`, which the model refreshes every 50 samples from its last 200 labelled points. A non-degenerate separability vector has floor 0.2 and maximum 1.0. The model uses uniform weights during initialisation or if it finds no separation.
- **R2** applies the same between/within-class statistic to exponentially weighted class masses, means and sums of squared deviations. At each sample, every class's mass and scatter are decayed by λ. With λ = 1, the statistic matches a batch over the entire history.
- **R3** uses the model's separability formula on the last 200 wins of each rule. The highest activation is computed in the same state as the prediction and the true label is added only after predicting. At least 20 wins and two classes are required. Coverage counts eligible active rules. A merge preserves the surviving rule's own history.
- **R4** is the support-weighted mean of stored `rule.w`. A rule copies R1 when created and its weights change on a merge. A feature enters the rule as `h = w·a + (1−w)·e`, where `a` is membership and `e` is the neutral element. In the reference setting the cap of five disables merges, and new rules are usually pruned immediately. The initial uniform weights can therefore persist for the entire run.

## Two rules from the archived experiments

| Dataset | Rule 1 antecedents shown in the export | Consequent | Interpretation |
|---|---|---|---|
| Elec2 | `day` Medium (1.00), `nswprice` Low (0.22), `date` Low (0.20), `period` Medium (0.20), `nswdemand` Medium (0.20), `vicprice` Low (0.20) | Class 0, p = 0.554, support 747 | The membership for `day` enters at full strength. The other shown memberships are pulled closer to the neutral element. |
| Agrawal | `salary` Medium, `commission` Low, `age` Medium, `elevel` Low, `car` Low, `zipcode` Medium (all 0.11, rounded from 1/9) | Class 0, p = 0.679, support 127 | The stored weights give every input equal influence on how much the membership is pulled towards the neutral element. |

Sources [Elec2 export](../results_exp/Elec2_real/rules_ENF_Unim_Safe.tex) and [Agrawal export](../results_exp/Agrawal_synth/rules_ENF_Unim_Safe.tex). The historical Elec2 export includes `date`. The current protocol loader drops it.

## Analysis

- **H3** uses top-3 Jaccard at lag 200 within one concept, after the first 500 samples. The paired table compares R1 and R3 at the same checkpoints for each eligible rule-level pair, then averages differences within seeds. Rules must be active at both checkpoints and have valid, nonuniform local weights. The tables report matched pairs and how many active rules contributed.
- **H6** uses a window-count-weighted mean of valid R3 vectors for a single local ranking per checkpoint. Recovery is the first of three consecutive post-drift checkpoints with tie-aware top-3 precision of 1. The H6 CSV puts a right-censored run's follow-up in its censor-limit field and leaves its recovery time and exact pairwise delays blank.
- **Window sweep** uses global diagnostic windows on the same stream, with S3 stability and S1 recovery. The predicted range runs from half a window to a full window plus a 50-sample refresh interval. The plot shows mean ± sample SD across seeds. Both variants see the same global diagnostic windows for a given seed.

Two risks affect interpretation. Sparse winners can leave local windows short, stale or overlapping even at lag 200, so the analysis reports coverage, window bounds and eligible pairs. Recomputing activations adds work proportional to the active rule count, which can grow in `open`. Callback time is measured separately, and the single-seed measurement is easier to interpret than wall time from concurrent runs.

## Results · 40 protocol runs

| Result | Observation |
|---|---|
| H3, S3 `open` | Mean top-3 Jaccard was **1.000** for R1 and R4 and **0.948** for R3 across 10 seeds, using 145,027 eligible local pairs. In `ref`, R3 was **0.999** and R4 was uniform throughout in six seeds. |
| H6, S1 `open` | R1 recovered in **145** samples on average and the weighted mean of R3 in **2,635**. Both recovered in all 10 seeds. R4 was censored in all 10 seeds at **10,000** samples after drift. |
| Window sweep | All five windows had mean Jaccard **1.000** on S3. From 50 to 1,000 points, mean recovery on S1 was **50, 90, 145, 295, 545** samples. Each mean fell inside its predicted range. |
| Validity, S1 `open` | On clean post-drift checkpoints, mean precision@3 was **1.000** for R2 with λ = 0.99 or 0.995, **0.011** with λ = 1, and **0.806** for eligible R3 windows. |
| Coverage and cost, S1 `open` | Mean R3 coverage across analysis checkpoints and seeds was **61.35%**. Seed 0 had 44/73 valid rules at the drift and 78/106 at the end. Its WF3 callbacks took **1.50 ms/sample**. |

The first/last comparison contains 1,361 run-rule pairs with distinct valid windows. All 20 S1 runs reproduced the saved WF2 checkpoints for every field except runtime. The 200-point global diagnostic vector agreed exactly with R1 at all 16,000 checkpoints across both streams.
