# Workflow 4 — Baseline + Information Content (SQ4) & Incremental Feature Distances

## Installation
```bash
python -m pip install -r wf4_baseline_q4/requirements.txt
# Or reproduce the exact Conda environment:
# conda create -n AC2 --file wf4_baseline_q4/env.txt
```

## Execution (run from repository root)
1. **Validation & E0 equivalence tests:**
   ```bash
   python -m pytest wf4_baseline_q4/tests -v
   ```
2. **Timed demo (< 60s, outputs `demo_r1_heatmap.png` and `demo_rule_growth.png`):**
   ```bash
   python -m wf4_baseline_q4.run_wf4 --config configs/demo.json
   ```
3. **Full benchmark (`S1–S4` + `Elec2`, seeds `0–9`):**
   ```bash
   python -m wf4_baseline_q4.run_wf4 --full-sq4
   ```
4. **Eligibility check, statistical inference (`S1–S4`), `Elec2` descriptive summary & Table 1 row:**
   ```bash
   python -m wf4_baseline_q4.stats_q4
   ```

## Main Artifacts
- `wf4_baseline_q4/PROTOCOL.md`: Experimental design, predictive eligibility gate, diagnostic findings, and statistical plan.
- `wf4_baseline_q4/REPRODUCE.md`: `baseline-v0` snapshot, E0 equivalence results, and reproduction commands.
- `wf4_baseline_q4/results/wf4_runs.csv`: Full 500-run benchmark log (`5 streams x 10 variants x 10 seeds`).
- `wf4_baseline_q4/results/sq4_eligibility_check.csv` & `sq4_summary_stats.csv`: Predictive eligibility gate (`0/5` streams above trivial baselines by `> 2.0 p.p.`) and paired Wilcoxon + Holm statistics.
- `wf4_baseline_q4/results/elec2_descriptive_summary.csv` & `wf4_secondary_summary.csv`: Descriptive real-stream and secondary structural/resource summaries.