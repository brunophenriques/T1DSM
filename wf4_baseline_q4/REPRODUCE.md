# Baseline Reproduction & E0 Equivalence Report (`baseline-v0`)

## 1. Environment & Snapshot
- **Git Tag:** `baseline-v0` (PR: `baseline: runnable`)
- **Dependency Lock:** `wf4_baseline_q4/requirements.txt` and `wf4_baseline_q4/env.txt` (`river==0.26.1`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest`)
- **OS / Environment:** Windows (Conda environment `AC2`)

## 2. Baseline Reference Execution
```bash
python exp_sota_comparison.py
```
- Optional import `enfs_uni0_evolving` (`ENFSUni0Evolving`) is guarded with `try/except ImportError` so the baseline suite executes cleanly when optional modules are absent.
- Reference baseline summary tables are archived in `results_exp/exp_stream_summary.tex` (`Elec2_real` `ENF_Unim_Safe`: `N=10000`, `Acc_final=0.591`, `Acc_mean=0.616`, `Rules_final=5`).

## 3. E0 Baseline Equivalence & Normalization Comparison
Verified directly in `wf4_baseline_q4/tests/test_wf4.py::test_e0_real_baseline_vs_causal` comparing `exp_sota_comparison.py::prequential_nf_generic` against `wf4_baseline_q4.run_wf4.evaluate_single_run()` on `Agrawal(classification_function=0, seed=42, N=1000)` with reference parameters (`max_rules=5, safe_threshold=0.07, min_firing_for_update=0.05, tau_merge=0.62, drift_sensitivity=0.35`):

| Setup | Scaling | `Acc_final` | `Acc_mean` (traj.) | `Rules_final` | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `exp_sota_comparison.py` (`prequential_nf_generic`) | Offline `min_max_scale` | `0.6040` | `0.6107` | `5` | Reference baseline |
| `evaluate_single_run(..., use_causal_scaler=False)` | Offline `min_max_scale` | `0.6040` | `0.6107` | `5` | Exact sample-by-sample match (`y_pred` identical) |
| `evaluate_single_run(..., use_causal_scaler=True)` | `FastCausalMinMaxScaler` | `0.6020` | `0.6062` | `5` | Zero future leakage (`-0.20` p.p. final, `-0.45` p.p. mean) |

## 4. Workflow 4 Reproduction Commands (from Repository Root)
```bash
# 1. Install dependencies in a clean environment
python -m pip install -r wf4_baseline_q4/requirements.txt

# 2. Run all unit, leakage, wrapper, and E0 equivalence tests
python -m pytest wf4_baseline_q4/tests -v

# 3. Run timed demo (< 60s) and generate R1 heatmap + rule-growth plots
python -m wf4_baseline_q4.run_wf4 --config configs/demo.json

# 4. Run full SQ4 + incremental benchmark (S1-S4 + Elec2, seeds 0-9)
python -m wf4_baseline_q4.run_wf4 --full-sq4

# 5. Generate paired synthetic Wilcoxon + Holm stats, Elec2 & secondary summaries, and Table 1 row
python -m wf4_baseline_q4.stats_q4
```