# Baseline Reproduction & E0 Equivalence Report (`baseline-v0`)

## 1. Environment & Snapshot
- **Git Tag:** `baseline-v0` (PR: `baseline: runnable`)
- **Dependency Lock:** `wf4_baseline_q4/requirements.txt` and `wf4_baseline_q4/env.txt` (`river==0.26.1`, `evolvingfuzzysystems==0.1.5`, `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `pytest`)
- **OS / Environment:** Windows (Conda environment `AC2`)

## 2. Baseline Reference Execution vs. `results_exp/exp_stream_summary.tex`
```bash
python exp_sota_comparison.py
```
- Optional import `enfs_uni0_evolving` (`ENFSUni0Evolving`) is guarded with `try/except ImportError` so the baseline suite executes cleanly when optional modules are absent.
- Direct comparison between our reproduction runs (`FULL_REF`, `max_rules=5`, core defaults) and the archived reference table `results_exp/exp_stream_summary.tex` (`ENF_Unim_Safe`):

| Dataset | $N$ | Reference `Acc_final` | Reference `Acc_mean` | Reference `Rules_final` | Reference Time [s] | Our Run (`FULL_REF`) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Elec2_real` | `10,000` | `0.591` | `0.603` | `5` | `19.6s` | `0.5908` (`5` rules, `2.59s` causal) | Matches `Acc_final=0.591` & `5` rules (`NoChange=0.8375`) |
| `Agrawal_synth` (`fn=0`, $N=1000$) | `1,000` | — | — | — | — | `0.6040` (offline) / `0.6020` (causal) | Exact sample-by-sample match in `test_e0_real_baseline_vs_causal` |
| `ConceptDrift_Agrawal` / `S3` | `5,000` | `0.672` (`N=20k`) | `0.673` (`N=20k`) | `5` | `37.1s` (`N=20k`) | `0.5978` (`N=5k`, `5` rules, `0.95s`) | `MajorityClass=0.6037` on 50/50 split `fn=0 -> fn=2` |

## 3. E0 Baseline Equivalence & Normalization Comparison
Verified directly in `wf4_baseline_q4/tests/test_wf4.py::test_e0_real_baseline_vs_causal` comparing `exp_sota_comparison.py::prequential_nf_generic` against `wf4_baseline_q4.run_wf4.evaluate_single_run()` on `Agrawal(classification_function=0, seed=42, N=1000)` with `max_rules=5` and core defaults:

| Setup | Scaling | `Acc_final` | `Rules_final` | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `exp_sota_comparison.py` (`prequential_nf_generic`) | Offline `min_max_scale` | `0.6040` | `5` | Reference baseline |
| `evaluate_single_run(..., use_causal_scaler=False)` | Offline `min_max_scale` | `0.6040` | `5` | Exact match (`y_pred` identical) |
| `evaluate_single_run(..., use_causal_scaler=True)` | `FastCausalMinMaxScaler` | `0.6020` | `5` | Zero future leakage (`-0.20` p.p. vs. offline) |

## 4. Measured Execution Runtimes & Reproduction Commands
- **Unit, wrapper & E0 suite (`pytest wf4_baseline_q4/tests -v`):** `~3s`
- **Timed demo (`python -m wf4_baseline_q4.run_wf4 --config configs/demo.json`):** `~3s` (generates `demo_r1_heatmap.png` and `demo_rule_growth.png`)
- **Targeted variant refresh (`--refresh-variants SHUFFLED_W FULL_REF NO_W_REF --workers 4`):** `~2–3 min` (`~8 min` single-threaded)
- **Post-drift Top-$k$ ranking recovery (`--drift-ranking --workers 4`):** `~2 min` across `S1–S4` (`10` seeds)
- **Full 500-run benchmark (`--full-sq4`, single-threaded):** `~38.6 min` (`2316s` total; `Elec2` accounts for `1407s`)
- **Statistical inference & Table 1 row (`python -m wf4_baseline_q4.stats_q4`):** `< 1s`