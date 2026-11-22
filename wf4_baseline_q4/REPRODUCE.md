# Baseline Reproduction Report (`baseline-v0`)

## 1. Environment & Snapshot
- **Git Tag:** `baseline-v0`
- **Dependency Lock:** `wf4_baseline_q4/env.txt`
- **OS / Environment:** Windows (Conda environment `AC2`; see `env.txt` for exact Python and package versions)

## 2. Reproduction Command
```bash
python exp_sota_comparison.py
```

Timed execution command used on Windows CMD:
```cmd
python -c "import time, runpy; t0=time.perf_counter(); runpy.run_path('exp_sota_comparison.py', run_name='__main__'); dt=time.perf_counter()-t0; print(f'\nTotal runtime: {dt:.2f}s ({dt/60:.2f} min)')"
```

## 3. Measured Runtime
- **Total wall-clock runtime (all 8 datasets $\times$ 7 models = 56 runs):** `692.08s` (`11.53 min`)
- **`ENF_Unim_Safe` runtime range:** `0.4s` (`Phishing_real`, $N=1{,}250$) to `17.4s` (`Sine_synth`, $N=40{,}000$)
- **`ENF_Unim_PA` runtime range:** `0.5s` (`Phishing_real`, $N=1{,}250$) to `21.0s` (`Sine_synth`, $N=40{,}000$)

## 4. UnimNeuron Comparison vs. `results_exp/exp_stream_summary.tex`

All predictive accuracies, final rule counts, `Pred_fallbacks`, and `P_instabilities` match the reference LaTeX summary at reported precision ($\Delta\text{Acc} = 0.000$ across all datasets):

| Dataset | Model | $N$ | Ref Acc$_{\text{final}}$ | Our Acc$_{\text{final}}$ | Ref Acc$_{\text{mean}}$ | Our Acc$_{\text{mean}}$ | $\Delta$ Acc | Ref Rules | Our Rules | Ref Time | Our Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Elec2_real` | `ENF_Unim_Safe` | 10,000 | 0.591 | 0.5906 | 0.603 | 0.6033 | 0.000 | 5 | 5 | — | 3.6s |
| `Elec2_real` | `ENF_Unim_PA` | 10,000 | 0.797 | 0.7973 | 0.803 | 0.8030 | 0.000 | 5 | 5 | — | 4.4s |
| `CreditCard_real` | `ENF_Unim_Safe` | 5,000 | 0.999 | 0.9994 | 0.999 | 0.9991 | 0.000 | 5 | 5 | — | 1.8s |
| `CreditCard_real` | `ENF_Unim_PA` | 5,000 | 0.999 | 0.9994 | 0.999 | 0.9991 | 0.000 | 5 | 5 | — | 2.1s |
| `Phishing_real` | `ENF_Unim_Safe` | 1,250 | 0.549 | 0.5488 | 0.553 | 0.5534 | 0.000 | 5 | 5 | — | 0.4s |
| `Phishing_real` | `ENF_Unim_PA` | 1,250 | 0.530 | 0.5304 | 0.554 | 0.5542 | 0.000 | 5 | 5 | — | 0.5s |
| `Bananas_real` | `ENF_Unim_Safe` | 5,000 | 0.541 | 0.5414 | 0.544 | 0.5441 | 0.000 | 5 | 5 | 9.4s | 1.7s |
| `Bananas_real` | `ENF_Unim_PA` | 5,000 | 0.526 | 0.5264 | 0.535 | 0.5350 | 0.000 | 5 | 5 | 11.8s | 2.1s |
| `ConceptDrift_Agrawal` | `ENF_Unim_Safe` | 20,000 | 0.672 | 0.6718 | 0.673 | 0.6729 | 0.000 | 5 | 5 | 37.1s | 6.8s |
| `ConceptDrift_Agrawal` | `ENF_Unim_PA` | 20,000 | 0.607 | 0.6073 | 0.609 | 0.6085 | 0.000 | 5 | 5 | 45.0s | 8.2s |
| `Agrawal_synth` | `ENF_Unim_Safe` | 20,000 | 0.673 | 0.6728 | 0.669 | 0.6692 | 0.000 | 5 | 5 | 37.2s | 8.7s |
| `Agrawal_synth` | `ENF_Unim_PA` | 20,000 | 0.610 | 0.6102 | 0.607 | 0.6069 | 0.000 | 5 | 5 | 44.8s | 10.3s |
| `RandomRBFDrift_synth` | `ENF_Unim_Safe` | 20,000 | 0.672 | 0.6717 | 0.674 | 0.6735 | 0.000 | 5 | 5 | — | 8.4s |
| `RandomRBFDrift_synth` | `ENF_Unim_PA` | 20,000 | 0.604 | 0.6042 | 0.609 | 0.6085 | 0.000 | 5 | 5 | — | 10.3s |
| `Sine_synth` | `ENF_Unim_Safe` | 40,000 | 0.560 | 0.5599 | 0.563 | 0.5633 | 0.000 | 5 | 5 | — | 17.4s |
| `Sine_synth` | `ENF_Unim_PA` | 40,000 | 0.868 | 0.8676 | 0.861 | 0.8609 | 0.000 | 5 | 5 | — | 21.0s |

## 5. Complete Full-Suite Benchmark Log (All 7 Models $\times$ 8 Datasets)

| Dataset | Model | $N$ | Acc$_{\text{final}}$ | Acc$_{\text{mean}}$ | Rules$_{\text{final}}$ | Time [s] | Fallbacks | Instabilities |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Elec2_real` | `ePL` | 10,000 | 0.7014 | 0.6597 | 1 | 1.3 | 0 | 0 |
| `Elec2_real` | `ePL_plus` | 10,000 | 0.7367 | 0.6905 | 1 | 1.4 | 0 | 0 |
| `Elec2_real` | `exTS` | 10,000 | 0.7286 | 0.7012 | 7 | 6.5 | 0 | 421 |
| `Elec2_real` | `Simpl_eTS` | 10,000 | 0.8021 | 0.7227 | 1 | 1.4 | 0 | 0 |
| `Elec2_real` | `eMG` | 10,000 | 0.5706 | 0.5491 | 1 | 1.7 | 0 | 0 |
| `Elec2_real` | `ENF_Unim_Safe` | 10,000 | 0.5906 | 0.6033 | 5 | 3.6 | 0 | 0 |
| `Elec2_real` | `ENF_Unim_PA` | 10,000 | 0.7973 | 0.8030 | 5 | 4.4 | 0 | 0 |
| `CreditCard_real` | `ePL` | 5,000 | 0.9594 | 0.8307 | 6 | 1.9 | 0 | 0 |
| `CreditCard_real` | `ePL_plus` | 5,000 | 0.9594 | 0.8307 | 1 | 1.3 | 0 | 0 |
| `CreditCard_real` | `exTS` | 5,000 | 0.9594 | 0.8307 | 3 | 18.6 | 0 | 160 |
| `CreditCard_real` | `Simpl_eTS` | 5,000 | 0.9594 | 0.8307 | 1 | 1.2 | 0 | 0 |
| `CreditCard_real` | `eMG` | 5,000 | 0.9594 | 0.8307 | 4 | 2.2 | 0 | 0 |
| `CreditCard_real` | `ENF_Unim_Safe` | 5,000 | 0.9994 | 0.9991 | 5 | 1.8 | 0 | 0 |
| `CreditCard_real` | `ENF_Unim_PA` | 5,000 | 0.9994 | 0.9991 | 5 | 2.1 | 0 | 0 |
| `Phishing_real` | `ePL` | 1,250 | 0.7408 | 0.4757 | 44 | 1.7 | 0 | 0 |
| `Phishing_real` | `ePL_plus` | 1,250 | 0.7520 | 0.4855 | 1 | 0.2 | 0 | 0 |
| `Phishing_real` | `exTS` | 1,250 | 0.7536 | 0.4835 | 56 | 2.0 | 0 | 0 |
| `Phishing_real` | `Simpl_eTS` | 1,250 | 0.7512 | 0.4850 | 1 | 0.2 | 0 | 0 |
| `Phishing_real` | `eMG` | 1,250 | 0.7248 | 0.4643 | 116 | 8.2 | 0 | 0 |
| `Phishing_real` | `ENF_Unim_Safe` | 1,250 | 0.5488 | 0.5534 | 5 | 0.4 | 0 | 0 |
| `Phishing_real` | `ENF_Unim_PA` | 1,250 | 0.5304 | 0.5542 | 5 | 0.5 | 0 | 0 |
| `Bananas_real` | `ePL` | 5,000 | 0.5054 | 0.4357 | 1 | 0.4 | 0 | 0 |
| `Bananas_real` | `ePL_plus` | 5,000 | 0.5118 | 0.4400 | 1 | 0.6 | 0 | 0 |
| `Bananas_real` | `exTS` | 5,000 | 0.5122 | 0.4497 | 3 | 3.8 | 0 | 153 |
| `Bananas_real` | `Simpl_eTS` | 5,000 | 0.5454 | 0.4649 | 1 | 0.6 | 0 | 0 |
| `Bananas_real` | `eMG` | 5,000 | 0.5026 | 0.4413 | 1 | 1.0 | 0 | 0 |
| `Bananas_real` | `ENF_Unim_Safe` | 5,000 | 0.5414 | 0.5441 | 5 | 1.7 | 0 | 0 |
| `Bananas_real` | `ENF_Unim_PA` | 5,000 | 0.5264 | 0.5350 | 5 | 2.1 | 0 | 0 |
| `ConceptDrift_Agrawal` | `ePL` | 20,000 | 0.6574 | 0.6288 | 56 | 110.3 | 0 | 0 |
| `ConceptDrift_Agrawal` | `ePL_plus` | 20,000 | 0.6368 | 0.6177 | 1 | 2.9 | 0 | 0 |
| `ConceptDrift_Agrawal` | `exTS` | 20,000 | 0.5757 | 0.5704 | 5 | 15.5 | 0 | 897 |
| `ConceptDrift_Agrawal` | `Simpl_eTS` | 20,000 | 0.6653 | 0.6377 | 1 | 2.8 | 0 | 0 |
| `ConceptDrift_Agrawal` | `eMG` | 20,000 | 0.5797 | 0.5532 | 34 | 57.3 | 0 | 0 |
| `ConceptDrift_Agrawal` | `ENF_Unim_Safe` | 20,000 | 0.6718 | 0.6729 | 5 | 6.8 | 0 | 0 |
| `ConceptDrift_Agrawal` | `ENF_Unim_PA` | 20,000 | 0.6073 | 0.6085 | 5 | 8.2 | 0 | 0 |
| `Agrawal_synth` | `ePL` | 20,000 | 0.6573 | 0.6212 | 62 | 128.7 | 0 | 0 |
| `Agrawal_synth` | `ePL_plus` | 20,000 | 0.6404 | 0.6076 | 1 | 3.7 | 0 | 0 |
| `Agrawal_synth` | `exTS` | 20,000 | 0.5986 | 0.5822 | 4 | 20.1 | 0 | 893 |
| `Agrawal_synth` | `Simpl_eTS` | 20,000 | 0.6663 | 0.6333 | 1 | 3.6 | 0 | 0 |
| `Agrawal_synth` | `eMG` | 20,000 | 0.5803 | 0.5496 | 36 | 75.0 | 0 | 0 |
| `Agrawal_synth` | `ENF_Unim_Safe` | 20,000 | 0.6728 | 0.6692 | 5 | 8.7 | 0 | 0 |
| `Agrawal_synth` | `ENF_Unim_PA` | 20,000 | 0.6102 | 0.6069 | 5 | 10.3 | 0 | 0 |
| `RandomRBFDrift_synth` | `ePL` | 20,000 | 0.6305 | 0.5993 | 3 | 5.1 | 0 | 0 |
| `RandomRBFDrift_synth` | `ePL_plus` | 20,000 | 0.6288 | 0.5980 | 1 | 3.3 | 0 | 0 |
| `RandomRBFDrift_synth` | `exTS` | 20,000 | 0.6638 | 0.6453 | 6 | 18.2 | 0 | 906 |
| `RandomRBFDrift_synth` | `Simpl_eTS` | 20,000 | 0.7107 | 0.6733 | 1 | 3.6 | 0 | 0 |
| `RandomRBFDrift_synth` | `eMG` | 20,000 | 0.6542 | 0.6265 | 2 | 5.8 | 0 | 0 |
| `RandomRBFDrift_synth` | `ENF_Unim_Safe` | 20,000 | 0.6717 | 0.6735 | 5 | 8.4 | 0 | 0 |
| `RandomRBFDrift_synth` | `ENF_Unim_PA` | 20,000 | 0.6042 | 0.6085 | 5 | 10.3 | 0 | 0 |
| `Sine_synth` | `ePL` | 40,000 | 0.9189 | 0.8945 | 2 | 5.4 | 0 | 0 |
| `Sine_synth` | `ePL_plus` | 40,000 | 0.9189 | 0.8947 | 2 | 6.7 | 0 | 0 |
| `Sine_synth` | `exTS` | 40,000 | 0.9088 | 0.8879 | 4 | 21.4 | 0 | 1893 |
| `Sine_synth` | `Simpl_eTS` | 40,000 | 0.9693 | 0.9448 | 1 | 6.4 | 0 | 0 |
| `Sine_synth` | `eMG` | 40,000 | 0.9153 | 0.8908 | 4 | 18.0 | 0 | 0 |
| `Sine_synth` | `ENF_Unim_Safe` | 40,000 | 0.5599 | 0.5633 | 5 | 17.4 | 0 | 0 |
| `Sine_synth` | `ENF_Unim_PA` | 40,000 | 0.8676 | 0.8609 | 5 | 21.0 | 0 | 0 |

## 6. Documented Differences & Leakage Note (`E0`)
- Every single predictive metric, final rule count, fallback count, and instability indicator across all 56 model–dataset combinations matches `results_exp/exp_stream_summary.tex` identically. Only wall-clock runtime differs due to local hardware.
- In our `run_stream` / `wf4_baseline_q4` experimental pipeline, online feature normalization uses strictly past observations ($x_{1 \dots t-1}$) rather than global offline scaling to prevent any future information leakage.