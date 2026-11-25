# WF5: window analysis and restabilisation (SQ1 × SQ2)

Owners: Minês + Bruno. Protocol: `PROTOCOL.md` §4 (window analysis) and §5 (restabilisation time).
Delivers Figure 2b and the restabilisation times. Branch `experiment/wf5-windows`.

## How to run

```powershell
# S3 (Gaussian, protocol config) was not saved in main; this regenerates it (~2 min, ref only)
python run_relevance.py --config configs/protocol/s3_gauss_stationary.json --variants ref --outdir results_wf5 --jobs 4
python tests/test_wf5_windows.py
python wf5_windows/analyze_wf5.py                      # all streams -> wf5_windows/results/
python wf5_windows/analyze_wf5.py --streams S3 Elec2   # Minês's part only (files get a _S3_Elec2 suffix)
python wf5_windows/analyze_wf5.py --streams S1 S2 S4   # Bruno's part only
python wf5_windows/figure2b.py                         # Figure 2b (needs the full run above)
```

| File | Content |
|---|---|
| `windows.py` | windowing, window comparisons, restabilisation (no model is run) |
| `analyze_wf5.py` | runs both analyses on the saved runs, writes `results/` |
| `figure2b.py` | Figure 2b |
| `results/table_wf5.md` | summary table (mean ± SD [95% bootstrap CI]) and paired tests |
| `results/windows_per_run.csv` | one row per run: first vs last, consecutive (min/mean), pre-drift vs last, top-k sets |
| `results/consecutive_pairs.csv` | every consecutive window pair, with a `crosses_drift` flag |
| `results/window_means.csv` | mean relevance per window, feature and seed (the data of Figure 2b) |
| `results/restab_per_run.csv` | threshold, trough, restabilisation times and WF2 ranking recovery per run |

## Data

R1 = G only. On every stream `ref` and `open` give identical G (checked by `analyze_wf5.py`), so the
`ref` runs are used.

| Stream | Runs | k |
|---|---|---|
| S3 | `results_wf5/s3_gauss_stationary/ref` (new, protocol config, seeds 0–9) | 3 |
| Elec2 | `results_relevance/wf1_elec2_nodate` (WF1's run without `date`, one order) | 3 |
| S1 / S2 | `results_wf2/s1_gauss_abrupt/ref`, `results_wf2/s2_gauss_gradual/ref` (seeds 0–9) | 3 |
| S4 | `results_wf2/s4_sea_drift/ref` (seeds 0–9) | 2 |

The S3 saved in main (`results_relevance/wf1_s3`) is the uniform-sum stream with k = 2 (S3u-like), not the
protocol's Gaussian S3, so S3 was run again from `configs/protocol/`. With the same seed, S3 and S1 have
identical G before the drift.

## Method

- **Windows:** non-overlapping 1,000-sample windows; window j holds the checkpoints whose newest sample is
  in [1,000 j, 1,000 (j + 1)). Window relevance = mean of G over the **clean** checkpoints in it (§4; same
  rule as `relevance/analysis.py`: buffer of 200 samples from one concept, no gradual-transition sample).
  The first window therefore averages t = 200…1,000.
- **Comparisons:** top-k Jaccard and weighted τ (`scipy.stats.weightedtau`) between consecutive windows,
  the first and the last window, and on drift streams the last pre-drift window and the last window.
  Consecutive pairs that overlap a drift or transition are kept apart (`crosses_drift`).
- **Last pre-drift window:** the last window ending at or before the transition start: 9,000–10,000 on
  S1/S4, 8,000–9,000 on S2 (its transition starts at 9,500).
- **Restabilisation:** s = weighted τ between G at t and at t − 200. Threshold = 10th percentile of s over
  clean pre-drift pairs. Trough = minimum of s within 1,000 samples after the drift (from the transition
  start to 1,000 samples after its end). If the trough is not below the threshold the ranking never
  destabilised (time 0). Otherwise restabilised = first checkpoint from the trough on where s ≥ threshold
  for 3 checkpoints in a row. Reported from the transition end (protocol convention, §10) and from the drift
  point (as `recovery_from_dp`); identical for abrupt drift. The protocol search starts after transition
  end; the midpoint search starts at the trough and retains signed negative offsets. Never restabilised → censored at the end.
- **Placebo:** the same rule on S3 with a fake drift at 10,000 gives the noise floor; paired with S1/S2 by
  seed (same pre-drift data). Wilcoxon signed-rank, rank-biserial, Holm over the four tests.

## Results

Full numbers: [`results/table_wf5.md`](results/table_wf5.md). Figure 2b:
[`fig2b_first_last_window.png`](fig2b_first_last_window.png).

**Window analysis.**
- S3: the top-3 set is {x0, x1, x2} in every window of every seed (Jaccard 1 for all consecutive pairs and
  first vs last). Weighted τ first vs last is only 0.59 ± 0.11: the irrelevant features sit at the 0.2
  floor, so their order is noise. Same pattern as H1 (τ-b well below 1 while the top-k set is stable).
- S1 and S2: first vs last Jaccard 0 in 10/10 seeds ({x0, x1, x2} → {x5, x6, x7}); last pre-drift vs last
  also 0. All consecutive pairs that do not overlap the drift have Jaccard 1, so the ranking changes once,
  at the drift, and is stable on both sides.
- S4 (negative control, §12): the top-2 set {x0, x1} before and after the drift is identical in 10/10
  seeds (Jaccard 1), as required.
- Elec2: first vs last window have the same top-3 {period, nswprice, nswdemand} (Jaccard 1, wτ 1), but
  4 of 9 consecutive pairs have Jaccard 0.5: the third place alternates between `period` and `day`, whose
  window means are nearly tied (0.21–0.32). nswprice is first in every window; nswdemand is second
  but its weight moves between 0.27 and 0.50 across windows (0.39 first, 0.50 last), so the bar difference
  in Figure 2b is within-run fluctuation, not a trend. So "first = last" on Elec2 hides within-run churn at rank 3. One real order, so no CI.

**Restabilisation (samples).**

| Stream | Trough below threshold | Descriptive time from transition end (protocol) | Descriptive time from drift point | WF2 ranking recovery (from dp) |
|---|---|---|---|---|
| S1 | 10/10 | 380 ± 95 | 380 ± 95 | 145 ± 16 |
| S2 | 10/10 | 50 ± 0 (floor) | 330 ± 48 | 190 ± 32 |
| S4 | 0/10 | 0 | 0 | — (trivial, §12) |
| S3 placebo | 5/10 | 295 ± 409 | 295 ± 409 | — |

- On S1/S2 the trough of weighted τ is far below the placebo's in 10/10 seeds (Wilcoxon p = 0.002,
  Holm 0.008, r = −1): the drift visibly destabilises the ranking.
- Restabilisation lags WF2's ranking recovery by 235 samples on S1 (150–450) and 140 on S2 (from dp).
  That is expected by construction: a lag-200 comparison is back to normal only once both snapshots come
  from the new concept, i.e. roughly ranking recovery + 200.
- The **time** itself is not distinguishable from the placebo (S1 p = 0.50, S2 p = 0.77, Holm 1.0): half of
  the stationary seeds also dip below a 10th-percentile threshold within 1,000 samples, and then need
  0–1,050 samples to "restabilise". The restabilisation time is therefore a noisy summary; the trough depth
  and the WF2 ranking recovery are the informative quantities.
- S2 measured from the transition end hits the one-checkpoint floor (50 in every seed), as WF2 found for
  ranking recovery (§14, 23 Nov); the from-drift-point value is the one to compare with S1.
- S4: weighted τ with 3 features takes only a few values; the trough equals the threshold (0.18) in every
  seed, so it never drops below it.

## Protocol decisions

The five methodological choices and their motivations, classification and known timing are recorded in
[PROTOCOL.md §14](../PROTOCOL.md#14-deviations). All remain **pending team approval**.
No approval or protocol-freeze history is inferred.

Restabilisation statuses distinguish `no_destabilisation` (no observed recovery event), `recovered`
and `censored`; `undefined` cases carry a reason and NaN times. No-drop descriptive times remain zero,
but `restabilised` is false. Descriptive means and paired time comparisons include these zeros;
observed recovery means include only recovered runs. Censoring limits are separate from observed times;
legacy `*_censored` fields retain their compatibility convention.

The protocol search starts strictly after transition end (S2 floor: 50 samples). The midpoint search
starts at the trough and may recover before the midpoint: signed negative `restab_from_dp` values
are retained. Each search has its own status in the CSV. This measures ranking consistency, distinct
from WF2 ranking recovery; the S2 transition-end floor does not establish faster recovery than S1.
