import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-)."""
    diff = x - y
    diff = diff[~np.isclose(diff, 0.0)]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    return (w_plus - w_minus) / (w_plus + w_minus)


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_vals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def compare_pair_across_streams(df: pd.DataFrame, var_a: str, var_b: str, metric: str = "accuracy", streams=None):
    if streams is None:
        streams = ["S1", "S2", "S3", "S4", "Elec2"]

    rows = []
    for ds in streams:
        sub = df[df["dataset"] == ds]
        a = sub[sub["variant"] == var_a].sort_values("seed")[metric].to_numpy()
        b = sub[sub["variant"] == var_b].sort_values("seed")[metric].to_numpy()
        assert len(a) == 10 and len(b) == 10, f"Expected 10 paired seeds for {ds} ({var_a} vs {var_b})"

        diff = a - b
        if np.allclose(diff, 0.0):
            stat, p_val, r_rb = 0.0, 1.0, 0.0
        else:
            stat, p_val = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            r_rb = rank_biserial_paired(a, b)

        scale = 100.0 if metric in {"accuracy", "macro_f1"} else 1.0
        rows.append({
            "comparison": f"{var_a} vs {var_b}",
            "metric": metric,
            "dataset": ds,
            "mean_A": round(a.mean() * scale, 4),
            "std_A": round(a.std(ddof=1) * scale, 4),
            "mean_B": round(b.mean() * scale, 4),
            "std_B": round(b.std(ddof=1) * scale, 4),
            "delta": round((a.mean() - b.mean()) * scale, 4),
            "W_stat": float(stat),
            "p_raw": round(float(p_val), 4),
            "r_rb": round(float(r_rb), 4),
        })

    res_df = pd.DataFrame(rows)
    res_df["p_holm"] = [round(p, 4) for p in holm_bonferroni(res_df["p_raw"].tolist())]
    return res_df


def run_all_q4_stats(results_csv: str = "wf4_baseline_q4/results/wf4_runs.csv"):
    df = pd.read_csv(results_csv)
    streams = ["S1", "S2", "S3", "S4", "Elec2"]

    # 1. Primary SQ4: FULL vs SHUFFLED_W (max_rules=None)
    sq4_df = compare_pair_across_streams(df, "FULL", "SHUFFLED_W", metric="accuracy", streams=streams)
    print("\n=== 1. PRIMARY SQ4: FULL vs. SHUFFLED_W (max_rules=None, Information Content) ===")
    print(sq4_df.to_string(index=False))

    # 2. Scale control on reference config: FULL_REF vs NO_W_REF (max_rules=5)
    scale_df = compare_pair_across_streams(df, "FULL_REF", "NO_W_REF", metric="accuracy", streams=streams)
    print("\n=== 2. SCALE CONTROL: FULL_REF vs. NO_W_REF (max_rules=5, Weight Scale Test) ===")
    print(scale_df.to_string(index=False))

    # 3. Incremental Feature-Distance Variants vs FULL (Accuracy & Latency)
    inc_dfs = []
    for inc_var in ["INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"]:
        for m in ["accuracy", "ms_per_sample"]:
            inc_dfs.append(compare_pair_across_streams(df, inc_var, "FULL", metric=m, streams=streams))
    inc_all = pd.concat(inc_dfs, ignore_index=True)
    print("\n=== 3. INCREMENTAL FEATURE-DISTANCE VARIANTS vs. FULL (Accuracy) ===")
    print(inc_all[inc_all["metric"] == "accuracy"].to_string(index=False))

    # 4. Peak Memory Profile (seed 0)
    mem_df = df[df["seed"] == 0].pivot(index="dataset", columns="variant", values="peak_mem_kb")
    print("\n=== 4. PEAK MEMORY (KB, tracemalloc on seed 0) ===")
    print(mem_df.round(2).to_string())

    # Save full statistical summary
    summary_out = Path("wf4_baseline_q4/results/sq4_summary_stats.csv")
    pd.concat([sq4_df, scale_df, inc_all], ignore_index=True).to_csv(summary_out, index=False)
    print(f"\nFull statistical summary saved to: {summary_out}")

    # Print ready-to-paste LaTeX row for Table 1 (SQ4)
    mean_delta = sq4_df["delta"].mean()
    sig_count = int((sq4_df["p_holm"] < 0.05).sum())
    med_r = sq4_df["r_rb"].median()
    print("\n=== LaTeX Row for Table 1 (SQ4) ===")
    print(
        f"SQ4 (Info. Content) & FULL vs. SHUFFLED\\_W ($K=\\infty$) & "
        f"$\\Delta\\text{{Acc}} = {mean_delta:+.2f}\\%$ & "
        f"{sig_count}/{len(streams)} sig. ($p_{{\\text{{Holm}}}} < 0.05$, median $r = {med_r:.2f}$) \\\\"
    )


if __name__ == "__main__":
    run_all_q4_stats()