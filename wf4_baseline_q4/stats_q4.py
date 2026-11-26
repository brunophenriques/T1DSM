import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-) e contagem de pares não nulos."""
    diff = x - y
    diff = diff[~np.isclose(diff, 0.0)]
    n_nonzero = int(len(diff))
    if n_nonzero == 0:
        return np.nan, 0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    r_rb = (w_plus - w_minus) / (w_plus + w_minus)
    return float(r_rb), n_nonzero


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    """Calcula Holm-Bonferroni sobre p-valores completos sem arredondamento prévio."""
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_vals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def validate_runs_dataframe(df: pd.DataFrame) -> None:
    """Valida métricas finitas (separadamente de max_rules=None), max_rules e seeds 0-9 completas."""
    metric_cols = [
        "n_samples", "accuracy", "macro_f1", "final_rules",
        "rules_created_id_approx", "rules_deleted_id_approx",
        "rule_turnover_id_approx", "ms_per_sample",
    ]
    assert not df[metric_cols].isna().any().any(), "Existem NaN nas métricas numéricas!"
    assert np.isfinite(df[metric_cols].to_numpy(dtype=float)).all(), "Existem valores não finitos nas métricas!"

    unbounded_vars = {"FULL", "SHUFFLED_W", "NO_W", "INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"}
    ref_vars = {"FULL_REF", "NO_W_REF"}

    for var in unbounded_vars:
        sub = df[df["variant"] == var]
        assert len(sub) > 0, f"Variante em falta: {var}"
        assert sub["max_rules"].isna().all(), f"Esperado max_rules=None para {var}, encontrado {sub['max_rules'].unique()}"

    for var in ref_vars:
        sub = df[df["variant"] == var]
        assert len(sub) > 0, f"Variante em falta: {var}"
        assert (sub["max_rules"] == 5).all(), f"Esperado max_rules=5 para {var}, encontrado {sub['max_rules'].unique()}"

    expected_seeds = list(range(10))
    for (ds, var), grp in df.groupby(["dataset", "variant"]):
        assert not grp["seed"].duplicated().any(), f"Seeds duplicadas em {ds} / {var}"
        assert sorted(grp["seed"].tolist()) == expected_seeds, f"Seeds incompletas em {ds} / {var}"


def evaluate_predictive_eligibility(df: pd.DataFrame, min_margin_pp: float = 2.0) -> pd.DataFrame:
    """Critério pré-especificado: FULL deve superar MajorityClass e NoChange em > min_margin_pp p.p."""
    rows = []
    for ds, grp in df.groupby("dataset", sort=False):
        acc_full = float(grp[grp["variant"] == "FULL"]["accuracy"].mean() * 100.0)
        acc_maj = float(grp[grp["variant"] == "MajorityClass"]["accuracy"].mean() * 100.0)
        acc_nc = float(grp[grp["variant"] == "NoChange"]["accuracy"].mean() * 100.0)
        best_trivial = max(acc_maj, acc_nc)
        margin_pp = acc_full - best_trivial
        rows.append({
            "dataset": ds,
            "full_acc_pct": acc_full,
            "majority_acc_pct": acc_maj,
            "nochange_acc_pct": acc_nc,
            "best_trivial_pct": best_trivial,
            "margin_vs_trivial_pp": margin_pp,
            "eligible_sq4": bool(margin_pp > min_margin_pp),
        })
    return pd.DataFrame(rows)


def compare_synthetic_paired(
    df: pd.DataFrame,
    family: str,
    var_a: str,
    var_b: str,
    expected_max_rules,
    metric: str = "accuracy",
    synth_streams=None,
    min_nonzero_for_rrb: int = 5,
) -> pd.DataFrame:
    if synth_streams is None:
        synth_streams = ["S1", "S2", "S3", "S4"]

    expected_seeds = list(range(10))
    rows = []
    raw_pvals = []

    for ds in synth_streams:
        sub_a = df[(df["dataset"] == ds) & (df["variant"] == var_a)][["dataset", "seed", "max_rules", metric]].copy()
        sub_b = df[(df["dataset"] == ds) & (df["variant"] == var_b)][["dataset", "seed", "max_rules", metric]].copy()

        if expected_max_rules is None:
            assert sub_a["max_rules"].isna().all() and sub_b["max_rules"].isna().all()
        else:
            assert (sub_a["max_rules"] == expected_max_rules).all() and (sub_b["max_rules"] == expected_max_rules).all()

        merged = pd.merge(
            sub_a, sub_b, on=["dataset", "seed"], suffixes=("_A", "_B"), validate="one_to_one"
        ).sort_values("seed")
        assert list(merged["seed"]) == expected_seeds

        a = merged[f"{metric}_A"].to_numpy(dtype=float)
        b = merged[f"{metric}_B"].to_numpy(dtype=float)
        diff = a - b

        if np.allclose(diff, 0.0):
            stat, p_val, r_rb, n_nonzero = 0.0, 1.0, np.nan, 0
        else:
            stat, p_val = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            r_rb_raw, n_nonzero = rank_biserial_paired(a, b)
            r_rb = r_rb_raw if n_nonzero >= min_nonzero_for_rrb else np.nan

        raw_pvals.append(float(p_val))
        scale = 100.0 if metric in {"accuracy", "macro_f1"} else 1.0
        rows.append({
            "family": family,
            "comparison": f"{var_a} vs {var_b}",
            "max_rules": "None" if expected_max_rules is None else str(expected_max_rules),
            "metric": metric,
            "dataset": ds,
            "mean_A": a.mean() * scale,
            "std_A": a.std(ddof=1) * scale,
            "mean_B": b.mean() * scale,
            "std_B": b.std(ddof=1) * scale,
            "delta": (a.mean() - b.mean()) * scale,
            "n_nonzero": n_nonzero,
            "W_stat": float(stat),
            "p_raw": float(p_val),
            "r_rb": r_rb,
        })

    adj_pvals = holm_bonferroni(raw_pvals)
    for row, p_adj in zip(rows, adj_pvals):
        row["p_holm"] = p_adj

    return pd.DataFrame(rows)


def summarize_elec2_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["dataset"] == "Elec2"].copy()
    rows = []
    for var, grp in sub.groupby("variant", sort=False):
        acc_pct = grp["accuracy"].to_numpy(dtype=float) * 100.0
        f1_pct = grp["macro_f1"].to_numpy(dtype=float) * 100.0
        lat = grp["ms_per_sample"].to_numpy(dtype=float)
        mem = grp["peak_mem_kb"].dropna().to_numpy(dtype=float)
        rows.append({
            "variant": var,
            "max_rules": "None" if pd.isna(grp["max_rules"].iloc[0]) else int(grp["max_rules"].iloc[0]),
            "acc_mean_pct": float(acc_pct.mean()),
            "acc_std_perm_pct": float(acc_pct.std(ddof=1)),
            "macro_f1_mean_pct": float(f1_pct.mean()),
            "final_rules": int(grp["final_rules"].iloc[0]),
            "turnover_id_approx": int(grp["rule_turnover_id_approx"].iloc[0]),
            "ms_per_sample_mean": float(lat.mean()),
            "ms_per_sample_std": float(lat.std(ddof=1)),
            "peak_mem_kb_seed0": float(mem[0]) if len(mem) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_secondary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo secundário completo (inclui NO_W sem limite de regras, macro-F1, regras, turnover e recursos)."""
    rows = []
    for (ds, var), grp in df.groupby(["dataset", "variant"], sort=False):
        mem = grp["peak_mem_kb"].dropna().to_numpy(dtype=float)
        rows.append({
            "dataset": ds,
            "variant": var,
            "max_rules": "None" if pd.isna(grp["max_rules"].iloc[0]) else int(grp["max_rules"].iloc[0]),
            "acc_mean_pct": float(grp["accuracy"].mean() * 100.0),
            "acc_std_pct": float(grp["accuracy"].std(ddof=1) * 100.0),
            "macro_f1_mean_pct": float(grp["macro_f1"].mean() * 100.0),
            "final_rules_mean": float(grp["final_rules"].mean()),
            "turnover_id_approx_mean": float(grp["rule_turnover_id_approx"].mean()),
            "ms_per_sample_mean": float(grp["ms_per_sample"].mean()),
            "peak_mem_kb_seed0": float(mem[0]) if len(mem) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def run_all_q4_stats(results_csv: str = "wf4_baseline_q4/results/wf4_runs.csv"):
    df = pd.read_csv(results_csv)
    validate_runs_dataframe(df)
    synth_streams = ["S1", "S2", "S3", "S4"]

    # 0. Critério pré-especificado de elegibilidade preditiva (> +2.0 p.p. acima de MajorityClass e NoChange)
    elig_df = evaluate_predictive_eligibility(df, min_margin_pp=2.0)
    n_eligible = int(elig_df["eligible_sq4"].sum())
    print("\n=== 0. PREDICTIVE ELIGIBILITY GATE (FULL > max(MajorityClass, NoChange) + 2.0 p.p.) ===")
    print(elig_df.round(4).to_string(index=False))

    # 1. Análise diagnóstica em S1-S4 (max_rules=None)
    sq4_df = compare_synthetic_paired(
        df, family="SQ4_diagnostic_all_synth", var_a="FULL", var_b="SHUFFLED_W", expected_max_rules=None, metric="accuracy", synth_streams=synth_streams
    )
    print("\n=== 1. DIAGNOSTIC SQ4 (Synthetic S1-S4, Paired by Seed, max_rules=None) ===")
    print(sq4_df.round(4).to_string(index=False))

    # 2. Controlos secundários: NO_W sem limite (max_rules=None) e controlo de escala (max_rules=5 + defaults)
    now_unbounded_df = compare_synthetic_paired(
        df, family="SQ4_no_w_unbounded", var_a="FULL", var_b="NO_W", expected_max_rules=None, metric="accuracy", synth_streams=synth_streams
    )
    scale_df = compare_synthetic_paired(
        df, family="SQ4_scale_control_ref", var_a="FULL_REF", var_b="NO_W_REF", expected_max_rules=5, metric="accuracy", synth_streams=synth_streams
    )
    print("\n=== 2. SECONDARY CONTROLS (NO_W max_rules=None & Scale Control max_rules=5 + defaults) ===")
    print(pd.concat([now_unbounded_df, scale_df], ignore_index=True).round(4).to_string(index=False))

    # 3. Estimadores Incrementais vs FULL (max_rules=None)
    inc_dfs = []
    for inc_var in ["INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"]:
        inc_dfs.append(
            compare_synthetic_paired(
                df, family=f"INC_{inc_var}_acc", var_a=inc_var, var_b="FULL", expected_max_rules=None, metric="accuracy", synth_streams=synth_streams
            )
        )
        inc_dfs.append(
            compare_synthetic_paired(
                df, family=f"INC_{inc_var}_lat", var_a=inc_var, var_b="FULL", expected_max_rules=None, metric="ms_per_sample", synth_streams=synth_streams
            )
        )
    inc_all = pd.concat(inc_dfs, ignore_index=True)

    elec2_df = summarize_elec2_descriptive(df)
    sec_df = summarize_secondary_metrics(df)

    print("\n=== 3. ELEC2 DESCRIPTIVE SUMMARY (Fixed Real Stream; NoChange=83.75% vs FULL=59.69%) ===")
    print(elec2_df.round(4).to_string(index=False))

    out_dir = Path("wf4_baseline_q4/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([sq4_df, now_unbounded_df, scale_df, inc_all], ignore_index=True).to_csv(out_dir / "sq4_summary_stats.csv", index=False)
    elec2_df.to_csv(out_dir / "elec2_descriptive_summary.csv", index=False)
    sec_df.to_csv(out_dir / "wf4_secondary_summary.csv", index=False)
    elig_df.to_csv(out_dir / "sq4_eligibility_check.csv", index=False)

    mean_delta = sq4_df["delta"].mean()
    sig_count = int((sq4_df["p_holm"] < 0.05).sum())
    elec_full = float(elec2_df.loc[elec2_df["variant"] == "FULL", "acc_mean_pct"].iloc[0])
    elec_nc = float(elec2_df.loc[elec2_df["variant"] == "NoChange", "acc_mean_pct"].iloc[0])

    print("\n=== LaTeX Row for Table 1 (SQ4) ===")
    print(
        f"SQ4 (Info. Content) & FULL vs. SHUFFLED\\_W ($K=\\infty$) & "
        f"{n_eligible}/{len(elig_df)} eligible ($\\Delta\\text{{Acc}}_{{\\text{{S1--S4}}}} = {mean_delta:+.2f}$ p.p.) & "
        f"{sig_count}/{len(synth_streams)} sig. ($p_{{\\text{{Holm}}}} < 0.05$; all $\\le$ trivial baselines, "
        f"e.g. Elec2: {elec_full:.2f}\\% vs. NoChange {elec_nc:.2f}\\%) \\\\"
    )


if __name__ == "__main__":
    run_all_q4_stats()