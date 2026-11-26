import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from river import datasets
from river.datasets import synth

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from wf4_baseline_q4.wrappers import (
    AblationWeightWrapper,
    MajorityClassBaseline,
    NoChangeBaseline,
    attach_rule_event_counter,
)
from wf4_baseline_q4.incremental_weights import ENFUnimNeuronIncremental
from wf4_baseline_q4.plots import plot_r1_heatmap, plot_rule_growth


DRIFT_GROUND_TRUTH = {
    "S1": {"drift_t": 2000, "pre_idx": [0, 1], "post_idx": [4, 5]},
    "S2": {"drift_t": 2500, "pre_idx": [0, 1], "post_idx": [2, 3]},
    "S3": {"drift_t": 2500, "pre_idx": [0], "post_idx": [0, 7]},     # age -> (age, salary)
    "S4": {"drift_t": 2500, "pre_idx": [0, 1], "post_idx": [0, 1]},  # sea_att1, sea_att2 vs 4 noise features
}


class FastCausalMinMaxScaler:
    """Normalização MinMax online estritamente causal em NumPy (usa apenas x_0 ... x_{t-1})."""

    def __init__(self, n_features: int):
        self.n_features = int(n_features)
        self.min_val = np.full(self.n_features, np.inf, dtype=float)
        self.max_val = np.full(self.n_features, -np.inf, dtype=float)
        self.seen = 0

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if self.seen == 0:
            return np.zeros(self.n_features, dtype=float)
        denom = self.max_val - self.min_val
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        return np.where(denom > 0.0, (x - self.min_val) / safe_denom, 0.0)

    def learn_one(self, x: np.ndarray) -> None:
        self.min_val = np.minimum(self.min_val, x)
        self.max_val = np.maximum(self.max_val, x)
        self.seen += 1


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def load_stream(name: str, seed: int, n_samples_override: int | None = None):
    rng = np.random.default_rng(seed)

    if name in {"S1", "S1_abrupt_switch"}:
        n = n_samples_override or 4000
        d = 8
        dp = n // 2
        X = rng.normal(0.0, 1.0, size=(n, d))
        y = np.zeros(n, dtype=int)
        y[:dp] = ((X[:dp, 0] + 0.9 * X[:dp, 1]) > 0.0).astype(int)
        y[dp:] = ((X[dp:, 4] - 0.9 * X[dp:, 5]) > 0.0).astype(int)
        return X, y, 2, [dp], [f"f{j}" for j in range(d)]

    if name == "S2":
        n = n_samples_override or 4000
        d = 8
        t_start, t_end = int(0.375 * n), int(0.625 * n)
        X = rng.normal(0.0, 1.0, size=(n, d))
        y = np.zeros(n, dtype=int)
        for t in range(n):
            if t < t_start:
                p_new = 0.0
            elif t >= t_end:
                p_new = 1.0
            else:
                p_new = (t - t_start) / float(t_end - t_start)
            use_new = rng.random() < p_new
            score = (X[t, 2] + X[t, 3]) if use_new else (X[t, 0] + X[t, 1])
            y[t] = int(score > 0.0)
        return X, y, 2, [t_start, t_end], [f"f{j}" for j in range(d)]

    if name == "S3":
        n = n_samples_override or 5000
        dp = n // 2
        s_pre = synth.Agrawal(classification_function=0, seed=seed)
        s_post = synth.Agrawal(classification_function=2, seed=seed + 100)
        X_list, y_list, keys = [], [], None
        for xd, label in s_pre.take(dp):
            if keys is None:
                keys = sorted(xd.keys())
            X_list.append([float(xd[k]) for k in keys])
            y_list.append(int(label))
        for xd, label in s_post.take(n - dp):
            X_list.append([float(xd[k]) for k in keys])
            y_list.append(int(label))
        return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int), 2, [dp], list(keys)

    if name == "S4":
        n = n_samples_override or 5000
        dp = n // 2
        s_pre = synth.SEA(variant=0, noise=0.1, seed=seed)
        s_post = synth.SEA(variant=2, noise=0.1, seed=seed + 100)
        X_list, y_list, keys = [], [], None
        for xd, label in s_pre.take(dp):
            if keys is None:
                keys = sorted(xd.keys())
            base = [float(xd[k]) for k in keys]
            X_list.append(base + rng.uniform(0.0, 10.0, size=3).tolist())
            y_list.append(int(label))
        for xd, label in s_post.take(n - dp):
            base = [float(xd[k]) for k in keys]
            X_list.append(base + rng.uniform(0.0, 10.0, size=3).tolist())
            y_list.append(int(label))
        fnames = [f"sea_{k}" for k in keys] + ["noise_0", "noise_1", "noise_2"]
        return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int), 2, [dp], fnames

    if name == "Elec2":
        n = n_samples_override or 10000
        stream = datasets.Elec2()
        X_list, y_list, keys = [], [], None
        for xd, label in stream.take(n):
            if keys is None:
                keys = sorted(xd.keys())
            X_list.append([float(xd[k]) for k in keys])
            y_list.append(int(bool(label)))
        return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int), 2, [], list(keys)

    raise ValueError(f"Stream desconhecida: {name}")


def build_model(variant: str, n_features: int, n_classes: int, seed: int):
    if variant == "FULL":
        base = attach_rule_event_counter(
            EvolvingNeuroFuzzyUnimSafe(n_features=n_features, n_classes=n_classes, max_rules=None, random_state=42)
        )
        return AblationWeightWrapper(base, mode="FULL", seed=seed), None
    if variant == "SHUFFLED_W":
        base = attach_rule_event_counter(
            EvolvingNeuroFuzzyUnimSafe(n_features=n_features, n_classes=n_classes, max_rules=None, random_state=42)
        )
        return AblationWeightWrapper(base, mode="SHUFFLED_W", seed=seed), None
    if variant == "NO_W":
        base = attach_rule_event_counter(
            EvolvingNeuroFuzzyUnimSafe(n_features=n_features, n_classes=n_classes, max_rules=None, random_state=42)
        )
        return AblationWeightWrapper(base, mode="NO_W", seed=seed), None
    if variant == "FULL_REF":
        base = attach_rule_event_counter(
            EvolvingNeuroFuzzyUnimSafe(n_features=n_features, n_classes=n_classes, max_rules=5, random_state=42)
        )
        return AblationWeightWrapper(base, mode="FULL", seed=seed), 5
    if variant == "NO_W_REF":
        base = attach_rule_event_counter(
            EvolvingNeuroFuzzyUnimSafe(n_features=n_features, n_classes=n_classes, max_rules=5, random_state=42)
        )
        return AblationWeightWrapper(base, mode="NO_W", seed=seed), 5
    if variant in {"INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"}:
        model = attach_rule_event_counter(
            ENFUnimNeuronIncremental(
                n_features=n_features,
                n_classes=n_classes,
                inc_method=variant,
                alpha_inc=0.05,
                update_rule_weights=False,
                max_rules=None,
                random_state=42,
            )
        )
        return model, None
    if variant == "MajorityClass":
        return MajorityClassBaseline(n_classes=n_classes), None
    if variant == "NoChange":
        return NoChangeBaseline(default_class=0), None

    raise ValueError(f"Variante desconhecida: {variant}")


def evaluate_single_run(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    variant: str,
    seed: int,
    checkpoint_interval: int = 50,
    track_trajectory: bool = False,
    use_causal_scaler: bool = True,
    first_step_zero_pred: bool = False,
):
    n_samples, n_features = X.shape
    model, max_rules_val = build_model(variant, n_features, n_classes, seed)
    scaler = FastCausalMinMaxScaler(n_features) if use_causal_scaler else None
    preds = np.zeros(n_samples, dtype=int)

    prev_rule_ids = set()
    created_cum = 0
    deleted_cum = 0
    checkpoints, active_hist, created_hist, deleted_hist, weights_hist = [], [], [], [], []

    t0 = time.perf_counter()

    for t in range(n_samples):
        xt = X[t]
        yt = int(y[t])

        xt_scaled = scaler.transform_one(xt) if scaler is not None else xt
        if first_step_zero_pred and t == 0:
            preds[t] = 0
        else:
            preds[t] = int(model.predict(xt_scaled.reshape(1, -1))[0])

        if scaler is not None:
            scaler.learn_one(xt)
        model.partial_fit(xt_scaled.reshape(1, -1), np.array([yt], dtype=int))

        if hasattr(model, "rules"):
            curr_ids = {id(r) for r in model.rules}
            created_cum += len(curr_ids - prev_rule_ids)
            deleted_cum += len(prev_rule_ids - curr_ids)
            prev_rule_ids = curr_ids

            if track_trajectory and (t + 1) % checkpoint_interval == 0:
                checkpoints.append(t + 1)
                active_hist.append(model.n_rules_)
                created_hist.append(created_cum)
                deleted_hist.append(deleted_cum)
                weights_hist.append(model.current_feature_weights.copy())

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    acc = float(np.mean(preds == y))
    macro_f1 = compute_macro_f1(y, preds, n_classes)

    c_exact = int(getattr(model, "rules_created_exact", created_cum))
    e_exact = int(getattr(model, "rules_evicted_exact", deleted_cum))
    m_exact = int(getattr(model, "rules_merged_exact", 0))

    metrics = {
        "max_rules": "None" if max_rules_val is None else str(max_rules_val),
        "n_samples": n_samples,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "final_rules": int(getattr(model, "n_rules_", 0)),
        "rules_created_id_approx": int(created_cum),
        "rules_deleted_id_approx": int(deleted_cum),
        "rule_turnover_id_approx": int(created_cum + deleted_cum),
        "rules_created_exact": c_exact,
        "rules_evicted_exact": e_exact,
        "rules_merged_exact": m_exact,
        "rule_turnover_exact": int(c_exact + e_exact + m_exact),
        "ms_per_sample": round(elapsed_ms / max(n_samples, 1), 4),
    }

    traj = None
    if track_trajectory and checkpoints:
        traj = {
            "checkpoints": checkpoints,
            "active_rules": active_hist,
            "created_cum": created_hist,
            "deleted_cum": deleted_hist,
            "weights": np.asarray(weights_hist, dtype=float),
        }

    return metrics, traj


def measure_peak_memory_kb(X: np.ndarray, y: np.ndarray, n_classes: int, variant: str, seed: int = 0) -> float:
    n_samples, n_features = X.shape
    tracemalloc.start()
    model, _ = build_model(variant, n_features, n_classes, seed)
    scaler = FastCausalMinMaxScaler(n_features)
    for t in range(n_samples):
        xt = X[t]
        yt = int(y[t])
        xt_scaled = scaler.transform_one(xt)
        _ = model.predict(xt_scaled.reshape(1, -1))
        scaler.learn_one(xt)
        model.partial_fit(xt_scaled.reshape(1, -1), np.array([yt], dtype=int))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak_bytes / 1024.0, 2)


def _eval_task(task):
    ds, var, seed = task
    X, y, n_classes, _, _ = load_stream(ds, seed=seed)
    m, _ = evaluate_single_run(X, y, n_classes, variant=var, seed=seed, track_trajectory=False)
    return ds, var, seed, m


def _topk_precision(w: np.ndarray, true_idx: list[int]) -> float:
    k = len(true_idx)
    top_k = set(np.argsort(-np.asarray(w, dtype=float))[:k].tolist())
    return len(top_k.intersection(true_idx)) / float(k)


def _drift_task(task):
    ds, var, seed = task
    gt = DRIFT_GROUND_TRUTH[ds]
    drift_t = gt["drift_t"]
    pre_idx = gt["pre_idx"]
    post_idx = gt["post_idx"]

    X, y, n_classes, _, _ = load_stream(ds, seed=seed)
    _, traj = evaluate_single_run(
        X, y, n_classes, variant=var, seed=seed, checkpoint_interval=50, track_trajectory=True
    )
    cps = np.asarray(traj["checkpoints"], dtype=int)
    W = traj["weights"]

    pre_mask = cps == (drift_t - 50)
    post50_mask = cps == (drift_t + 50)
    post500_mask = cps == (drift_t + 500)

    prec_pre = _topk_precision(W[pre_mask][0], pre_idx)
    prec_post50 = _topk_precision(W[post50_mask][0], post_idx)
    prec_post500 = _topk_precision(W[post500_mask][0], post_idx)
    prec_final = _topk_precision(W[-1], post_idx)

    # Checkpoints após drift até 3 checkpoints consecutivos com top-k precision == 1.0
    post_cps = cps[cps > drift_t]
    post_W = W[cps > drift_t]
    post_precs = [_topk_precision(w_row, post_idx) for w_row in post_W]
    recovery_obs = np.nan
    for i in range(len(post_precs) - 2):
        if post_precs[i] == 1.0 and post_precs[i + 1] == 1.0 and post_precs[i + 2] == 1.0:
            recovery_obs = float(post_cps[i] - drift_t)
            break

    return {
        "dataset": ds,
        "variant": var,
        "seed": seed,
        "k_pre": len(pre_idx),
        "k_post": len(post_idx),
        "topk_prec_pre_drift": prec_pre,
        "topk_prec_post_50": prec_post50,
        "topk_prec_post_500": prec_post500,
        "topk_prec_final": prec_final,
        "recovery_obs_3consec": recovery_obs,
    }


def run_drift_ranking_study(out_csv: str = "wf4_baseline_q4/results/sq4_drift_ranking_recovery.csv", workers: int = 1):
    t0 = time.perf_counter()
    streams = ["S1", "S2", "S3", "S4"]
    variants = ["FULL", "INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF"]
    seeds = list(range(10))
    tasks = [(ds, var, s) for ds in streams for var in variants for s in seeds]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_drift_task, tasks))
    else:
        rows = [_drift_task(t) for t in tasks]

    df_raw = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(out_path, index=False)

    summary = (
        df_raw.groupby(["dataset", "variant"], sort=False)
        .agg(
            topk_pre_mean=("topk_prec_pre_drift", "mean"),
            topk_post50_mean=("topk_prec_post_50", "mean"),
            topk_post500_mean=("topk_prec_post_500", "mean"),
            topk_final_mean=("topk_prec_final", "mean"),
            recovered_seeds=("recovery_obs_3consec", lambda s: int(s.notna().sum())),
            median_recovery_obs=("recovery_obs_3consec", "median"),
        )
        .reset_index()
    )
    summary_path = out_path.with_name("sq4_drift_ranking_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== POST-DRIFT TOP-K FEATURE RANKING RECOVERY (S1-S4, 10 seeds) ===")
    print(summary.round(4).to_string(index=False))
    print(f"Saved -> {out_path} and {summary_path} in {time.perf_counter() - t0:.2f}s")


def run_refresh_variants(
    variants_to_refresh: list[str],
    out_csv: str = "wf4_baseline_q4/results/wf4_runs.csv",
    workers: int = 1,
):
    t0 = time.perf_counter()
    out_path = Path(out_csv)
    df_existing = pd.read_csv(out_path)
    # Preserva max_rules como string ("None" ou "5") para compatibilidade exata com read_csv
    df_existing["max_rules"] = df_existing["max_rules"].fillna("None").astype(str)
    for col, default_val in [
        ("rules_created_exact", df_existing["rules_created_id_approx"]),
        ("rules_evicted_exact", df_existing["rules_deleted_id_approx"]),
        ("rules_merged_exact", 0),
        ("rule_turnover_exact", df_existing["rule_turnover_id_approx"]),
    ]:
        if col not in df_existing.columns:
            df_existing[col] = default_val

    streams = ["S1", "S2", "S3", "S4", "Elec2"]
    seeds = list(range(10))

    mem_map = {}
    for ds in streams:
        X0, y0, n_cls0, _, _ = load_stream(ds, seed=0)
        for var in variants_to_refresh:
            mem_map[(ds, var)] = measure_peak_memory_kb(X0, y0, n_cls0, variant=var, seed=0)

    tasks = [(ds, var, s) for ds in streams for var in variants_to_refresh for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_eval_task, tasks))
    else:
        results = [_eval_task(t) for t in tasks]

    new_rows = []
    for ds, var, seed, m in results:
        new_rows.append({
            "dataset": ds,
            "variant": var,
            "seed": seed,
            **m,
            "peak_mem_kb": mem_map[(ds, var)] if seed == 0 else np.nan,
        })

    df_kept = df_existing[~df_existing["variant"].isin(variants_to_refresh)].copy()
    df_merged = pd.concat([df_kept, pd.DataFrame(new_rows)], ignore_index=True)

    variant_order = [
        "FULL", "SHUFFLED_W", "NO_W", "FULL_REF", "NO_W_REF",
        "INC_FISHER_L2", "INC_L1_CENTROID", "INC_RELIEFF", "MajorityClass", "NoChange",
    ]
    ds_order = ["S1", "S2", "S3", "S4", "Elec2"]
    df_merged["ds_rank"] = df_merged["dataset"].map({d: i for i, d in enumerate(ds_order)})
    df_merged["var_rank"] = df_merged["variant"].map({v: i for i, v in enumerate(variant_order)})
    df_merged = df_merged.sort_values(["ds_rank", "seed", "var_rank"]).drop(columns=["ds_rank", "var_rank"])

    df_merged.to_csv(out_path, index=False)
    print(f"Refreshed {variants_to_refresh} in {out_path} ({len(df_merged)} rows) in {time.perf_counter() - t0:.2f}s")


def run_demo(config_path: str):
    t_start = time.perf_counter()
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    stream_name = cfg.get("stream", "S1_abrupt_switch")
    n_samples = int(cfg.get("n_samples", 1500))
    seed = int(cfg.get("seed", 0))
    interval = int(cfg.get("checkpoint_interval", 25))
    variants = cfg.get("variants", ["FULL", "SHUFFLED_W", "NO_W", "INC_FISHER_L2"])

    X, y, n_classes, drift_points, feature_names = load_stream(stream_name, seed=seed, n_samples_override=n_samples)
    print(f"=== T17 Demo ({stream_name}, N={n_samples}, seed={seed}) ===", flush=True)

    full_traj = None
    for var in variants:
        m, traj = evaluate_single_run(
            X, y, n_classes, variant=var, seed=seed,
            checkpoint_interval=interval, track_trajectory=(var == "FULL"),
        )
        if var == "FULL":
            full_traj = traj
        print(
            f"  {var:<16} | Acc={m['accuracy']:.4f} | MacroF1={m['macro_f1']:.4f} | "
            f"Rules={m['final_rules']:>3} | Turnover={m['rule_turnover_exact']:>3} | "
            f"{m['ms_per_sample']:.3f} ms/sample",
            flush=True,
        )

    out_cfg = cfg.get("outputs", {})
    heatmap_path = Path(out_cfg.get("heatmap_path", "wf4_baseline_q4/results/demo_r1_heatmap.png"))
    growth_path = Path(out_cfg.get("rule_growth_path", "wf4_baseline_q4/results/demo_rule_growth.png"))
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    growth_path.parent.mkdir(parents=True, exist_ok=True)

    if full_traj is not None:
        plot_r1_heatmap(
            full_traj["checkpoints"],
            full_traj["weights"],
            feature_names=feature_names,
            out_path=str(heatmap_path),
            title=f"R1 Feature-Relevance Trajectory ({stream_name})",
            drift_points=drift_points,
        )
        plot_rule_growth(
            full_traj["checkpoints"],
            full_traj["active_rules"],
            full_traj["created_cum"],
            full_traj["deleted_cum"],
            out_path=str(growth_path),
            title=f"Rule-Base Growth & Turnover ({stream_name})",
            drift_points=drift_points,
        )
        print(f"Saved R1 heatmap -> {heatmap_path}")
        print(f"Saved rule growth -> {growth_path}")

    print(f"Demo finished in {time.perf_counter() - t_start:.2f}s")


def run_full_sq4(out_csv: str = "wf4_baseline_q4/results/wf4_runs.csv"):
    streams = ["S1", "S2", "S3", "S4", "Elec2"]
    seeds = list(range(10))
    variants = [
        "FULL",
        "SHUFFLED_W",
        "NO_W",
        "FULL_REF",
        "NO_W_REF",
        "INC_FISHER_L2",
        "INC_L1_CENTROID",
        "INC_RELIEFF",
        "MajorityClass",
        "NoChange",
    ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset", "variant", "seed", "max_rules", "n_samples",
        "accuracy", "macro_f1", "final_rules",
        "rules_created_id_approx", "rules_deleted_id_approx", "rule_turnover_id_approx",
        "rules_created_exact", "rules_evicted_exact", "rules_merged_exact", "rule_turnover_exact",
        "ms_per_sample", "peak_mem_kb",
    ]

    t_total = time.perf_counter()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in streams:
            print(f"\n=== Dataset: {ds} (seeds 0-9) ===", flush=True)
            X0, y0, n_cls0, _, _ = load_stream(ds, seed=0)
            mem_by_variant = {
                var: measure_peak_memory_kb(X0, y0, n_cls0, variant=var, seed=0)
                for var in variants
            }

            for seed in seeds:
                t_seed = time.perf_counter()
                X, y, n_classes, _, _ = load_stream(ds, seed=seed)
                for var in variants:
                    m, _ = evaluate_single_run(
                        X, y, n_classes, variant=var, seed=seed, track_trajectory=False
                    )
                    row = {
                        "dataset": ds,
                        "variant": var,
                        "seed": seed,
                        **m,
                        "peak_mem_kb": mem_by_variant[var] if seed == 0 else np.nan,
                    }
                    writer.writerow(row)
                f.flush()
                print(f"  Completed seed {seed}/9 ({time.perf_counter() - t_seed:.1f}s)", flush=True)

    print(f"\nAll SQ4 & incremental runs saved to {out_path} in {time.perf_counter() - t_total:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workflow 4 Runner: SQ4 & Incremental Feature Distances")
    parser.add_argument("--config", type=str, default=None, help="Path to demo JSON config")
    parser.add_argument("--full-sq4", action="store_true", help="Run full S1-S4 + Elec2 benchmark (10 seeds)")
    parser.add_argument(
        "--refresh-variants",
        nargs="+",
        default=None,
        help="Re-run only specified variants and update existing wf4_runs.csv",
    )
    parser.add_argument(
        "--drift-ranking",
        action="store_true",
        help="Compute post-drift Top-k feature ranking recovery across S1-S4 (10 seeds)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for refresh/drift")
    parser.add_argument("--out-csv", type=str, default="wf4_baseline_q4/results/wf4_runs.csv")
    args = parser.parse_args()

    if args.config:
        run_demo(args.config)
    elif args.full_sq4:
        run_full_sq4(out_csv=args.out_csv)
    elif args.refresh_variants:
        run_refresh_variants(args.refresh_variants, out_csv=args.out_csv, workers=args.workers)
    elif args.drift_ranking:
        run_drift_ranking_study(workers=args.workers)
    else:
        parser.print_help()