import numpy as np
import matplotlib.pyplot as plt


def plot_rule_growth(
    checkpoints,
    active_rules,
    created_cum,
    deleted_cum,
    out_path: str,
    title: str = "UnimNeuron Rule-Base Evolution Over Time",
    drift_points=None,
):
    """
    Plots active rules alongside cumulative created and deleted/merged rule counts
    (object-ID change approximation).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(checkpoints, active_rules, label="Active rules", color="#1f77b4", linewidth=2.2)
    ax.plot(checkpoints, created_cum, label="Cumulative created (ID approx.)", color="#2ca02c", linestyle="--", linewidth=1.6)
    ax.plot(checkpoints, deleted_cum, label="Cumulative deleted / merged (ID approx.)", color="#d62728", linestyle=":", linewidth=1.8)

    if drift_points:
        for idx, dp in enumerate(drift_points):
            ax.axvline(
                dp,
                color="#555555",
                linestyle="-.",
                alpha=0.75,
                label="Concept drift" if idx == 0 else None,
            )

    ax.set_xlabel("Stream observation ($t$)")
    ax.set_ylabel("Number of rules")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_r1_heatmap(
    checkpoints,
    weight_matrix: np.ndarray,
    feature_names=None,
    out_path: str = "wf4_baseline_q4/results/demo_r1_heatmap.png",
    title: str = "Feature Relevance Trajectory Heatmap (R1)",
    drift_points=None,
):
    W = np.asarray(weight_matrix, dtype=float)
    n_steps, n_features = W.shape
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(n_features)]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(W.T, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)

    tick_idx = np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(checkpoints[i]) for i in tick_idx])
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names)

    if drift_points:
        for dp in drift_points:
            pos = int(np.argmin(np.abs(np.asarray(checkpoints) - dp)))
            ax.axvline(pos, color="white", linestyle="--", linewidth=1.6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Feature relevance weight ($w_j$)")
    ax.set_xlabel("Stream observation ($t$)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)