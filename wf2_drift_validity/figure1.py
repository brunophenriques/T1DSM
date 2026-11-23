from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_wf2/s1_gauss_abrupt"
OUTDIR = ROOT / "results_wf2/figures"

SEEDS = range(10)
DRIFT = 10000


def load_checkpoints(variant):
    runs = []

    for seed in SEEDS:
        path = (
            RESULTS / variant
            / f"seed_{seed}" / "checkpoints.npz"
        )

        with np.load(path) as data:
            runs.append({
                key: data[key].copy()
                for key in ["t", "G", "acc_win", "n_rules"]
            })

    t = runs[0]["t"]

    for run in runs:
        assert np.array_equal(run["t"], t)

    return {
        "t": t,
        "G": np.stack([r["G"] for r in runs]),
        "accuracy": np.stack(
            [r["acc_win"] for r in runs]
        ),
        "rules": np.stack(
            [r["n_rules"] for r in runs]
        ),
    }


def plot_mean_std(ax, t, values, label, color):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)

    ax.plot(
        t, mean,
        color=color,
        label=label,
        linewidth=1.5,
    )

    ax.fill_between(
        t,
        mean - std,
        mean + std,
        color=color,
        alpha=0.15,
        linewidth=0,
    )


def main():
    ref = load_checkpoints("ref")
    opened = load_checkpoints("open")

    t = ref["t"]

    # R1 depends on stream statistics, not rule count.
    assert np.allclose(
        ref["G"], opened["G"], equal_nan=True
    )

    relevance = ref["G"].mean(axis=0)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 8),
        sharex=True,
        layout="constrained",
        gridspec_kw={
            "height_ratios": [2, 1, 1]
        },
    )

    # A — Global feature relevance
    ax = axes[0]

    step = t[1] - t[0]

    heatmap = ax.imshow(
        relevance.T,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[
            t[0] - step / 2,
            t[-1] + step / 2,
            relevance.shape[1] - 0.5,
            -0.5,
        ],
    )

    ax.set_yticks(range(relevance.shape[1]))
    ax.set_yticklabels([
        f"x{i}" for i in range(relevance.shape[1])
    ])

    ax.set_ylabel("Feature")
    ax.set_title(
        "(a) Global feature relevance (R1)",
        loc="left",
    )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Mean relevance",
        pad=0.02,
    )

    # B — Predictive accuracy
    ax = axes[1]

    plot_mean_std(
        ax, t, ref["accuracy"],
        "Ref", "tab:blue",
    )

    plot_mean_std(
        ax, t, opened["accuracy"],
        "Open", "tab:orange",
    )

    ax.axhline(
        0.5,
        color="gray",
        linestyle=":",
        linewidth=1,
        label="Chance level",
    )

    ax.set_ylabel("Accuracy")
    ax.set_title(
        "(b) Windowed accuracy (500 samples)",
        loc="left",
    )

    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)

    # C — Rule count
    ax = axes[2]

    plot_mean_std(
        ax, t, ref["rules"],
        "Ref", "tab:blue",
    )

    plot_mean_std(
        ax, t, opened["rules"],
        "Open", "tab:orange",
    )

    ax.set_ylabel("Number of rules")
    ax.set_xlabel("Samples processed")
    ax.set_title(
        "(c) Rule-base evolution",
        loc="left",
    )

    ax.legend(loc="best", fontsize=8)

    # Common drift markers
    for ax in axes:
        ax.axvline(
            DRIFT,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
        )
        ax.set_xlim(0, 20000)

    for ax in axes[1:]:
        ax.grid(alpha=0.2)

    fig.suptitle(
        "S1 — Abrupt Feature-Relevance Drift",
        fontsize=14,
        fontweight="bold",
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTDIR / "figure1.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTDIR / "figure1.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure 1 saved to:", OUTDIR)


if __name__ == "__main__":
    main()