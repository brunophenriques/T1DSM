#!/usr/bin/env python
"""Figure 2b: first vs last 1,000-sample window, mean R1 relevance per feature (S3 | S1 | Elec2).

    python wf5_windows/analyze_wf5.py      # writes results/window_means.csv first
    python wf5_windows/figure2b.py

Bars = mean over seeds of the window's mean relevance; whiskers = SD over seeds (Elec2: one run, none).
Grey bands mark the truly relevant features of the first window, outlined bands those of the last.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "wf5_windows" / "results"
FIRST, LAST = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated: CVD ΔE 24.7, contrast >= 3:1)
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
PANELS = [("S3", "S3 stationary (Gaussian)", [0, 1, 2], [0, 1, 2]),
          ("S1", "S1 abrupt swap at 10,000", [0, 1, 2], [5, 6, 7]),
          ("Elec2", "Elec2 (real order, no date)", None, None)]


def main():
    m = pd.read_csv(RES / "window_means.csv")
    per_run = pd.read_csv(RES / "windows_per_run.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "font.family": "DejaVu Sans"})
    widths = [10, 10, 7]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=widths), sharey=True)
    for ax, (name, title, rel_first, rel_last) in zip(axes, PANELS):
        g = m[m.stream == name]; pr = per_run[per_run.stream == name]
        feats = list(dict.fromkeys(g.feature))
        x = np.arange(len(feats)); w = 0.36
        for rel, style in ((rel_first, dict(color=BAND, lw=0)), (rel_last, dict(fill=False, ec=INK2, lw=0.8, ls=":"))):
            for i in rel or []:
                ax.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1.08, zorder=0, **style))
        for off, col, which, lab in ((-w / 2 - 0.01, FIRST, "first", "first window"),
                                     (w / 2 + 0.01, LAST, "last", "last window")):
            rows = [g[(g.seed == s) & (g.window == int(pr[pr.seed == s][which].iloc[0]))] for s in pr.seed]
            vals = np.array([[r[r.feature == f].relevance.iloc[0] for f in feats] for r in rows])
            mu = vals.mean(axis=0); sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else None
            ax.bar(x + off, mu, w, color=col, label=lab, zorder=2, edgecolor="none")
            if sd is not None:
                ax.errorbar(x + off, mu, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
        jac = pr.first_last_jaccard.mean(); wt = pr.first_last_wtau
        stat = f"Jaccard {jac:.2f} · wτ {wt.mean():.2f}" + (f" ± {wt.std(ddof=1):.2f}" if len(wt) > 1 else "")
        ax.set_title(f"{title}\n{stat}", fontsize=9, color=INK, loc="left")
        ax.set_xticks(x, feats, rotation=0 if name != "Elec2" else 35, ha="center" if name != "Elec2" else "right")
        ax.set_xlim(-0.6, len(feats) - 0.4); ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("mean relevance G in window")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Rectangle((0, 0), 1, 1, color=BAND), plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK2, ls=":")]
    l += ["truly relevant (first window)", "truly relevant (last window)"]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "wf5_windows" / f"fig2b_first_last_window.{ext}", dpi=200, bbox_inches="tight")
    print("wrote wf5_windows/fig2b_first_last_window.{png,pdf}")


if __name__ == "__main__":
    main()
