"""The synthetic-truth figure: bias per analytic choice, and editing against known ectopy.

Reads what `src/synthetic.py` wrote. Panel (a) is the whole grid scored against the Hurst exponent
the series were generated at, so each level is placed by how far it moves the estimate away from
the truth and in which direction. Panel (b) is the ectopic arm: beats injected at a known rate,
then the four editing levels scored against the clean series.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import AXES  # noqa: E402

FIGDIR = "figures"
OUT = f"{FIGDIR}/fig7_synthetic_bias.png"
INK, GREY, ACCENT = "#1a1a1a", "#9e9e9e", "#2e74b5"
EDIT_COLOURS = {"none": "#c0392b", "pct20_drop": "#2e74b5", "pct20_interp": "#7f7f7f",
                "mad4_drop": "#2e8b57"}


def main():
    grid = pd.read_parquet("results/synthetic_grid.parquet")
    ect = pd.read_csv("results/synthetic_ectopic.csv")

    axes = [a for a in AXES if a in grid.columns]
    rows, ylabels, seps = [], [], []
    for ax_name in axes:
        for level, g in grid.groupby(ax_name):
            b = g.bias.to_numpy(dtype=float)
            b = b[np.isfinite(b)]
            rows.append((np.median(b), np.percentile(b, 5), np.percentile(b, 95)))
            ylabels.append(f"{ax_name} = {level}")
        seps.append(len(rows) - 0.5)
    seps.pop()

    fig = plt.figure(figsize=(7.2, 3.2 + 0.16 * len(rows)))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16 * len(rows), 2.0], hspace=0.28)

    ax0 = fig.add_subplot(gs[0])
    y = np.arange(len(rows))
    med = np.array([r[0] for r in rows])
    lo = np.array([r[1] for r in rows])
    hi = np.array([r[2] for r in rows])
    ax0.axvline(0, color=INK, lw=1.0, zorder=1)
    ax0.hlines(y, lo, hi, color=GREY, lw=2.2, zorder=2)
    ax0.scatter(med, y, s=16, color=ACCENT, zorder=3)
    for s in seps:
        ax0.axhline(s, color="#e8e8e8", lw=0.8)
    ax0.set_yticks(y)
    ax0.set_yticklabels(ylabels, fontsize=6.0)
    ax0.tick_params(axis="y", length=0)
    ax0.invert_yaxis()
    ax0.set_xlabel(r"bias in recovered $\alpha$ (estimate $-$ true Hurst exponent)", fontsize=9)
    ax0.tick_params(axis="x", labelsize=8)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.text(-0.17, 1.02, "(a)", transform=ax0.transAxes, fontsize=10, fontweight="bold",
             va="bottom")

    ax1 = fig.add_subplot(gs[1])
    for level, g in ect.groupby("ectopic"):
        m = g.groupby("rate").bias.apply(lambda s: s.abs().mean())
        ax1.plot(m.index * 100, m.to_numpy(), marker="o", ms=4, lw=1.4,
                 color=EDIT_COLOURS.get(level, INK), label=level)
    ax1.set_xlabel("injected ectopic beats (% of intervals)", fontsize=9)
    ax1.set_ylabel(r"mean $|$bias$|$ in $\alpha$", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=7.5, frameon=False, title="ectopic handling", title_fontsize=7.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(-0.17, 1.02, "(b)", transform=ax1.transAxes, fontsize=10, fontweight="bold",
             va="bottom")

    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUT.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
