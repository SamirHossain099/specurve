"""Figure 1 -- the specification curve.

Upper panel: every specification's effect size, sorted. Lower panel: a dot matrix showing
which analytic choices are active in each region, so the reader can see *which* decision
moves the estimate.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from analyze import AXES  # noqa: E402

# Figures live in figures/, results/ holds data. Same split as project 05: a directory that mixes
# 4 MB parquet grids with the PNGs a journal wants uploaded is one a submission has to be sorted
# out of by hand.
FIGDIR = "figures"
BS = chr(92)      # a literal backslash, kept out of source so heredocs cannot mangle it


def spec_curve(eff, out=f"{FIGDIR}/fig1_spec_curve.png",
               metric_label=r"Cohen's $d$ (young $-$ elderly)", n_bins=60):
    """The curve, with the lower panel as an enrichment map rather than a dot matrix.

    The dot matrix this replaced drew one mark per specification per level: 9,216 marks a row,
    which fills the panel and hides the very structure it exists to show. Here the ordered
    specifications are binned, and each cell is how often a level occurs in that bin relative to
    how often it occurs overall. A level that is indifferent to the effect size sits at 1 and
    prints white; a level that concentrates at one end of the curve prints as a band.
    """
    import matplotlib.colors as mcolors

    eff = eff.sort_values("d").reset_index(drop=True)
    n = len(eff)
    x = np.arange(n)
    d = eff.d.to_numpy(dtype=float)
    sig = (eff.p < 0.05).to_numpy()

    levels = [(ax, sorted(pd.unique(eff[ax]), key=str)) for ax in AXES]
    n_rows = sum(len(v) for _, v in levels)

    fig = plt.figure(figsize=(7.2, 3.0 + 0.135 * n_rows))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 0.135 * n_rows], hspace=0.05)

    # ---- (a) the curve --------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.axhline(0, color="black", lw=0.9, zorder=1)
    ax0.scatter(x[~sig], d[~sig], s=2.0, c="#bdbdbd", linewidths=0, label="p $\\geq$ 0.05", zorder=2)
    ax0.scatter(x[sig], d[sig], s=2.0, c="#c0392b", linewidths=0, label="p < 0.05", zorder=3)
    med = float(np.nanmedian(d))
    ax0.axhline(med, color="#2e74b5", lw=1.2, ls="--", zorder=4,
                label=f"median $d$ = {med:.3f}")

    com = eff[eff.is_common]
    if len(com):
        ci = eff.index[eff.is_common].to_numpy()
        ax0.scatter(ci, com.d, s=18, facecolors="none", edgecolors="#1a1a1a", linewidths=0.9,
                    zorder=5, label=f"conventional defaults (n = {len(com)})")

    ax0.set_ylabel(metric_label, fontsize=9)
    ax0.set_xlim(-n * 0.01, n * 1.01)
    ax0.set_xticks([])
    ax0.tick_params(labelsize=8)
    ax0.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2, handletextpad=0.4,
               columnspacing=1.2)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.text(-0.17, 1.04, "(a)", transform=ax0.transAxes, fontsize=10, fontweight="bold",
             va="bottom")

    # ---- (b) which levels sit where -------------------------------------------
    edges = np.linspace(0, n, n_bins + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    per_bin = np.bincount(which, minlength=n_bins).astype(float)

    grid, yticks, ylabels, seps = [], [], [], []
    row = 0
    for axis, lv in levels:
        col = eff[axis].to_numpy()
        for level in lv:
            m = col == level
            share = np.bincount(which[m], minlength=n_bins) / np.maximum(per_bin, 1)
            ratio = share / max(m.mean(), 1e-12)
            # a 3-bin moving average: the cartesian product alternates levels from bin to bin, so
            # the raw ratio prints as speckle and hides the trend the panel exists to show
            grid.append(np.convolve(ratio, np.ones(3) / 3.0, mode="same"))
            yticks.append(row)
            ylabels.append(f"{axis} = {level}")
            row += 1
        seps.append(row - 0.5)
    seps.pop()

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    im = ax1.imshow(np.vstack(grid), aspect="auto", cmap="RdBu_r", norm=norm,
                    extent=(0, n, len(grid) - 0.5, -0.5), interpolation="nearest")
    for y in seps:
        ax1.axhline(y, color="white", lw=1.6)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels, fontsize=6.0)
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlabel("specifications, ordered by effect size", fontsize=9)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.text(-0.17, 1.015, "(b)", transform=ax1.transAxes, fontsize=10, fontweight="bold",
             va="bottom")

    cb = fig.colorbar(im, ax=ax1, location="bottom", fraction=0.055, pad=0.16, aspect=42,
                      ticks=[0, 1, 2])
    cb.set_label("how often a level occurs in that part of the curve, "
                 "relative to how often it occurs overall", fontsize=7)
    cb.ax.set_xticklabels(["never", "as often", "twice as often"], fontsize=7)
    cb.outline.set_visible(False)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")  # vector copy
    plt.close(fig)
    print(f"wrote {out}")
    return out


def variance_bar(vd, out=f"{FIGDIR}/fig2_variance.png"):
    vd = vd.sort_values("eta_sq", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(vd.axis, vd.eta_sq, color="#4c78a8")
    ax.set_xlabel(r"$\eta^2$: share of variance in the effect size")
    ax.set_title("Which unreported analytic choice moves the result most?", fontsize=11)
    for i, (a, e) in enumerate(zip(vd.axis, vd.eta_sq)):
        ax.text(e + 0.004, i, f"{e:.3f}", va="center", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")  # vector copy
    print(f"wrote {out}")
    return out


ALPHA = BS + "alpha"
DELTA_ALPHA = BS + "Delta" + BS + "alpha"
LABEL = {
    ("alpha", ""): f"Cohen's $d$ (young $-$ elderly), DFA ${ALPHA}$",
    ("delta_alpha", ""): f"Cohen's $d$ (young $-$ elderly), ${DELTA_ALPHA}$",
    ("alpha", "repl_chf2db_vs_nsrdb_"): f"Cohen's $d$ (chf2db $-$ nsrdb), DFA ${ALPHA}$",
    ("delta_alpha", "repl_chf2db_vs_nsrdb_"):
        f"Cohen's $d$ (chf2db $-$ nsrdb), ${DELTA_ALPHA}$",
}


def render(metric, prefix=""):
    """One (metric, contrast) pair. `prefix` selects the replication effect tables.

    The replication figures existed before this entry point did -- they were made by an
    interactive call -- which meant four of the study's figures were not reproducible from any
    command. They are now.
    """
    eff = pd.read_csv(f"results/effects_{prefix}{metric}.csv")
    vd = pd.read_csv(f"results/variance_{prefix}{metric}.csv")
    tag = "repl_" if prefix else "spec_curve_"
    spec_curve(eff, out=f"{FIGDIR}/fig1_{tag}{metric}.png",
               metric_label=LABEL[(metric, prefix)])
    variance_bar(vd, out=f"{FIGDIR}/fig2_{'repl_' if prefix else 'variance_'}{metric}.png")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        for m in ("alpha", "delta_alpha"):
            for pre in ("", "repl_chf2db_vs_nsrdb_"):
                render(m, pre)
    else:
        render(args[0], args[1] if len(args) > 1 else "")
