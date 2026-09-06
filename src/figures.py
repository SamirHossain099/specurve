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


def spec_curve(eff, out=f"{FIGDIR}/fig1_spec_curve.png", metric_label=r"Cohen's $d$ (young $-$ old)"):
    eff = eff.sort_values("d").reset_index(drop=True)
    x = np.arange(len(eff))
    d = eff.d.to_numpy(dtype=float)
    sig = (eff.p < 0.05).to_numpy()

    levels = [(ax, list(pd.unique(eff[ax]))) for ax in AXES]
    n_rows = sum(len(v) for _, v in levels)

    fig = plt.figure(figsize=(13, 4 + 0.19 * n_rows))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 0.19 * n_rows], hspace=0.06)

    # ---- upper: the curve -----------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(x[~sig], d[~sig], s=3, c="#c9c9c9", label="p >= 0.05")
    ax0.scatter(x[sig], d[sig], s=3, c="#e45756", label="p < 0.05")
    ax0.axhline(0, color="black", lw=1.2)
    med = float(np.nanmedian(d))
    ax0.axhline(med, color="#4c78a8", lw=1.5, ls="--", label=f"median d = {med:.3f}")

    com = eff[eff.is_common]
    if len(com):
        ci = eff.index[eff.is_common].to_numpy()
        ax0.scatter(ci, com.d, s=26, facecolors="none", edgecolors="#1b1b1b",
                    linewidths=1.0, label=f"'commonly used' subgrid (n={len(com)})", zorder=5)

    ax0.set_ylabel(metric_label)
    ax0.set_xlim(-len(eff) * 0.01, len(eff) * 1.01)
    ax0.set_xticks([])
    frac_sig = sig.mean()
    flips = (d > 0).any() and (d < 0).any()
    ax0.set_title(
        f"Specification curve: {len(eff):,} defensible DFA pipelines, one dataset, one question\n"
        f"{100*frac_sig:.1f}% reach p<0.05   |   d spans {d.min():.2f} to {d.max():.2f}"
        f"   |   sign flips: {'YES' if flips else 'no'}", fontsize=11)
    ax0.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax0.grid(alpha=0.2, ls=":")

    # ---- lower: which choices are active --------------------------------------
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    row = 0
    yticks, ylabels = [], []
    for axis, lv in levels:
        col = eff[axis].to_numpy()
        for level in lv:
            m = col == level
            ax1.scatter(x[m], np.full(m.sum(), row), s=1.1, c="#3b3b3b", marker="s")
            yticks.append(row)
            ylabels.append(f"{axis} = {level}")
            row += 1
        row += 0.6
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels, fontsize=6.2)
    ax1.set_ylim(-1, row)
    ax1.invert_yaxis()
    ax1.set_xlabel("specifications, ordered by effect size")
    ax1.grid(alpha=0.15, ls=":", axis="x")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=185, bbox_inches="tight")
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
    fig.savefig(out, dpi=185, bbox_inches="tight")
    print(f"wrote {out}")
    return out


ALPHA = BS + "alpha"
DELTA_ALPHA = BS + "Delta" + BS + "alpha"
LABEL = {
    ("alpha", ""): f"Cohen's $d$ (young $-$ old), DFA ${ALPHA}$",
    ("delta_alpha", ""): f"Cohen's $d$ (young $-$ old), ${DELTA_ALPHA}$",
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
