"""Figure 3 -- record length crossed with the frozen specification grid.

Left: the effect at the median specification against record length, with the full span of the
9,216 specifications shaded behind it and the `commonly used` subgrid's span drawn as a darker
ribbon. Right: the eta^2 of each analytic axis against record length, which is where the
reordering shows -- the axis that dominates is not the same axis at every length.

Reads `results/length_ladder.csv`, produced by `src/length_control.py`. Draws whatever rungs are
present, so it is useful while the ladder is still filling.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import AXES, effect_table, variance_decomposition  # noqa: E402
from length_control import CONTRASTS, MATCHED, grid_path, load_rung  # noqa: E402

FIGDIR = "figures"     # figures/ for PNGs, results/ for data -- see src/figures.py

LABEL = {"alpha": r"DFA $\alpha$", "delta_alpha": r"multifractal width $\Delta\alpha$"}
TITLE = {"aging": "Fantasia: young $-$ old", "disease": "chf2db $-$ nsrdb"}
# Axes worth naming in the legend; the rest are drawn thin and grey. Chosen as the union of the
# axes that top any cell in the primary analysis, so the selection is not made from this figure.
HIGHLIGHT = ["ectopic", "s_max_frac", "s_min", "order"]
COLOR = {"ectopic": "#e45756", "s_max_frac": "#4c78a8", "s_min": "#72b7b2", "order": "#f58518"}


def _x(df):
    """Truncation length for plotting; 0 (the full-length run) is drawn at its true median.

    `df["truncate"]` and not `df.truncate` -- the latter resolves to DataFrame.truncate, the
    method, and comparing a bound method to an int raises rather than returning the column.
    """
    return np.where(df["truncate"] > 0, df["truncate"], df["median_length"]).astype(float)


def effect_vs_length(lad, out=f"{FIGDIR}/fig3_length_effect.png"):
    cells = [(c, m) for c in ["aging", "disease"] for m in ["alpha", "delta_alpha"]
             if len(lad[(lad.contrast == c) & (lad.metric == m)])]
    if not cells:
        print("no ladder rows yet")
        return None
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.3), squeeze=False)
    for ax, (c, m) in zip(axes[0], cells):
        d = lad[(lad.contrast == c) & (lad.metric == m)].sort_values("median_length")
        x = _x(d)
        ax.fill_between(x, d.d_min, d.d_max, color="#c9c9c9", alpha=0.55,
                        label="all 9,216 specifications")
        ax.fill_between(x, d.common_d_min, d.common_d_max, color="#4c78a8", alpha=0.30,
                        label="'commonly used' subgrid (n=32)")
        ax.plot(x, d.d_median, "o-", color="#1b1b1b", lw=1.8, ms=5,
                label="median specification")
        ax.axhline(0, color="black", lw=1.0)
        ax.axvline(MATCHED, color="#e45756", lw=1.1, ls="--", alpha=0.8)
        ax.text(MATCHED, ax.get_ylim()[1], f" length-matched\n {MATCHED:,}", fontsize=7,
                color="#e45756", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("RR intervals per record (log scale)")
        ax.set_ylabel(f"Cohen's $d$, {LABEL[m]}")
        ax.set_title(f"{TITLE[c]}, {LABEL[m]}", fontsize=10.5)
        ax.grid(alpha=0.2, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("The effect at the median defensible specification depends on record length",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=185, bbox_inches="tight")
    print(f"wrote {out}")
    return out


def eta_vs_length(out=f"{FIGDIR}/fig4_length_variance.png", metrics=("alpha", "delta_alpha")):
    """Recompute the full eta^2 vector at every rung.

    `length_ladder.csv` stores only the top two axes per rung, which is enough for the table but
    not for a figure showing the whole ranking move. This re-reads the per-rung grids, which are
    already on disk, so it costs a few seconds and no refitting.
    """
    rows = []
    for contrast, spec in CONTRASTS.items():
        for m in metrics:
            for n in [None, *sorted({1000, 2000, 4000, MATCHED, 8000, 16000, 32000})]:
                try:
                    if n is not None and not all(os.path.exists(grid_path(c, n))
                                                 for c in spec["cohorts"]):
                        continue
                    df = pd.concat([load_rung(c, n) for c in spec["cohorts"]], ignore_index=True)
                except FileNotFoundError:
                    continue
                eff = effect_table(df, metric=m, g1=spec["g1"], g2=spec["g2"])
                if eff.empty:
                    continue
                vd = variance_decomposition(eff)
                length = (float(np.nanmedian(df.groupby("record").n_input.first())) if n is None
                          else float(n))
                for r in vd.itertuples():
                    rows.append(dict(contrast=contrast, metric=m, length=length,
                                     axis=r.axis, eta_sq=r.eta_sq))
    if not rows:
        print("no rungs on disk yet")
        return None
    eta = pd.DataFrame(rows)
    eta.to_csv("results/length_eta.csv", index=False)

    cells = sorted({(r.contrast, r.metric) for r in eta.itertuples()},
                   key=lambda t: (t[0] != "aging", t[1]))
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.0), squeeze=False)
    for ax, (c, m) in zip(axes[0], cells):
        sub = eta[(eta.contrast == c) & (eta.metric == m)]
        for axis in AXES:
            s = sub[sub.axis == axis].sort_values("length")
            if s.empty:
                continue
            if axis in HIGHLIGHT:
                ax.plot(s.length, s.eta_sq, "o-", lw=1.9, ms=4.5,
                        color=COLOR[axis], label=axis)
            else:
                ax.plot(s.length, s.eta_sq, "-", lw=0.8, color="#bdbdbd", alpha=0.8)
        ax.axvline(MATCHED, color="#e45756", lw=1.0, ls="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("RR intervals per record (log scale)")
        ax.set_ylabel(r"$\eta^2$ on the effect size")
        ax.set_title(f"{TITLE[c]}, {LABEL[m]}", fontsize=10.5)
        ax.grid(alpha=0.2, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    # Outside the axes: in the Fantasia alpha panel the curves cross in the upper left, which is
    # where a "best" legend lands, and it sat on top of the first data point.
    axes[0][-1].legend(fontsize=8, frameon=False, title="axis", title_fontsize=8,
                       loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.suptitle("Which analytic choice dominates is itself a function of record length",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=185, bbox_inches="tight")
    print(f"wrote {out}  (and results/length_eta.csv)")
    return out


if __name__ == "__main__":
    lad = pd.read_csv("results/length_ladder.csv")
    effect_vs_length(lad)
    eta_vs_length()
