"""Figure: the 20% rule latching on record f2y05.

Section 3.4 is the most concrete thing in the paper and existed only as prose. This draws it.

Left: the first 1,000 RR intervals of `f2y05`, with the two genuine ectopic beats marked, the
running "last accepted" reference drawn as a step line so the ratchet is visible, and the region
the rule deletes shaded. Right: the same record under the previous-raw-interval variant of the
same rule at the same threshold.

The point the figure has to make in one look: the reference walks up during the post-ectopic
recovery, every step of the walk being within 20% of the one before it, and then never comes back
down, so every normal beat afterwards falls outside the band.
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

from data import load_cached  # noqa: E402
from ectopic_case import PCT, keep_previous_raw  # noqa: E402

FIGDIR = "figures"


def reference_track(rr, pct=PCT):
    """The value the rule is comparing against at each step, and whether the beat was kept."""
    rr = np.asarray(rr, float)
    ref = np.empty(len(rr))
    keep = np.ones(len(rr), bool)
    prev = rr[0]
    for i in range(len(rr)):
        ref[i] = prev
        if i and abs(rr[i] - prev) > pct * prev:
            keep[i] = False
        else:
            prev = rr[i]
    return ref, keep


def draw(record="f2y05", cohort="fantasia", n=1000, out=f"{FIGDIR}/fig5_ectopic_latch.png"):
    rr = np.asarray(load_cached(cohort)[record][:n], dtype=float)
    x = np.arange(len(rr))
    ref, keep_a = reference_track(rr)
    keep_b = keep_previous_raw(rr)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)

    for ax, keep, title, show_ref in (
        (axes[0], keep_a,
         f"Compare with the last ACCEPTED interval: {keep_a.sum():,} of {len(rr):,} kept "
         f"({100 * keep_a.mean():.1f}%)", True),
        (axes[1], keep_b,
         f"Compare with the previous RAW interval: {keep_b.sum():,} of {len(rr):,} kept "
         f"({100 * keep_b.mean():.1f}%)", False),
    ):
        dropped = np.where(~keep)[0]           # shaded first, so the trace sits on top
        if dropped.size:
            runs = np.split(dropped, np.where(np.diff(dropped) > 1)[0] + 1)
            for j, run in enumerate(runs):
                ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color="#e45756", alpha=0.13, lw=0,
                           label="deleted by the rule" if j == 0 else None)
        ax.plot(x, rr, lw=0.7, color="#4c78a8", label="RR interval")
        if show_ref:
            ax.step(x, ref, where="post", lw=1.4, color="#1b1b1b", alpha=0.85,
                    label="reference (last accepted)")
            for sign in (1 - PCT, 1 + PCT):
                ax.step(x, ref * sign, where="post", lw=0.8, ls=":", color="#1b1b1b", alpha=0.6,
                        label=f"acceptance band, {int(PCT * 100)}%" if sign < 1 else None)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("RR interval index")
        ax.grid(alpha=0.2, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, len(rr))

    axes[0].set_ylabel("RR interval (ms)")

    # Text in axes coordinates, arrow tips in data coordinates, so nothing lands on an axis label
    # when the y-range changes. Headroom is added first, because the annotations sit inside it.
    ax = axes[0]
    lo, hi = float(rr.min()), float(rr.max())
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.10 * (hi - lo))
    nl = chr(10)

    def note(text, tip, xy_frac, color="#333333", weight="normal", size=8.0):
        ax.annotate(text, xy=tip, xycoords="data", xytext=xy_frac, textcoords="axes fraction",
                    fontsize=size, ha="left", va="center", color=color, fontweight=weight,
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=color, shrinkA=3, shrinkB=3))

    note(f"two genuine ectopic beats ({rr[89]:.0f} and {rr[93]:.0f} ms),{nl}"
         f"both removed correctly",
         (93, float(rr[93])), (0.16, 0.95))
    note(f"then {rr[95]:.0f} ms is ACCEPTED, only "
         f"{100 * (rr[95] / rr[94] - 1):.0f}% above the beat{nl}"
         f"before it, and becomes the reference",
         (95, float(rr[95])), (0.30, 0.72), color="#c0392b", weight="bold", size=8.5)
    note(f"every later beat sits more than {int(PCT * 100)}% below that frozen{nl}"
         f"reference, so nothing is ever accepted again",
         (700, float(np.median(rr[96:]))), (0.34, 0.44), color="#c0392b")

    # Legends go under the axes. Inside, they compete with the annotations for the one empty
    # region of the left panel, and an annotation that overlaps a legend is worse than either.
    for a in axes:
        a.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=4,
                 bbox_to_anchor=(0.5, -0.16), handlelength=1.6, columnspacing=1.2)
    over = 100 * float((np.abs(np.diff(rr)) / rr[:-1] > PCT).mean())
    fig.suptitle(
        f"One record, one rule, two standard implementations: {record}, a screened-healthy young "
        f"subject{nl}{over:.1f}% of successive differences exceed {int(PCT * 100)}%, yet the "
        f"left-hand variant deletes {100 * (1 - keep_a.mean()):.1f}% of the record",
        fontsize=11, y=1.06)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")  # vector copy
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    draw()
