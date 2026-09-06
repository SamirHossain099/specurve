"""Two-way interactions between analytic axes.

Why this exists
---------------
The main analysis reports a one-way eta^2 per axis, which answers "which choice moves the result
most" and nothing else. That leaves an obvious question unanswered in a design that can answer it
for free: **do the axes interact?**

The grid is a complete cartesian product, so it is balanced and orthogonal by construction and a
two-way decomposition is exact rather than approximate. For a pair (A, B):

    SS_cells(A,B) = sum over (a,b) cells of n_ab * (mean_ab - grand)^2
    SS_AB         = SS_cells(A,B) - SS_A - SS_B
    eta^2_AB      = SS_AB / SS_total

An earlier draft of section 2.4 said "the axes are not orthogonal in effect", which was loose
wording for a real thing: the *design* is orthogonal, and it is the *response surface* that is
non-additive. This module measures how non-additive.

Why it matters for the paper's recommendation: if ectopic handling and the scale range interact,
then "report both" is a stronger claim than "report the two biggest", because the effect of one
cannot be read off without knowing the other.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import AXES  # noqa: E402

CELLS = {
    "fantasia_alpha": ("effects_alpha.csv", "Fantasia (aging), alpha"),
    "fantasia_delta_alpha": ("effects_delta_alpha.csv", "Fantasia (aging), delta-alpha"),
    "disease_alpha": ("effects_repl_chf2db_vs_nsrdb_alpha.csv", "Disease, alpha"),
    "disease_delta_alpha": ("effects_repl_chf2db_vs_nsrdb_delta_alpha.csv", "Disease, delta-alpha"),
}


def _ss_between(d, labels, grand):
    """Between-group sum of squares for one grouping."""
    ss = 0.0
    for lv in pd.unique(labels):
        m = labels == lv
        ss += int(m.sum()) * (d[m].mean() - grand) ** 2
    return ss


def decompose(eff, axes=AXES):
    """One-way eta^2 for every axis and two-way eta^2 for every pair, from one effect table."""
    d = eff["d"].to_numpy(dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    grand = d.mean()
    sst = ((d - grand) ** 2).sum()
    if sst <= 0:
        raise ValueError("no variance in the effect size")

    cols = {ax: eff.loc[ok, ax].to_numpy() for ax in axes}
    main = {ax: _ss_between(d, cols[ax], grand) for ax in axes}

    rows = []
    for a, b in itertools.combinations(axes, 2):
        joint = np.array([f"{x}|{y}" for x, y in zip(cols[a], cols[b])])
        ss_ab = _ss_between(d, joint, grand) - main[a] - main[b]
        # A complete factorial makes this exact, but floating point can put a true zero a hair
        # below it; clamp so a -1e-17 does not print as a negative variance share.
        ss_ab = max(ss_ab, 0.0)
        rows.append(dict(axis_a=a, axis_b=b,
                         eta_sq_a=main[a] / sst, eta_sq_b=main[b] / sst,
                         eta_sq_interaction=ss_ab / sst))
    inter = pd.DataFrame(rows).sort_values("eta_sq_interaction", ascending=False)
    one_way = pd.DataFrame([dict(axis=ax, eta_sq=main[ax] / sst) for ax in axes]
                           ).sort_values("eta_sq", ascending=False)
    return one_way.reset_index(drop=True), inter.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6, help="pairs to print per cell")
    ap.add_argument("--out", default="results/interactions.csv")
    a = ap.parse_args()

    frames, summary = [], {}
    for key, (fname, label) in CELLS.items():
        path = os.path.join("results", fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} missing")
            continue
        eff = pd.read_csv(path)
        one_way, inter = decompose(eff)
        inter.insert(0, "cell", key)
        frames.append(inter)

        total_inter = float(inter.eta_sq_interaction.sum())
        total_main = float(one_way.eta_sq.sum())
        top = inter.iloc[0]
        summary[key] = dict(
            label=label,
            sum_main_eta_sq=round(total_main, 4),
            sum_two_way_eta_sq=round(total_inter, 4),
            largest_interaction=f"{top.axis_a} x {top.axis_b}",
            largest_interaction_eta_sq=round(float(top.eta_sq_interaction), 4),
            ectopic_x_s_max_frac=round(float(
                inter[((inter.axis_a == "ectopic") & (inter.axis_b == "s_max_frac"))
                      | ((inter.axis_a == "s_max_frac") & (inter.axis_b == "ectopic"))
                      ].eta_sq_interaction.iloc[0]), 4),
        )
        print(f"\n=== {label}")
        print(f"  main effects sum to eta^2 = {total_main:.3f}; "
              f"two-way interactions sum to {total_inter:.3f}")
        print(inter.head(a.top).to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}"))

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
        with open("results/interactions_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {a.out} and results/interactions_summary.json")


if __name__ == "__main__":
    main()
