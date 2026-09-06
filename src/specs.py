"""The specification grid -- PRE-REGISTERED.

Every axis below is a choice that appears in the published DFA/MFDFA literature and that
papers routinely leave unreported. The grid is the cartesian product.

The defence against "you manufactured instability by including specifications no competent
analyst would use" is that each level is justified from practice, and that we additionally
report the curve restricted to a `commonly_used` subgrid (see `is_common`).

DO NOT add axes after seeing results. Note additions in PREREGISTRATION.md with a date.
"""
import itertools

import numpy as np

# ---------------------------------------------------------------- axis definitions
# Each entry: level -> short justification for the pre-registration table.

ECTOPIC = {
    "none": "no correction; used when recordings are pre-screened as artefact-free",
    "pct20_drop": "Malik-style 20% rule, deletion -- the most common HRV convention",
    "pct20_interp": "same detection, linear interpolation -- preserves series length",
    "mad4_drop": "robust 4-MAD rule on successive differences",
}

DETREND = {
    "none": "raw RR series; DFA already removes polynomial trends within windows",
    "linear": "global linear detrend, common when slow drift is present",
    "quadratic": "global quadratic detrend",
}

NORMALISE = {
    "none": "raw ms units",
    "zscore": "unit variance; standard when comparing subjects with different HR",
}

ORDER = {
    1: "DFA1 -- the original Peng et al. formulation",
    2: "DFA2 -- the most widely used default",
    3: "DFA3 -- removes cubic trends",
    4: "DFA4 -- used for strongly nonstationary records",
}

S_MIN = {
    16: "common lower bound; below ~10 the polynomial fit is unstable",
    32: "conservative lower bound used when order >= 3",
}

S_MAX_FRAC = {
    0.05: "N/20 -- conservative, avoids few-window bias at large scales",
    0.10: "N/10 -- the most common convention",
    0.25: "N/4 -- permissive; used in short-record studies",
}

N_SCALES = {
    20: "typical resolution",
    30: "finer scale sampling",
}

SPACING = {
    "log": "log-spaced scales -- near-universal in this literature",
    "linear": "linear spacing -- occasionally used",
}

OVERLAP = {
    False: "non-overlapping windows -- the original formulation",
    True: "50% overlapping windows -- improves statistics on short records",
}

Q_GRID = {
    "coarse": np.arange(-5, 5.1, 1.0),
    "fine": np.arange(-5, 5.1, 0.5),
}

# The subgrid a conventional analyst would plausibly pick without thinking hard.
COMMON = dict(
    ectopic={"pct20_drop", "none"},
    detrend={"none", "linear"},
    normalise={"none", "zscore"},
    order={1, 2},
    s_min={16},
    s_max_frac={0.10},
    n_scales={20},
    spacing={"log"},
    overlap={False},
)


def build_grid():
    axes = [
        ("ectopic", list(ECTOPIC)),
        ("detrend", list(DETREND)),
        ("normalise", list(NORMALISE)),
        ("order", list(ORDER)),
        ("s_min", list(S_MIN)),
        ("s_max_frac", list(S_MAX_FRAC)),
        ("n_scales", list(N_SCALES)),
        ("spacing", list(SPACING)),
        ("overlap", list(OVERLAP)),
        ("q_grid", list(Q_GRID)),
    ]
    names = [a[0] for a in axes]
    out = []
    for combo in itertools.product(*[a[1] for a in axes]):
        out.append(dict(zip(names, combo)))
    return out


def is_common(spec):
    return all(spec[k] in v for k, v in COMMON.items())


def grid_summary():
    g = build_grid()
    n_common = sum(is_common(s) for s in g)
    sizes = {
        "ectopic": len(ECTOPIC), "detrend": len(DETREND), "normalise": len(NORMALISE),
        "order": len(ORDER), "s_min": len(S_MIN), "s_max_frac": len(S_MAX_FRAC),
        "n_scales": len(N_SCALES), "spacing": len(SPACING), "overlap": len(OVERLAP),
        "q_grid": len(Q_GRID),
    }
    return dict(total=len(g), common=n_common, axis_sizes=sizes)


if __name__ == "__main__":
    s = grid_summary()
    print("axis sizes:", s["axis_sizes"])
    print("total specifications:", s["total"])
    print("'commonly used' subgrid:", s["common"])
