"""mfdfa-multiverse -- specification-curve (multiverse) analysis for parameterised estimators.

Two layers, deliberately separated:

* `grid` and `curve` know nothing about DFA. They run a cartesian product of analytic choices
  against a set of units and turn the result into a specification curve, joint inference and a
  variance decomposition. Lift them for any pipeline whose parameters go unreported.
* `dfa` is the estimator this package was built around: DFA and MFDFA with every analytic choice
  as an explicit named argument and no silent defaults, plus the RR preprocessing axes, which in
  this literature are not separable from the estimator.

The design rules the package enforces, each of which was a bug before it was a rule:

1. **Nothing defaults silently** -- a specification is a dict of every axis.
2. **Failures are recorded, not dropped**, so the denominator stays honest.
3. **One definition of the effect size**, with the scalar and vectorised forms pinned equal on
   unbalanced groups.
4. **Inference is on the curve** -- a permutation test of the median effect across specifications
   -- not on a specification chosen after seeing it.
5. **A non-finite observed statistic never yields a p-value.**

Minimal use::

    from mfdfa_multiverse import grid, curve, dfa

    AXES = {"ectopic": ["none", "pct20_drop"], "detrend": ["none", "linear"],
            "normalise": ["none"], "order": [1, 2], "s_min": [16],
            "s_max_frac": [0.1], "n_scales": [20], "spacing": ["log"],
            "overlap": [False], "q_grid": ["coarse"]}

    df = grid.run(AXES, series, dfa.estimate, meta={k: {"group": g[k]} for k in series})
    eff = curve.effect_table(df, "alpha", list(AXES), g1="young", g2="old")
    print(curve.summarise(eff))
    print(curve.variance_decomposition(eff, list(AXES)))
    print(curve.permutation_test_median(df, "alpha", g1="young"))

See the project README for the study this came out of.
"""
from . import curve, dfa, grid  # noqa: F401
from .curve import (  # noqa: F401
    cohens_d,
    effect_table,
    permutation_test_median,
    pooled_d_matrix,
    summarise,
    variance_decomposition,
)
from .grid import build, failure_report, freeze_hash, run, summary  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "grid", "curve", "dfa",
    "build", "run", "freeze_hash", "summary", "failure_report",
    "effect_table", "variance_decomposition", "permutation_test_median", "summarise",
    "cohens_d", "pooled_d_matrix",
]
