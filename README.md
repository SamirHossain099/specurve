# specurve

Specification-curve (multiverse) analysis for estimators whose parameters are chosen rather than
derived, with detrended fluctuation analysis and its multifractal extension as the worked case.

This repository is both an installable package and the reproduction material for a study of how
far a physiological result moves across the space of defensible DFA and MFDFA pipelines. The study
pre-registered 9,216 specifications, froze the grid before computing any group-level effect, and
ran it over three open PhysioNet cohorts: 801,792 fits, no failures.

## Install

```bash
pip install -e .
```

Requires Python 3.10 or later, with numpy, scipy, pandas and joblib. `pyarrow` is needed to read
and write the grid files, `matplotlib` only for the figure scripts, and `wfdb` only to fetch the
data. Developed and tested on Python 3.12; earlier versions are declared but not exercised.

## The package

Two layers, deliberately separated.

**`grid` and `curve` know nothing about DFA.** They take a mapping of axis name to levels and a
callable that turns one specification plus one series into a dict of estimates. Lift them for any
pipeline whose parameters go unreported.

```python
from specurve import grid, curve, dfa

AXES = {"ectopic": ["none", "pct20_drop"], "detrend": ["none", "linear"],
        "normalise": ["none", "zscore"], "order": [1, 2], "s_min": [16],
        "s_max_frac": [0.1], "n_scales": [20], "spacing": ["log"],
        "overlap": [False], "q_grid": ["coarse"]}

df  = grid.run(AXES, series, dfa.estimate, meta={k: {"group": g[k]} for k in series})
eff = curve.effect_table(df, "alpha", list(AXES), g1="young", g2="old")

curve.summarise(eff)                              # the curve: median, range, sign flips
curve.variance_decomposition(eff, list(AXES))     # which choice moves the result most
curve.permutation_test_median(df, "alpha", g1="young")   # inference across the whole grid
```

**`dfa` is the estimator**: DFA and MFDFA with every analytic choice as an explicit named
argument, plus the RR-interval preprocessing axes, which in this literature are not separable from
the estimator.

`grid.freeze_hash(AXES)` fingerprints a grid, so a pre-registration can name one and a later run
can prove it used the same one.

### Four rules the package enforces

Each was a bug in this study before it was a rule.

1. **Nothing defaults silently.** A specification is a dictionary of every axis. There is no "the
   remaining parameters were whatever the library does".
2. **Failures are recorded, not dropped.** An estimator that cannot evaluate a specification
   returns `ok=False` and still produces a row, so the denominator stays honest. It mattered here:
   a silently vanishing specification would have removed one subject from a quarter of the grid
   without appearing in any count.
3. **One definition of the effect size.** `cohens_d` and `pooled_d_matrix` are the scalar and
   vectorised forms of the same estimator, pinned equal by the tests on deliberately unbalanced
   groups. They had drifted apart, and a balanced cohort hid it.
4. **A non-finite observed statistic never yields a p-value.** `abs(null) >= nan` is all-False, so
   a p-value built by counting exceedances reports maximal significance for a comparison that is
   undefined. Guard the observation, not just the null.

## Reproducing the study

Data are three public PhysioNet databases, open access, no credentials, no data-use agreement.
Nothing here needs a GPU.

| Cohort | n | Role |
|---|---|---|
| `fantasia` | 40 (20 young, 20 elderly) | primary contrast, healthy aging |
| `nsrdb` | 18 | healthy control |
| `chf2db` | 29 | congestive heart failure |

```bash
python src/data.py fantasia nsrdb chf2db      # fetch and cache the RR series
python src/run_grid.py                        # the 9,216-specification grid, about 10 min
python src/analyze.py --metric alpha --n-perm 1000
python src/analyze.py --metric delta_alpha --n-perm 1000
python src/replicate.py --cohorts nsrdb chf2db
python src/interactions.py                    # two-way decomposition
python src/length_control.py                  # record-length ladder, about 2.5 h, resumable
python src/figures.py all && python src/figure_length.py && python src/ectopic_case.py
```

`src/estimate_cost.py` benchmarks a cohort before you commit to its full grid. `src/resources.py`
caps BLAS threads before numpy loads and must be imported first in every entry point; a test
enforces that.

## What is in `results/`

Enough to check every number in the paper's sections 3 to 5 without refitting anything.

| File | Contents |
|---|---|
| `effects_*.csv` | one row per specification: effect size, p-value, group means, every axis level |
| `summary_*.json` | curve summary, permutation test and one-way variance decomposition |
| `variance_*.csv` | one-way eta-squared per axis |
| `interactions.csv`, `interactions_summary.json` | every two-way interaction, all four cells |
| `length_ladder.csv`, `length_eta.csv`, `length_matched_verdict.json` | the record-length control |
| `ectopic_case.json` | the 20 percent rule latch, measured |

The raw per-fit grids (`results/*.parquet`, 70 MB) are not in git. They regenerate from `src/` in
about 2.5 CPU hours, and they are in the archived deposit.

## Verification

- The analysis stage is deterministic: re-running it from the cached grids reproduces all eight
  shipped result files byte for byte. The permutation test is seeded.
- 801,792 fits across the three primary grids, **zero failures**. A specification yielding fewer
  than 200 usable samples or fewer than 4 valid scales is counted as a failure rather than dropped,
  so that is a real zero and not a filtered one.
- 96 tests, `ruff` clean. The suite pins the estimator against known analytic nulls, pins the two
  implementations of Cohen's *d* against each other on unbalanced groups, and asserts that the
  package reproduces the study's own scripts exactly. An artifact that has drifted from the paper
  is no longer evidence for it.
- Some tests read the manuscript, which is not in this repository, and skip cleanly without it.

## Pre-registration

`PREREGISTRATION.md` holds the grid as it was frozen, the justification for every level from
published practice, the analysis plan, and every deviation with its date and reason. Two analyses
are post hoc and labelled: the record-length control and the interaction decomposition.

## Citing this

Archived at Zenodo: [doi:10.5281/zenodo.22490156](https://doi.org/10.5281/zenodo.22490156).
That is the concept DOI and always resolves to the latest release; each release also gets its own
version DOI. `CITATION.cff` carries the machine-readable metadata, and GitHub's "Cite this
repository" button reads it.

A manuscript describing the study is under review. Its reference will be added on acceptance.

## License

MIT. See `LICENSE`.
