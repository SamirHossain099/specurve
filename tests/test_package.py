"""The installable package: does it work standalone, and does it agree with the study's code?

Two obligations. The package has to be usable by someone who has never seen this project -- that
is the point of shipping it -- and it must not quietly disagree with the scripts that produced the
paper's numbers, or the artifact stops being evidence for the paper.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import mfdfa as legacy_mfdfa  # noqa: E402
import preprocess as legacy_pre  # noqa: E402
from mfdfa_multiverse import curve, dfa, grid  # noqa: E402

QS = np.arange(-5, 5.1, 1.0)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    series = {f"s{i:02d}": np.cumsum(rng.normal(0, 1, 2500)) + 800.0 for i in range(10)}
    meta = {k: {"group": "a" if i < 5 else "b"} for i, k in enumerate(series)}
    return series, meta


AXES = {"ectopic": ["none", "pct20_drop"], "detrend": ["none", "linear"],
        "normalise": ["none", "zscore"], "order": [1, 2], "s_min": [16],
        "s_max_frac": [0.1], "n_scales": [20], "spacing": ["log"],
        "overlap": [False], "q_grid": ["coarse"]}


# ------------------------------------------------------------------ agreement with the study

def test_package_dfa_reproduces_the_studys_estimator(synthetic):
    """If these ever diverge, the released artifact stops being evidence for the paper."""
    series, _ = synthetic
    x = series["s00"]
    for order in (1, 2, 3):
        a = legacy_mfdfa.hurst_from_fluct(
            *legacy_mfdfa.mfdfa(x, QS, s_min=16, s_max_frac=0.1, n_scales=20, order=order))
        b = dfa.hurst_from_fluct(
            *dfa.mfdfa(x, QS, s_min=16, s_max_frac=0.1, n_scales=20, order=order))
        assert np.allclose(a, b, equal_nan=True), f"order={order}"


def test_package_preprocessing_reproduces_the_studys(synthetic):
    series, _ = synthetic
    x = series["s00"]
    for ect in ("none", "pct20_drop", "pct20_interp", "mad4_drop"):
        for det in ("none", "linear", "quadratic"):
            spec = dict(ectopic=ect, detrend=det, normalise="zscore")
            assert np.allclose(legacy_pre.apply_spec(x, spec), dfa.apply_spec(x, spec),
                               equal_nan=True), f"{ect}/{det}"


def test_package_cohens_d_reproduces_the_studys():
    from analyze import cohens_d as legacy_d
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=29), rng.normal(1, 2, size=18)   # deliberately unbalanced
    assert curve.cohens_d(a, b) == pytest.approx(legacy_d(a, b), abs=1e-12)


# ------------------------------------------------------------------ the package standing alone

def test_grid_build_is_the_cartesian_product():
    g = grid.build(AXES)
    assert len(g) == 2 * 2 * 2 * 2
    assert len({tuple(sorted(s.items(), key=str)) for s in g}) == len(g)   # all distinct


def test_freeze_hash_is_stable_and_sensitive():
    """A pre-registration names this hash. It must not move when nothing moved, and must move
    when an axis does."""
    assert grid.freeze_hash(AXES) == grid.freeze_hash(dict(AXES))
    changed = dict(AXES, order=[1, 2, 3])
    assert grid.freeze_hash(changed) != grid.freeze_hash(AXES)
    reordered = {k: AXES[k] for k in reversed(list(AXES))}
    assert grid.freeze_hash(reordered) != grid.freeze_hash(AXES), "axis order changes spec_id"


def test_run_produces_one_row_per_spec_and_unit(synthetic):
    series, meta = synthetic
    df = grid.run(AXES, series, dfa.estimate, meta=meta)
    assert len(df) == len(grid.build(AXES)) * len(series)
    assert set(df.columns) >= {"spec_id", "unit", "group", "alpha", "delta_alpha", "ok"}


def test_failures_are_recorded_not_dropped(synthetic):
    """A specification that cannot be evaluated must still produce a row."""
    series, meta = synthetic
    short = {k: v[:120] for k, v in series.items()}            # below the 200-sample floor
    df = grid.run(AXES, short, dfa.estimate, meta=meta)
    rep = grid.failure_report(df)
    assert len(df) == len(grid.build(AXES)) * len(short)
    assert rep["failed"] == len(df) and rep["frac"] == 1.0


def test_is_common_marks_the_subgrid(synthetic):
    series, meta = synthetic
    df = grid.run(AXES, series, dfa.estimate, meta=meta,
                  is_common=lambda s: s["order"] == 2 and s["ectopic"] == "none")
    per_spec = df.groupby("spec_id").is_common.first()
    assert per_spec.sum() == 4          # 2 detrend x 2 normalise


def test_the_two_effect_size_forms_agree_through_the_public_api(synthetic):
    """The bug this package exists to not repeat: two estimators for one quantity."""
    series, meta = synthetic
    df = grid.run(AXES, series, dfa.estimate, meta=meta)
    eff = curve.effect_table(df, "alpha", list(AXES), g1="a", g2="b")
    s = curve.summarise(eff)
    pt = curve.permutation_test_median(df, "alpha", g1="a", n_perm=100)
    assert abs(s["d_median"] - pt["observed_median_d"]) < 1e-12


def test_permutation_rejects_a_group_label_that_is_not_present(synthetic):
    series, meta = synthetic
    df = grid.run(AXES, series, dfa.estimate, meta=meta)
    with pytest.raises(ValueError):
        curve.permutation_test_median(df, "alpha", g1="not_a_group", n_perm=10)


def test_variance_decomposition_recovers_an_exact_null(synthetic):
    """`normalise` cannot change h(q=2); alpha is invariant to affine rescaling. The package must
    recover that, which is the only internal correctness check the design offers."""
    series, meta = synthetic
    df = grid.run(AXES, series, dfa.estimate, meta=meta)
    eff = curve.effect_table(df, "alpha", list(AXES), g1="a", g2="b")
    vd = curve.variance_decomposition(eff, list(AXES))
    eta = dict(zip(vd.axis, vd.eta_sq))
    assert eta["normalise"] < 1e-8


def test_the_latching_and_non_latching_ectopic_rules_are_both_available():
    """Section 3.4's finding, exposed as two named levels rather than one unstated default.

    The construction is the mechanism from `f2y05`, minimally: a recovery ramp in which every
    step is *within* 20% of the one before, so each is accepted and walks the reference upward,
    followed by a return to baseline that now sits more than 20% below it. 800 -> 880 (+10%,
    accepted) -> 1050 (+19.3%, accepted) -> 780, which is 25.7% below the latched 1050 and is
    therefore rejected, as is every subsequent normal beat.

    Both variants flag that one 25.7% step -- it is a genuine >20% change and any 20% rule should.
    The difference is recovery: the raw variant drops one interval and carries on; the
    last-accepted variant never accepts anything again.
    """
    rr = np.concatenate([np.full(60, 800.0), [880.0, 1050.0], np.full(200, 780.0)])
    latched = dfa.ectopic_pct(rr, 0.20)
    raw = dfa.ectopic_pct_raw(rr, 0.20)
    assert len(latched) == 62, "the reference should latch at 1050 and accept nothing after it"
    assert len(raw) == len(rr) - 1, "the raw variant should drop the one real step and recover"
    assert len(raw) > 4 * len(latched)
    assert "pct20_drop" in dfa.ECTOPIC and "pct20_drop_raw" in dfa.ECTOPIC
