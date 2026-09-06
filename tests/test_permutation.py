"""Tests for the permutation test.

This exists because of a real bug: `permutation_test_median` hardcoded `== "young"`, so on any
cohort not labelled young/old the mask was all-False, one group was empty, the observed effect
was NaN -- and `np.abs(null) >= nan` is all-False, which returned **p = 1/(n_perm+1) = 0.001**.
A meaningless comparison reported itself as the most significant result possible.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import analyze  # noqa: E402
from analyze import permutation_test_median  # noqa: E402


def make_df(groups=("young", "old"), n_per=10, n_specs=40, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gi, g in enumerate(groups):
        for r in range(n_per):
            rec = f"{g}_{r}"
            base = effect if gi == 0 else 0.0
            for s in range(n_specs):
                rows.append(dict(spec_id=s, record=rec, group=g,
                                 alpha=base + rng.normal(1.0, 0.2)))
    return pd.DataFrame(rows)


def test_works_on_young_old():
    out = permutation_test_median(make_df(), n_perm=100)
    assert np.isfinite(out["observed_median_d"])
    assert out["p_perm"] is not None


def test_works_on_arbitrary_group_names():
    """The regression: cohort labels are not always young/old."""
    out = permutation_test_median(make_df(groups=("nsrdb", "chf2db")), n_perm=100)
    assert np.isfinite(out["observed_median_d"]), "NaN on non-young/old labels -- the old bug"
    assert out["p_perm"] is not None
    assert set(out["groups"]) == {"nsrdb", "chf2db"}


def test_real_effect_is_detected():
    out = permutation_test_median(make_df(effect=1.5), n_perm=200)
    assert out["p_perm"] < 0.05, out


def test_no_effect_is_not_significant():
    out = permutation_test_median(make_df(effect=0.0, seed=3), n_perm=200)
    assert out["p_perm"] > 0.05, out


def test_explicit_g1_flips_the_sign():
    df = make_df(groups=("a", "b"), effect=1.5)
    a = permutation_test_median(df, n_perm=100, g1="a")["observed_median_d"]
    b = permutation_test_median(df, n_perm=100, g1="b")["observed_median_d"]
    assert np.sign(a) == -np.sign(b) and abs(abs(a) - abs(b)) < 1e-9


def test_bad_g1_raises():
    with pytest.raises(ValueError):
        permutation_test_median(make_df(), n_perm=10, g1="nonexistent")


def test_three_groups_raises():
    df = make_df(groups=("a", "b", "c"))
    with pytest.raises(ValueError):
        permutation_test_median(df, n_perm=10)


def test_nan_observation_withholds_p_value():
    """The core guard: a NaN effect must NOT report a p-value.

    Constant values within each group give pooled SD = 0, so d = 0/0 = NaN while the pivot
    still has all its columns -- the exact path that previously returned p = 0.001.
    """
    df = make_df()
    df["alpha"] = 1.0
    out = permutation_test_median(df, n_perm=50)
    assert out["p_perm"] is None, "a NaN observation must not produce a p-value"
    assert "error" in out


def test_all_nan_input_raises_rather_than_reporting():
    """Degenerate input must fail loudly, not return a number."""
    df = make_df()
    df["alpha"] = np.nan
    with pytest.raises(ValueError):
        permutation_test_median(df, n_perm=10)


def test_pooled_d_matrix_matches_cohens_d_on_unbalanced_groups():
    """The matrix estimator and the scalar one must be the same estimator.

    They were not: `med_effect` used sqrt((sa**2 + sb**2)/2), which equals the n-1 weighted
    pooled SD only when the groups are the same size. Fantasia is 20 vs 20, so the bug was
    invisible there and surfaced only on the 29-vs-18 replication contrast.
    """
    rng = np.random.default_rng(7)
    M = rng.normal(size=(50, 47))
    mask = np.zeros(47, dtype=bool)
    mask[:29] = True                      # deliberately unbalanced, like chf2db vs nsrdb
    got = analyze.pooled_d_matrix(M, mask)
    want = np.array([analyze.cohens_d(row[mask], row[~mask]) for row in M])
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_pooled_d_matrix_nans_below_min_group_size():
    M = np.arange(20, dtype=float).reshape(2, 10)
    mask = np.array([True, True] + [False] * 8)      # 2 < MIN_GROUP_N
    assert np.all(np.isnan(analyze.pooled_d_matrix(M, mask)))


@pytest.mark.parametrize("name", [
    "summary_alpha", "summary_delta_alpha",
    "summary_repl_chf2db_vs_nsrdb_alpha", "summary_repl_chf2db_vs_nsrdb_delta_alpha",
])
def test_permutation_observed_equals_curve_median_in_results(name):
    """Every shipped result file reports ONE median effect, not two.

    Reads `results/` rather than hardcoding, so it keeps tracking the number if the grid or
    the cohorts change.
    """
    path = os.path.join(ROOT, "results", f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"{name}.json not present")
    j = json.loads(open(path, encoding="utf-8").read())
    assert abs(j["summary"]["d_median"] - j["permutation"]["observed_median_d"]) < 1e-12
