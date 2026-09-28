"""The synthetic arm has to be right before it can be evidence about anything else.

Two kinds of check. The generator and the injector are tested directly, because a study that
scores bias against a "known" exponent is worthless if the series does not have that exponent. The
shipped results are then asserted against the claims the paper makes from them, the same rule the
rest of the suite follows: read `results/`, never a literal.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from run_grid import estimate  # noqa: E402
from synthetic import fgn, inject_ectopic, rr_series  # noqa: E402


def summary():
    path = os.path.join(RESULTS, "synthetic_summary.json")
    if not os.path.exists(path):
        pytest.skip("synthetic arm not run; python src/synthetic.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ the generator
@pytest.mark.parametrize("hurst", [0.6, 0.8])
def test_generated_series_have_the_exponent_they_claim(hurst):
    """A conventional specification on fractional Gaussian noise must return alpha near H, or the
    whole arm is scoring bias against the wrong number."""
    rng = np.random.default_rng(7)
    spec = dict(ectopic="none", detrend="none", normalise="none", order=1, s_min=16,
                s_max_frac=0.1, n_scales=20, spacing="log", overlap=False, q_grid="coarse")
    got = [estimate(rr_series(8192, hurst, rng), spec)["alpha"] for _ in range(5)]
    assert abs(float(np.mean(got)) - hurst) < 0.05, (hurst, got)


def test_the_generator_is_reproducible_from_its_seed():
    a = fgn(512, 0.7, np.random.default_rng(1))
    b = fgn(512, 0.7, np.random.default_rng(1))
    assert np.allclose(a, b)


def test_injection_preserves_length_and_the_local_mean():
    """A premature beat plus its compensatory pause leaves the sum of the pair unchanged, which is
    what makes the arm a test of editing rather than of a shifted mean."""
    rng = np.random.default_rng(3)
    clean = rr_series(2000, 0.8, rng)
    dirty, idx = inject_ectopic(clean, 0.02, rng)
    assert len(dirty) == len(clean)
    assert len(idx) == 40
    for i in idx:
        assert dirty[i] < clean[i]
        assert np.isclose(dirty[i] + dirty[i + 1], clean[i] + clean[i + 1])


def test_no_injection_leaves_the_series_alone():
    rng = np.random.default_rng(4)
    clean = rr_series(500, 0.7, rng)
    same, idx = inject_ectopic(clean, 0.0, rng)
    assert len(idx) == 0 and np.array_equal(clean, same)


# ------------------------------------------------------------------ the shipped results
def test_conventional_settings_are_nearly_unbiased():
    """The claim the section leads with: at conventional settings the implementation recovers the
    generating exponent, so the spread across the grid is not the estimator being broken."""
    assert abs(summary()["conventional_bias"]) < 0.02


def test_the_grid_moves_one_series_further_than_the_estimator_errs():
    """Within a single synthetic series, the span of recovered exponents across specifications is
    far larger than the bias at conventional settings."""
    s = summary()
    assert s["median_span_within_series"] > 10 * abs(s["conventional_bias"])
    assert s["median_span_within_series"] > 0.1


def test_editing_beats_not_editing_once_ectopy_is_frequent():
    """Section 5.1 argues from the literature that deletion distorts and that leaving ectopy in
    distorts more. At 5% injected beats the arm measures it."""
    import pandas as pd
    path = os.path.join(RESULTS, "synthetic_ectopic.csv")
    if not os.path.exists(path):
        pytest.skip("ectopic arm not run")
    ect = pd.read_csv(path)
    at5 = ect[ect.rate == 0.05].groupby("ectopic").bias.apply(lambda s: s.abs().mean())
    assert at5["none"] > at5["mad4_drop"], at5.to_dict()
