"""Tests for the DFA/MFDFA core.

DFA has analytic ground truth for standard processes, so correctness is checkable rather
than merely plausible:

    white noise          alpha ~ 0.5
    1/f (pink) noise     alpha ~ 1.0
    random walk (fBm)    alpha ~ 1.5
    monofractal series   delta-alpha ~ 0 (narrow singularity spectrum)

If these drift, the whole specification curve is measuring an implementation bug.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mfdfa import (  # noqa: E402
    hurst_from_fluct,
    integrate_profile,
    make_scales,
    mfdfa,
    singularity_width,
)
from preprocess import ECTOPIC, apply_spec  # noqa: E402

QS = np.arange(-5, 5.1, 0.5)


def alpha_of(x, **kw):
    kw.setdefault("s_min", 16)
    kw.setdefault("s_max_frac", 0.1)
    kw.setdefault("n_scales", 20)
    kw.setdefault("order", 2)
    scales, F = mfdfa(x, QS, **kw)
    hq = hurst_from_fluct(scales, F)
    return float(hq[int(np.argmin(np.abs(QS - 2.0)))])


def pink_noise(n, rng):
    """1/f noise via spectral synthesis."""
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(f)
    return np.fft.irfft(spec, n=n)


def test_white_noise_alpha_half():
    rng = np.random.default_rng(0)
    a = np.mean([alpha_of(rng.normal(size=20000)) for _ in range(3)])
    assert 0.42 < a < 0.58, f"white noise alpha={a:.3f}, expected ~0.5"


def test_random_walk_alpha_one_and_a_half():
    rng = np.random.default_rng(1)
    a = np.mean([alpha_of(np.cumsum(rng.normal(size=20000))) for _ in range(3)])
    assert 1.35 < a < 1.65, f"random walk alpha={a:.3f}, expected ~1.5"


def test_pink_noise_alpha_about_one():
    rng = np.random.default_rng(2)
    a = np.mean([alpha_of(pink_noise(20000, rng)) for _ in range(3)])
    assert 0.80 < a < 1.20, f"pink noise alpha={a:.3f}, expected ~1.0"


def test_alpha_ordering_is_monotone():
    """white < pink < random walk, which is the property the whole method rests on."""
    rng = np.random.default_rng(3)
    w = alpha_of(rng.normal(size=20000))
    p = alpha_of(pink_noise(20000, rng))
    r = alpha_of(np.cumsum(rng.normal(size=20000)))
    assert w < p < r, f"ordering violated: white={w:.3f} pink={p:.3f} walk={r:.3f}"


def test_monofractal_has_narrow_spectrum():
    rng = np.random.default_rng(4)
    scales, F = mfdfa(rng.normal(size=20000), QS, s_min=16, s_max_frac=0.1, n_scales=20, order=2)
    assert singularity_width(QS, hurst_from_fluct(scales, F)) < 0.5


def test_alpha_invariant_to_affine_rescaling():
    """DFA alpha must not change under z-scoring. This is why `normalise` came out as an
    exact null (eta^2 ~ 1e-11) in the specification curve."""
    rng = np.random.default_rng(5)
    x = pink_noise(20000, rng)
    a1 = alpha_of(x)
    a2 = alpha_of((x - x.mean()) / x.std())
    assert abs(a1 - a2) < 1e-6, f"{a1} vs {a2}"


def test_integrate_profile_is_mean_centred_cumsum():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(integrate_profile(x), np.cumsum(x - 2.5))


def test_make_scales_respects_bounds():
    s = make_scales(10000, s_min=16, s_max_frac=0.1, n_scales=20, spacing="log")
    assert s.min() >= 16 and s.max() <= 1000 and len(s) <= 20
    assert np.all(np.diff(s) > 0)


def test_make_scales_empty_when_range_invalid():
    assert len(make_scales(100, s_min=64, s_max_frac=0.1, n_scales=20)) == 0


def test_short_series_returns_nan_not_garbage():
    scales, F = mfdfa(np.random.default_rng(6).normal(size=50), QS,
                      s_min=16, s_max_frac=0.1, n_scales=20, order=2)
    hq = hurst_from_fluct(scales, F)
    assert len(scales) == 0 or np.all(~np.isfinite(hq)) or len(scales) < 4


@pytest.mark.parametrize("name", list(ECTOPIC))
def test_ectopic_handlers_preserve_plausible_rr(name):
    rng = np.random.default_rng(7)
    rr = 800 + 40 * rng.normal(size=5000)
    out = ECTOPIC[name](rr)
    assert len(out) > 0 and np.all(np.isfinite(out))
    assert 600 < np.mean(out) < 1000


def test_interp_preserves_length_drop_does_not():
    rng = np.random.default_rng(8)
    rr = 800 + 40 * rng.normal(size=5000)
    rr[::100] = 2000.0                      # inject ectopic-like spikes
    assert len(ECTOPIC["pct20_interp"](rr)) == len(rr)
    assert len(ECTOPIC["pct20_drop"](rr)) < len(rr)


def test_apply_spec_chain_runs():
    rng = np.random.default_rng(9)
    rr = 800 + 40 * rng.normal(size=3000)
    out = apply_spec(rr, dict(ectopic="pct20_drop", detrend="linear", normalise="zscore"))
    assert np.isfinite(out).all()
    assert abs(out.mean()) < 1e-8 and abs(out.std() - 1.0) < 1e-6
