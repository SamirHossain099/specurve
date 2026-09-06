"""Smoke tests for the specification-curve figures.

The grid takes minutes to run; a plotting crash at the end wastes it. These exercise the
plotting path on synthetic effect tables in milliseconds.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze import AXES  # noqa: E402
from figures import spec_curve, variance_bar  # noqa: E402


def _fake_effects(n=400, seed=0, flip_sign=True):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.3, 0.5, size=n) if flip_sign else rng.normal(1.0, 0.1, size=n)
    df = pd.DataFrame({
        "spec_id": np.arange(n),
        "d": d,
        "p": rng.uniform(0, 1, size=n),
        "is_common": rng.random(n) < 0.05,
    })
    for ax in AXES:
        df[ax] = rng.choice([f"{ax}_a", f"{ax}_b"], size=n)
    return df


def test_spec_curve_renders(tmp_path):
    out = str(tmp_path / "curve.png")
    spec_curve(_fake_effects(), out=out)
    assert os.path.exists(out) and os.path.getsize(out) > 10_000


def test_spec_curve_with_no_sign_flip(tmp_path):
    """Title logic branches on whether the sign flips; exercise both branches."""
    out = str(tmp_path / "curve_noflip.png")
    spec_curve(_fake_effects(flip_sign=False), out=out)
    assert os.path.exists(out)


def test_spec_curve_with_no_common_subgrid(tmp_path):
    eff = _fake_effects()
    eff["is_common"] = False
    out = str(tmp_path / "curve_nocommon.png")
    spec_curve(eff, out=out)
    assert os.path.exists(out)


def test_variance_bar_renders(tmp_path):
    vd = pd.DataFrame({"axis": AXES,
                       "eta_sq": np.linspace(0.31, 0.0, len(AXES)),
                       "n_levels": [2] * len(AXES)})
    out = str(tmp_path / "var.png")
    variance_bar(vd, out=out)
    assert os.path.exists(out) and os.path.getsize(out) > 5_000
