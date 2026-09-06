"""DFA / MFDFA core, written so every analytic choice is an explicit argument.

The whole point of this project is that these arguments are usually hidden. Nothing here
defaults silently: a specification is a dict of these parameters, and the specification
grid is the experiment.
"""
import numpy as np


def integrate_profile(x):
    """Cumulative sum of the mean-centred series (the DFA 'profile')."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x - x.mean())


def make_scales(n, s_min, s_max_frac, n_scales, spacing="log"):
    """Window sizes to evaluate. s_max_frac caps the largest window as a fraction of n."""
    s_max = int(n * s_max_frac)
    if s_max <= s_min:
        return np.array([], dtype=int)
    if spacing == "log":
        s = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), n_scales)).astype(int))
    else:
        s = np.unique(np.round(np.linspace(s_min, s_max, n_scales)).astype(int))
    return s[s >= 4]


def _fluct_for_scale(profile, s, order, overlap):
    """RMS residual of a polynomial fit within each window of length s."""
    n = len(profile)
    step = s // 2 if overlap else s
    starts = np.arange(0, n - s + 1, step)
    if len(starts) == 0:
        return None
    idx = starts[:, None] + np.arange(s)[None, :]
    seg = profile[idx]
    t = np.arange(s)
    # Vandermonde least squares once for all windows at this scale.
    V = np.vander(t, order + 1)
    coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
    resid = seg.T - V @ coef
    return (resid ** 2).mean(axis=0)


def mfdfa(x, qs, s_min=16, s_max_frac=0.1, n_scales=20, order=2,
          spacing="log", overlap=False):
    """Return (scales, F[q, scale]) for the given q values."""
    profile = integrate_profile(x)
    n = len(profile)
    scales = make_scales(n, s_min, s_max_frac, n_scales, spacing)
    qs = np.asarray(qs, dtype=float)
    F = np.full((len(qs), len(scales)), np.nan)
    for j, s in enumerate(scales):
        f2 = _fluct_for_scale(profile, int(s), order, overlap)
        if f2 is None or not np.all(np.isfinite(f2)):
            continue
        f2 = f2[f2 > 0]
        if f2.size == 0:
            continue
        for i, q in enumerate(qs):
            if abs(q) < 1e-10:                      # q -> 0 limit
                F[i, j] = np.exp(0.5 * np.mean(np.log(f2)))
            else:
                F[i, j] = np.mean(f2 ** (q / 2.0)) ** (1.0 / q)
    return scales, F


def hurst_from_fluct(scales, F):
    """Generalized Hurst h(q): slope of log F(q,s) vs log s."""
    out = np.full(F.shape[0], np.nan)
    ls = np.log(scales)
    for i in range(F.shape[0]):
        m = np.isfinite(F[i]) & (F[i] > 0)
        if m.sum() >= 3:
            out[i] = np.polyfit(ls[m], np.log(F[i][m]), 1)[0]
    return out


def singularity_width(qs, hq):
    """Multifractal width delta-alpha from the Legendre transform."""
    qs = np.asarray(qs, dtype=float)
    m = np.isfinite(hq)
    if m.sum() < 3:
        return np.nan
    q, h = qs[m], hq[m]
    tau = q * h - 1.0
    alpha = np.gradient(tau, q)
    return float(np.nanmax(alpha) - np.nanmin(alpha))
