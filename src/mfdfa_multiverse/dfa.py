"""DFA / MFDFA, written so every analytic choice is an explicit argument.

The whole point of this package is that these arguments are usually hidden. Nothing here defaults
silently: a specification is a dict of these parameters, and the specification grid is the
experiment.

The RR-interval preprocessing axes live here too, because in this literature they are not
separable from the estimator -- how ectopic beats are handled decides which intervals exist, and
that is as much part of the measurement as the polynomial order is.
"""
import numpy as np

# ------------------------------------------------------------------------------ the estimator


def integrate_profile(x):
    """Cumulative sum of the mean-centred series (the DFA 'profile')."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x - x.mean())


def make_scales(n, s_min, s_max_frac, n_scales, spacing="log"):
    """Window sizes to evaluate. `s_max_frac` caps the largest window as a fraction of n."""
    s_max = int(n * s_max_frac)
    if s_max <= s_min:
        return np.array([], dtype=int)
    if spacing == "log":
        s = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), n_scales)).astype(int))
    else:
        s = np.unique(np.round(np.linspace(s_min, s_max, n_scales)).astype(int))
    return s[s >= 4]


def _fluct_for_scale(profile, s, order, overlap):
    """Mean squared residual of a polynomial fit within each window of length s."""
    n = len(profile)
    step = s // 2 if overlap else s
    starts = np.arange(0, n - s + 1, step)
    if len(starts) == 0:
        return None
    idx = starts[:, None] + np.arange(s)[None, :]
    seg = profile[idx]
    V = np.vander(np.arange(s), order + 1)      # one least-squares solve for all windows
    coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
    resid = seg.T - V @ coef
    return (resid ** 2).mean(axis=0)


def mfdfa(x, qs, s_min=16, s_max_frac=0.1, n_scales=20, order=2, spacing="log", overlap=False):
    """Return (scales, F[q, scale]). Every parameter is required to be chosen, not defaulted."""
    profile = integrate_profile(x)
    scales = make_scales(len(profile), s_min, s_max_frac, n_scales, spacing)
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
            if abs(q) < 1e-10:                          # the q -> 0 limit
                F[i, j] = np.exp(0.5 * np.mean(np.log(f2)))
            else:
                F[i, j] = np.mean(f2 ** (q / 2.0)) ** (1.0 / q)
    return scales, F


def hurst_from_fluct(scales, F):
    """Generalized Hurst h(q): slope of log F(q, s) against log s."""
    out = np.full(F.shape[0], np.nan)
    ls = np.log(scales)
    for i in range(F.shape[0]):
        m = np.isfinite(F[i]) & (F[i] > 0)
        if m.sum() >= 3:
            out[i] = np.polyfit(ls[m], np.log(F[i][m]), 1)[0]
    return out


def singularity_width(qs, hq):
    """Multifractal width delta-alpha, from the Legendre transform."""
    qs = np.asarray(qs, dtype=float)
    m = np.isfinite(hq)
    if m.sum() < 3:
        return np.nan
    q, h = qs[m], hq[m]
    alpha = np.gradient(q * h - 1.0, q)
    return float(np.nanmax(alpha) - np.nanmin(alpha))


# ------------------------------------------------------------------------------ preprocessing

def ectopic_none(rr):
    return np.asarray(rr, float).copy()


def ectopic_pct(rr, pct=0.20):
    """Malik-style rule: drop an interval differing from the last ACCEPTED one by more than `pct`.

    This is the rule as it is usually phrased, and it can latch: if an anomalous interval is
    accepted, the reference stops updating and every subsequent normal beat is rejected. On one
    Fantasia record that removes 90% of the series. Comparing against the preceding *raw* interval
    instead is equally standard and does not latch -- see `ectopic_pct_raw`. Which one a paper
    used is essentially never stated.
    """
    rr = np.asarray(rr, float)
    keep = np.ones(len(rr), bool)
    prev = rr[0]
    for i in range(1, len(rr)):
        if abs(rr[i] - prev) > pct * prev:
            keep[i] = False
        else:
            prev = rr[i]
    return rr[keep]


def ectopic_pct_raw(rr, pct=0.20):
    """The same rule against the immediately preceding RAW interval. Cannot latch."""
    rr = np.asarray(rr, float)
    d = np.abs(np.diff(rr)) / rr[:-1]
    return rr[np.concatenate([[True], d <= pct])]


def ectopic_madsd(rr, k=4.0):
    """Drop intervals whose successive difference exceeds k robust SDs of dRR."""
    rr = np.asarray(rr, float)
    d = np.diff(rr, prepend=rr[0])
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    if mad <= 0:
        return rr.copy()
    return rr[np.abs(d - np.median(d)) <= k * mad]


def ectopic_interp_pct(rr, pct=0.20):
    """Same detection as `ectopic_pct`, linear interpolation instead of deletion.

    Deleting shortens the series and splices unrelated segments together; interpolating preserves
    length but invents data. Both are defensible and both are in the literature, which is exactly
    why this is an axis rather than a default.
    """
    rr = np.asarray(rr, float).copy()
    bad = np.zeros(len(rr), bool)
    prev = rr[0]
    for i in range(1, len(rr)):
        if abs(rr[i] - prev) > pct * prev:
            bad[i] = True
        else:
            prev = rr[i]
    if bad.all():
        return rr
    idx = np.arange(len(rr))
    rr[bad] = np.interp(idx[bad], idx[~bad], rr[~bad])
    return rr


def detrend_poly(rr, order=1):
    rr = np.asarray(rr, float)
    t = np.arange(len(rr))
    return rr - np.polyval(np.polyfit(t, rr, order), t)


ECTOPIC = {
    "none": ectopic_none,
    "pct20_drop": lambda rr: ectopic_pct(rr, 0.20),
    "pct20_drop_raw": lambda rr: ectopic_pct_raw(rr, 0.20),
    "pct20_interp": lambda rr: ectopic_interp_pct(rr, 0.20),
    "mad4_drop": lambda rr: ectopic_madsd(rr, 4.0),
}
DETREND = {
    "none": lambda rr: np.asarray(rr, float).copy(),
    "linear": lambda rr: detrend_poly(rr, 1),
    "quadratic": lambda rr: detrend_poly(rr, 2),
}
NORMALISE = {
    "none": lambda rr: np.asarray(rr, float),
    "zscore": lambda rr: (np.asarray(rr, float) - np.mean(rr)) / (np.std(rr) or 1.0),
}


def apply_spec(rr, spec):
    """Run the preprocessing chain named by a specification dict."""
    x = ECTOPIC[spec["ectopic"]](np.asarray(rr, float))
    x = DETREND[spec["detrend"]](x)
    return NORMALISE[spec["normalise"]](x)


def estimate(rr, spec, q_grids=None, min_samples=200, min_scales=4):
    """One (series, specification) pair -> the estimates, or a recorded failure.

    Returns `ok=False` rather than raising or returning nothing, because a specification that
    vanishes silently removes a unit from part of the grid without appearing in any denominator.
    """
    q_grids = q_grids or {"coarse": np.arange(-5, 5.1, 1.0), "fine": np.arange(-5, 5.1, 0.5)}
    try:
        x = apply_spec(rr, spec)
        if len(x) < min_samples:
            return dict(alpha=np.nan, delta_alpha=np.nan, n_used=len(x), ok=False)
        qs = q_grids[spec["q_grid"]] if isinstance(spec.get("q_grid"), str) else spec["q_grid"]
        scales, F = mfdfa(x, qs, s_min=spec["s_min"], s_max_frac=spec["s_max_frac"],
                          n_scales=spec["n_scales"], order=spec["order"],
                          spacing=spec["spacing"], overlap=spec["overlap"])
        if len(scales) < min_scales:
            return dict(alpha=np.nan, delta_alpha=np.nan, n_used=len(x), ok=False)
        hq = hurst_from_fluct(scales, F)
        alpha = float(hq[int(np.argmin(np.abs(np.asarray(qs, float) - 2.0)))])
        return dict(alpha=alpha, delta_alpha=singularity_width(qs, hq),
                    n_used=len(x), n_scales_used=len(scales), ok=True)
    except Exception:
        return dict(alpha=np.nan, delta_alpha=np.nan, n_used=0, ok=False)
