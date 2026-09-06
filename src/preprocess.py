"""RR-series preprocessing -- every step is an explicit, enumerable analytic choice.

These are the decisions that papers almost never report. Each function here is one axis of
the specification grid; nothing is applied silently.
"""
import numpy as np

# ---------------------------------------------------------------- ectopic beats

def ectopic_none(rr):
    return rr.copy()


def ectopic_pct(rr, pct=0.20):
    """Drop intervals differing from the previous one by more than `pct` (Malik-style)."""
    rr = np.asarray(rr, float)
    keep = np.ones(len(rr), bool)
    prev = rr[0]
    for i in range(1, len(rr)):
        if abs(rr[i] - prev) > pct * prev:
            keep[i] = False
        else:
            prev = rr[i]
    return rr[keep]


def ectopic_madsd(rr, k=4.0):
    """Drop intervals whose successive difference exceeds k robust SDs of dRR."""
    rr = np.asarray(rr, float)
    d = np.diff(rr, prepend=rr[0])
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    if mad <= 0:
        return rr.copy()
    return rr[np.abs(d - np.median(d)) <= k * mad]


def ectopic_interp_pct(rr, pct=0.20):
    """Same detection as `ectopic_pct` but linearly interpolate instead of deleting.

    Deleting shortens the series and splices unrelated segments together; interpolating
    preserves length but invents data. Both are defensible and both are in the literature --
    which is exactly why this is an axis rather than a default.
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


ECTOPIC = {
    "none": ectopic_none,
    "pct20_drop": lambda rr: ectopic_pct(rr, 0.20),
    "pct20_interp": lambda rr: ectopic_interp_pct(rr, 0.20),
    "mad4_drop": lambda rr: ectopic_madsd(rr, 4.0),
}


# ------------------------------------------------------------------- detrending

def detrend_none(rr):
    return rr.copy()


def detrend_poly(rr, order=1):
    rr = np.asarray(rr, float)
    t = np.arange(len(rr))
    return rr - np.polyval(np.polyfit(t, rr, order), t)


DETREND = {
    "none": detrend_none,
    "linear": lambda rr: detrend_poly(rr, 1),
    "quadratic": lambda rr: detrend_poly(rr, 2),
}


# ---------------------------------------------------------------- normalisation

NORMALISE = {
    "none": lambda rr: np.asarray(rr, float),
    "zscore": lambda rr: (np.asarray(rr, float) - np.mean(rr)) / (np.std(rr) or 1.0),
}


def apply_spec(rr, spec):
    """Run the preprocessing chain named by a specification dict."""
    x = ECTOPIC[spec["ectopic"]](np.asarray(rr, float))
    x = DETREND[spec["detrend"]](x)
    x = NORMALISE[spec["normalise"]](x)
    return x
