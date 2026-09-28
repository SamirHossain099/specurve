"""Run the frozen grid against series whose scaling exponent is known.

Everything else in this study measures how far an estimate moves. It cannot say which direction is
wrong, because the interbeat cohorts have no ground truth. Fractional Gaussian noise does: a series
generated at Hurst exponent H has DFA alpha = H, so the same 9,216 specifications can be scored as
bias rather than as spread.

Two arms:

* **bias arm** -- fGn at several H, at the length of the primary cohort, through the whole grid.
  Reports bias per axis level, which turns "this axis moves the answer" into "this axis moves the
  answer away from the truth, by this much, in this direction".
* **ectopic arm** -- the same series with ectopic beats injected at a known rate and known
  positions, run through the four ectopic levels with the other axes at conventional defaults.
  Deletion against interpolation against no editing, scored against the exponent of the clean
  series. Section 5.1 argues this mechanism from the literature; here it is measured.

Post hoc: added 2026-09-28, recorded as deviation D4 in PREREGISTRATION.md. The grid is untouched.

    python src/synthetic.py                 # both arms, about 4 minutes on 8 cores
    python src/synthetic.py --quick         # 4 series per H, for a smoke test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from run_grid import estimate  # noqa: E402
from specs import build_grid  # noqa: E402

HURSTS = (0.6, 0.7, 0.8, 0.9)
N_SERIES = 20
LENGTH = 6977                      # the median usable Fantasia record
MEAN_RR, SD_RR = 800.0, 40.0       # ms, so the injected beats are physiologically shaped
ECTOPIC_RATES = (0.0, 0.01, 0.05)
OUT = "results"


def fgn(n, hurst, rng):
    """Fractional Gaussian noise by the Davies-Harte method, which is exact rather than approximate.

    The autocovariance of fGn is gamma(k) = (|k-1|^2H - 2|k|^2H + |k+1|^2H)/2. Embedding it in a
    circulant matrix and taking the FFT gives a spectrum to draw from directly; the method fails
    loudly (negative eigenvalues) instead of quietly returning the wrong correlation structure.
    """
    k = np.arange(n)
    g = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
               + np.abs(k + 1) ** (2 * hurst))
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    if lam.min() < -1e-10:
        raise ValueError(f"circulant embedding failed at H={hurst}: negative eigenvalue")
    lam = np.clip(lam, 0, None)
    m = len(row)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    return np.fft.fft(np.sqrt(lam / (2 * m)) * z).real[:n]


def rr_series(n, hurst, rng):
    """fGn rescaled to interbeat intervals in milliseconds. Rescaling cannot change alpha."""
    x = fgn(n, hurst, rng)
    return MEAN_RR + SD_RR * (x - x.mean()) / (x.std() or 1.0)


def inject_ectopic(rr, rate, rng):
    """Premature beat plus compensatory pause, at known positions.

    The pair is what an ectopic beat does to an interval series: one interval much shorter than its
    neighbours, the next much longer, with the local mean preserved. Positions are kept so the arm
    can be scored against the clean series rather than against an editing rule's own opinion.
    """
    rr = np.asarray(rr, float).copy()
    if rate <= 0:
        return rr, np.array([], dtype=int)
    n_ect = int(round(rate * len(rr)))
    idx = np.sort(rng.choice(np.arange(4, len(rr) - 4), size=n_ect, replace=False))
    for i in idx:
        short = rr[i] * rng.uniform(0.50, 0.65)
        rr[i + 1] = rr[i + 1] + (rr[i] - short)
        rr[i] = short
    return rr, idx


# ------------------------------------------------------------------ arm 1: bias across the grid
def bias_arm(grid, n_series, n_jobs):
    rng = np.random.default_rng(20260928)
    series = [(h, s, rr_series(LENGTH, h, rng)) for h in HURSTS for s in range(n_series)]

    def one(si, spec):
        rows = []
        for h, s, rr in series:
            e = estimate(rr, spec)
            rows.append(dict(spec_id=si, hurst=h, series=s, **spec,
                             alpha=e["alpha"], ok=e["ok"]))
        return rows

    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(i, sp) for i, sp in enumerate(grid))
    df = pd.DataFrame([r for rows in out for r in rows])
    df["bias"] = df.alpha - df.hurst
    return df


# ------------------------------------------------------------------ arm 2: ectopic editing vs truth
def ectopic_arm(n_series, n_jobs):
    rng = np.random.default_rng(20260929)
    base = dict(detrend="none", normalise="none", order=1, s_min=16, s_max_frac=0.1,
                n_scales=20, spacing="log", overlap=False, q_grid="coarse")
    hurst = 0.8
    jobs = []
    for s in range(n_series):
        clean = rr_series(LENGTH, hurst, rng)
        for rate in ECTOPIC_RATES:
            dirty, idx = inject_ectopic(clean, rate, rng)
            jobs.append((s, rate, len(idx), dirty))

    def one(s, rate, n_ect, series):
        rows = []
        for level in ("none", "pct20_drop", "pct20_interp", "mad4_drop"):
            e = estimate(series, dict(base, ectopic=level))
            rows.append(dict(series=s, rate=rate, n_ectopic=n_ect, ectopic=level,
                             alpha=e["alpha"], n_used=e["n_used"], ok=e["ok"]))
        return rows

    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df["bias"] = df.alpha - hurst
    df["hurst"] = hurst
    return df


def summarise(bias, ect):
    axes = [c for c in ("ectopic", "detrend", "normalise", "order", "s_min", "s_max_frac",
                        "n_scales", "spacing", "overlap", "q_grid") if c in bias.columns]
    rows = []
    for ax in axes:
        for level, g in bias.groupby(ax):
            rows.append(dict(axis=ax, level=str(level), mean_bias=g.bias.mean(),
                             rmse=float(np.sqrt((g.bias ** 2).mean())),
                             abs_bias=float(g.bias.abs().mean()), n=len(g)))
    per_level = pd.DataFrame(rows).sort_values(["axis", "level"])

    spread = (bias.groupby(["hurst", "series"]).alpha.agg(["min", "max"])
              .assign(span=lambda d: d["max"] - d["min"]))
    summary = dict(
        n_fits=int(len(bias)),
        hursts=list(HURSTS),
        length=LENGTH,
        median_bias=float(bias.bias.median()),
        iqr_bias=[float(bias.bias.quantile(0.25)), float(bias.bias.quantile(0.75))],
        median_span_within_series=float(spread.span.median()),
        worst_axis_by_rmse=per_level.sort_values("rmse").iloc[-1][["axis", "level", "rmse"]].to_dict(),
        conventional_bias=float(
            bias[(bias.s_max_frac == 0.1) & (bias.s_min == 16) & (bias.order == 1)
                 & (bias.spacing == "log") & (~bias.overlap.astype(bool))].bias.median()),
    )
    ect_summary = (ect.groupby(["rate", "ectopic"]).bias
                   .agg(median_bias="median", mean_abs=lambda s: s.abs().mean())
                   .reset_index())
    return per_level, summary, ect_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="4 series per H, for a smoke test")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    n_series = 4 if a.quick else N_SERIES

    grid = build_grid()
    t0 = time.time()
    bias = bias_arm(grid, n_series, a.n_jobs)
    print(f"bias arm: {len(bias):,} fits in {time.time() - t0:.0f} s")
    t1 = time.time()
    ect = ectopic_arm(n_series, a.n_jobs)
    print(f"ectopic arm: {len(ect):,} fits in {time.time() - t1:.0f} s")

    os.makedirs(OUT, exist_ok=True)
    bias.to_parquet(f"{OUT}/synthetic_grid.parquet", index=False)
    per_level, summary, ect_summary = summarise(bias, ect)
    per_level.to_csv(f"{OUT}/synthetic_bias_by_level.csv", index=False)
    ect.to_csv(f"{OUT}/synthetic_ectopic.csv", index=False)
    ect_summary.to_csv(f"{OUT}/synthetic_ectopic_summary.csv", index=False)
    with open(f"{OUT}/synthetic_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(ect_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
