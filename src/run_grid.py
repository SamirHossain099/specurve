"""Run every specification against every record and record the resulting estimates.

Output: one row per (specification, record) with the DFA/MFDFA estimates. The group-level
effect sizes and the specification curve are computed downstream in analyze.py, so that the
expensive part runs once.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from data import fantasia_group, load_cached  # noqa: E402
from mfdfa import hurst_from_fluct, mfdfa, singularity_width  # noqa: E402
from preprocess import apply_spec  # noqa: E402
from specs import Q_GRID, build_grid, is_common  # noqa: E402


def estimate(rr, spec):
    """Return the estimates for one (series, specification) pair."""
    try:
        x = apply_spec(rr, spec)
        if len(x) < 200:
            return dict(alpha=np.nan, delta_alpha=np.nan, n_used=len(x), ok=False)
        qs = Q_GRID[spec["q_grid"]]
        scales, F = mfdfa(
            x, qs,
            s_min=spec["s_min"],
            s_max_frac=spec["s_max_frac"],
            n_scales=spec["n_scales"],
            order=spec["order"],
            spacing=spec["spacing"],
            overlap=spec["overlap"],
        )
        if len(scales) < 4:
            return dict(alpha=np.nan, delta_alpha=np.nan, n_used=len(x), ok=False)
        hq = hurst_from_fluct(scales, F)
        alpha = float(hq[int(np.argmin(np.abs(qs - 2.0)))])
        return dict(alpha=alpha, delta_alpha=singularity_width(qs, hq),
                    n_used=len(x), n_scales_used=len(scales), ok=True)
    except Exception:
        return dict(alpha=np.nan, delta_alpha=np.nan, n_used=0, ok=False)

def _one_spec(si, spec, recs, names, groups):
    rows = []
    for rec, rr, grp in zip(names, recs, groups):
        e = estimate(rr, spec)
        rows.append(dict(spec_id=si, record=rec, group=grp, **spec, **e))
    return rows

def main(cohort="fantasia", n_jobs=None, limit_specs=None, out="results/grid_fantasia.parquet",
         truncate=None):
    """`truncate` keeps only the first N intervals of every record.

    Added AFTER the main results, as a post-hoc control -- see PREREGISTRATION.md section 6.
    The specification grid itself is untouched; record length is an extra factor crossed with
    it. It exists because the primary contrast (Fantasia, ~7k intervals) and the replication
    contrast (nsrdb/chf2db, ~100k) differ in length by 14x, so "the dominant analytic choice
    changes with the population" is confounded with record length until this is run. Records
    shorter than N are kept whole and their true length is recorded in `n_input`.
    """
    data = load_cached(cohort)
    if not data:
        raise SystemExit(f"no cached records for {cohort}; run src/data.py {cohort} first")
    names = sorted(data)
    recs = [data[n] for n in names]
    if truncate:
        recs = [r[:truncate] for r in recs]
        kept = [len(r) for r in recs]
        print(f"truncated to first {truncate:,} intervals: "
              f"lengths now {min(kept):,}-{max(kept):,} "
              f"({sum(k < truncate for k in kept)}/{len(kept)} records were already shorter)")
    groups = [fantasia_group(n) if cohort == "fantasia" else cohort for n in names]
    resources.report("resources: ")
    print(f"{cohort}: {len(names)} records, groups={pd.Series(groups).value_counts().to_dict()}")

    if n_jobs is None or n_jobs <= 0:
        n_jobs = resources.THREADS
    grid = build_grid()
    if limit_specs:
        grid = grid[:limit_specs]
    print(f"specifications: {len(grid):,}  ->  {len(grid)*len(names):,} fits")

    t0 = time.time()
    # max_nbytes=None keeps joblib from spilling memmaps into TEMP, which
    # previously filled disk during long sweeps.
    batches = Parallel(n_jobs=n_jobs, verbose=5, batch_size=16,
                       max_nbytes=None)(
        delayed(_one_spec)(si, sp, recs, names, groups) for si, sp in enumerate(grid)
    )
    rows = [r for b in batches for r in b]
    df = pd.DataFrame(rows)
    df["is_common"] = [is_common(g) for g in grid for _ in names]
    df["truncate"] = truncate if truncate else 0
    df["n_input"] = [len(r) for _ in grid for r in recs]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    el = time.time() - t0
    print(f"\n{len(df):,} rows in {el:.0f}s ({el/len(df)*1000:.2f} ms/fit) -> {out}")
    print(f"failed fits: {(~df.ok).sum():,} ({100*(~df.ok).mean():.2f}%)")
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="fantasia")
    ap.add_argument("--n-jobs", type=int, default=0,
                    help="0 = auto (capped, leaves cores free)")
    ap.add_argument("--limit-specs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--truncate", type=int, default=None,
                    help="keep only the first N RR intervals of each record (post-hoc control)")
    a = ap.parse_args()
    default_out = (f"results/grid_{a.cohort}.parquet" if not a.truncate
                   else f"results/grid_{a.cohort}_trunc{a.truncate}.parquet")
    main(a.cohort, a.n_jobs, a.limit_specs, a.out or default_out, truncate=a.truncate)
