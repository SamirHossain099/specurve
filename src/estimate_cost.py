"""Estimate grid cost for a cohort before committing to it.

MFDFA cost scales with record length, and cohorts differ enormously: Fantasia records are
~7,000 RR intervals, nsrdb ~99,000. Running the full grid blind on a long-record cohort is how
you end up with an overnight job you did not budget for. Benchmark first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on
import time  # noqa: E402

import numpy as np  # noqa: E402

from data import load_cached  # noqa: E402
from run_grid import estimate  # noqa: E402
from specs import build_grid  # noqa: E402


def bench(cohort, n_specs=12, n_records=3, seed=0):
    data = load_cached(cohort)
    if not data:
        print(f"{cohort}: nothing cached -- run `python src/data.py {cohort}` first")
        return None
    names = sorted(data)
    lens = np.array([len(data[n]) for n in names])
    print(f"{cohort}: {len(names)} records | length min={lens.min():,} "
          f"median={int(np.median(lens)):,} max={lens.max():,}")

    rng = np.random.default_rng(seed)
    grid = build_grid()
    picks = rng.choice(len(grid), min(n_specs, len(grid)), replace=False)
    # Use the LONGEST records: cost is dominated by them, so this is the honest estimate.
    recs = [data[n] for n in sorted(names, key=lambda n: -len(data[n]))[:n_records]]

    t0 = time.time()
    n = 0
    for si in picks:
        for rr in recs:
            estimate(rr, grid[si])
            n += 1
    per = (time.time() - t0) / n

    total_fits = len(grid) * len(names)
    cpu_h = per * total_fits / 3600
    print(f"  {per*1000:.1f} ms/fit (worst-case records)")
    print(f"  full grid = {len(grid):,} specs x {len(names)} records = {total_fits:,} fits")
    print(f"  -> {cpu_h:.1f} CPU-hours; {cpu_h/resources.THREADS:.1f} h wall on "
          f"{resources.THREADS} workers")
    if cpu_h / resources.THREADS > 3:
        print("  WARNING: over 3 h wall. Consider truncating records or subsetting the grid.")
    return dict(cohort=cohort, ms_per_fit=per * 1000, total_fits=total_fits,
                cpu_hours=cpu_h, wall_hours=cpu_h / resources.THREADS)

if __name__ == "__main__":
    resources.report("resources: ")
    for c in (sys.argv[1:] or ["fantasia", "nsrdb", "chf2db"]):
        print()
        bench(c)
