"""Replication analysis across cohorts.

Fantasia gives a WITHIN-cohort contrast (young vs old). The replication cohorts do not: nsrdb
is healthy, chf2db is congestive heart failure, and neither has an internal age split. So the
replication asks a deliberately different question:

    Fantasia   : does the AGING effect survive the specification grid?
    nsrdb/chf2db: does the INSTABILITY itself replicate on a different contrast and a
                  very different record length (~100k intervals vs ~7k)?

Both cohorts are run through the identical frozen grid, so `spec_id` is comparable across
files and the two curves can be placed side by side.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import (  # noqa: E402
    effect_table,
    permutation_test_median,
    summarise,
    variance_decomposition,
)


def merge_cohorts(paths):
    """Concatenate per-cohort grid outputs. `spec_id` is comparable because the grid is frozen."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  missing, skipping: {p}")
            continue
        df = pd.read_parquet(p)
        frames.append(df)
        print(f"  {os.path.basename(p)}: {len(df):,} rows, "
              f"{df.spec_id.nunique():,} specs, groups={sorted(df.group.unique())}")
    if not frames:
        raise SystemExit("no cohort files found")
    out = pd.concat(frames, ignore_index=True)
    # Guard: the grids must actually match, or spec_id comparisons are meaningless.
    per = [set(f.spec_id.unique()) for f in frames]
    common = set.intersection(*per)
    if len(per) > 1 and any(len(s) != len(common) for s in per):
        print(f"  WARNING: spec_id sets differ; restricting to {len(common):,} shared specs")
        out = out[out.spec_id.isin(common)]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["nsrdb", "chf2db"])
    ap.add_argument("--metric", default="alpha")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()

    resources.report("resources: ")
    paths = [f"results/grid_{c}.parquet" for c in a.cohorts]
    print(f"merging {len(paths)} cohorts:")
    df = merge_cohorts(paths)

    groups = sorted(df.group.unique())
    if len(groups) != 2:
        raise SystemExit(f"need exactly 2 groups for a contrast, got {groups}")
    g1, g2 = groups
    print(f"\ncontrast: {g1} (n={df[df.group==g1].record.nunique()}) vs "
          f"{g2} (n={df[df.group==g2].record.nunique()})")

    prefix = a.out_prefix or f"repl_{'_vs_'.join(groups)}"
    eff = effect_table(df, metric=a.metric, g1=g1, g2=g2)
    eff.to_csv(f"results/effects_{prefix}_{a.metric}.csv", index=False)

    s = summarise(eff)
    print("\n=== SPECIFICATION CURVE (replication) ===")
    print(json.dumps(s, indent=2))

    vd = variance_decomposition(eff)
    vd.to_csv(f"results/variance_{prefix}_{a.metric}.csv", index=False)
    print("\n=== VARIANCE DECOMPOSITION ===")
    print(vd.to_string(index=False))

    pt = permutation_test_median(df, metric=a.metric, n_perm=a.n_perm)
    print("\n=== PERMUTATION TEST ON THE MEDIAN EFFECT ===")
    print(json.dumps(pt, indent=2))

    with open(f"results/summary_{prefix}_{a.metric}.json", "w") as fh:
        json.dump(dict(contrast=[g1, g2], summary=s, permutation=pt,
                       variance=vd.to_dict("records")), fh, indent=2)
    print(f"\nwrote results/*_{prefix}_{a.metric}.*")

    # The headline the replication is actually testing.
    d = eff.d.to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    print(f"\nINSTABILITY REPLICATES? span={d.min():.2f}..{d.max():.2f}  "
          f"sign_flips={'YES' if (d>0).any() and (d<0).any() else 'no'}  "
          f"frac_p05={float((eff.p<0.05).mean()):.3f}")

if __name__ == "__main__":
    main()
