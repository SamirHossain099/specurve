"""Record length as a crossed factor -- the control the cross-cohort claim needs.

Why this exists
---------------
The headline of the replication is that **the dominant analytic choice changes with the
population**: the scale range (`s_max_frac`) dominates the healthy-aging contrast, ectopic-beat
handling dominates the heart-failure contrast. That claim is confounded on arrival. Fantasia's
records hold ~7,000 RR intervals; nsrdb and chf2db hold ~100,000. A referee's first question is
whether the axis that dominates is a property of the cohort or a property of the record length,
and nothing in the primary analysis can separate them.

So every cohort is re-run through the **frozen** grid at matched truncations. Two questions:

1. **Length-matched replication.** Truncate all three cohorts to 4,731 intervals -- the shortest
   Fantasia record, so every record in every cohort is exactly that long and no record is ragged.
   If `ectopic` still dominates the disease contrast and `s_max_frac` still dominates the aging
   contrast at identical length, the population claim survives its own confound.
2. **Dose-response.** Run a ladder of lengths and watch the effect size, the width of the curve
   and the eta^2 ranking move. This is also the short-record honesty control the brief asked for:
   DFA is most contested exactly where records are short, and the ladder shows what happens there
   rather than asserting it.

The grid is not touched. Record length is an extra factor crossed with the pre-registered
specifications, added after the primary results and recorded as a deviation in
PREREGISTRATION.md section 6.

Resumable: each (cohort, length) grid is written to its own parquet and skipped if present, so an
interrupted run loses at most the rung in flight.
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

from analyze import effect_table, summarise, variance_decomposition  # noqa: E402
from run_grid import main as run_grid_main  # noqa: E402

# The shortest Fantasia record. Truncating every cohort here makes every record in the study
# exactly this long, which is what "length-matched" has to mean if it is to carry any weight.
MATCHED = 4731

LADDER = {
    "fantasia": [1000, 2000, 4000, MATCHED],
    "nsrdb": [1000, 2000, 4000, MATCHED, 8000, 16000, 32000],
    "chf2db": [1000, 2000, 4000, MATCHED, 8000, 16000, 32000],
}

# The full-length runs already exist and are the top rung of each ladder; they are read from
# disk rather than recomputed.
FULL = {"fantasia": "results/grid_fantasia.parquet",
        "nsrdb": "results/grid_nsrdb.parquet",
        "chf2db": "results/grid_chf2db.parquet"}

CONTRASTS = {
    "aging": dict(cohorts=["fantasia"], g1="young", g2="old"),
    "disease": dict(cohorts=["nsrdb", "chf2db"], g1="chf2db", g2="nsrdb"),
}


def grid_path(cohort, n):
    return f"results/grid_{cohort}_trunc{n}.parquet"


def ensure_grid(cohort, n, n_jobs=0):
    """Run one (cohort, length) grid unless its parquet is already on disk."""
    path = grid_path(cohort, n)
    if os.path.exists(path):
        print(f"  [skip] {path} exists")
        return path
    t0 = time.time()
    run_grid_main(cohort, n_jobs or None, None, path, truncate=n)
    print(f"  [done] {path} in {time.time() - t0:.0f}s", flush=True)
    return path


def load_rung(cohort, n):
    """One rung of the ladder. `n=None` means the full-length run.

    The full-length grids were written before `run_grid` recorded `n_input`, so that column has
    to be reconstructed for them from the cached RR series -- otherwise the full-length rung
    plots at x = NaN and silently disappears from the figure and the table.
    """
    path = FULL[cohort] if n is None else grid_path(cohort, n)
    df = pd.read_parquet(path)
    if "n_input" not in df.columns:
        from data import load_cached
        lengths = {k: len(v) for k, v in load_cached(cohort).items()}
        missing = set(df.record.unique()) - set(lengths)
        if missing:
            raise SystemExit(f"{cohort}: no cached series for {sorted(missing)[:3]}...; "
                             f"run src/data.py {cohort}")
        df["n_input"] = df.record.map(lengths).astype("int64")
    return df


def rung_row(contrast, n, metric, n_jobs=0):
    """Summarise one (contrast, length, metric) cell of the design."""
    spec = CONTRASTS[contrast]
    frames = []
    for c in spec["cohorts"]:
        if n is not None:
            ensure_grid(c, n, n_jobs)
        frames.append(load_rung(c, n))
    df = pd.concat(frames, ignore_index=True)

    eff = effect_table(df, metric=metric, g1=spec["g1"], g2=spec["g2"])
    if eff.empty:
        return None
    s = summarise(eff)
    vd = variance_decomposition(eff)
    top = vd.iloc[0]
    second = vd.iloc[1]
    lengths = df.groupby("record").n_input.first().to_numpy(dtype=float)
    return dict(
        contrast=contrast, metric=metric,
        truncate=(0 if n is None else n),
        median_length=float(np.nanmedian(lengths)),
        n_records=int(df.record.nunique()),
        d_median=s["d_median"], d_min=s["d_min"], d_max=s["d_max"],
        d_span=s["d_max"] - s["d_min"],
        frac_p05=s["frac_p05"], frac_sig_positive=s["frac_sig_positive"],
        frac_sig_negative=s["frac_sig_negative"], sign_flips=s["sign_flips"],
        common_d_median=s["common_d_median"],
        common_d_min=s["common_d_range"][0], common_d_max=s["common_d_range"][1],
        top_axis=top.axis, top_eta_sq=float(top.eta_sq),
        second_axis=second.axis, second_eta_sq=float(second.eta_sq),
        frac_failed=float((~df.ok).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", default=["alpha", "delta_alpha"])
    ap.add_argument("--contrasts", nargs="+", default=list(CONTRASTS))
    ap.add_argument("--matched-only", action="store_true",
                    help="run only the length-matched rung, skipping the dose-response ladder")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--out", default="results/length_ladder.csv")
    a = ap.parse_args()

    resources.report("resources: ")
    rows = []
    if os.path.exists(a.out):
        rows = pd.read_csv(a.out).to_dict("records")
        print(f"resuming from {a.out} with {len(rows)} rows")
    done = {(r["contrast"], r["metric"], int(r["truncate"])) for r in rows}

    for contrast in a.contrasts:
        cohorts = CONTRASTS[contrast]["cohorts"]
        rungs = sorted(set.intersection(*[set(LADDER[c]) for c in cohorts]))
        if a.matched_only:
            rungs = [MATCHED]
        for n in [*rungs, None]:                 # None = the existing full-length run
            key_n = 0 if n is None else n
            for metric in a.metrics:
                if (contrast, metric, key_n) in done:
                    print(f"[skip] {contrast} {metric} n={key_n}")
                    continue
                print(f"\n=== {contrast} / {metric} / truncate={key_n or 'full'}", flush=True)
                r = rung_row(contrast, n, metric, a.n_jobs)
                if r is None:
                    print("  no usable specifications at this rung")
                    continue
                rows.append(r)
                pd.DataFrame(rows).to_csv(a.out, index=False)   # write after every cell
                print(f"  d_median={r['d_median']:+.3f}  span={r['d_span']:.2f}  "
                      f"p05={r['frac_p05']:.3f}  top={r['top_axis']} "
                      f"(eta^2={r['top_eta_sq']:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)")

    # `df["truncate"]`, not `df.truncate` -- the latter resolves to DataFrame.truncate, the
    # method, so the comparison yields a scalar False and `df[False]` raises KeyError. Same trap
    # as in src/figure_length.py; a column named after a DataFrame method is worth a comment in
    # both places rather than a second debugging session.
    matched = df[df["truncate"] == MATCHED]
    verdict = {}
    for metric in a.metrics:
        m = matched[matched.metric == metric]
        verdict[metric] = {r.contrast: dict(top_axis=r.top_axis, top_eta_sq=round(r.top_eta_sq, 4))
                           for r in m.itertuples()}
    print("\n=== LENGTH-MATCHED VERDICT (all records exactly "
          f"{MATCHED:,} intervals) ===")
    print(json.dumps(verdict, indent=2))
    with open("results/length_matched_verdict.json", "w") as fh:
        json.dump(dict(matched_length=MATCHED, verdict=verdict), fh, indent=2)


if __name__ == "__main__":
    main()
