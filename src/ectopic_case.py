"""One record, one rule, two implementations, a ten-fold difference in surviving data.

Why this exists
---------------
The variance decomposition says ectopic-beat handling is the most consequential analytic choice
in three of the four contrast-by-endpoint cells. That is an eta^2, and an eta^2 is abstract. This
script is the concrete case behind it, and it was not looked for -- it surfaced as a failed-fit
count in the short-record rungs of the length ladder and was traced back.

`f2y05` is a young, healthy, rigorously screened Fantasia subject. In its first 1,000 RR
intervals, **0.5% of successive differences exceed 20%**. The Malik-style 20% rule -- the most
common convention in HRV -- deletes **90.6% of the record**.

The mechanism is a latch, and the rule fails by **accepting** the wrong interval rather than by
rejecting too many. The rule as usually stated compares each interval to the last *accepted* one.
On this record two genuine ectopic beats (indices 89 and 93, 1,524 ms and 1,972 ms) are correctly
removed. Then the post-ectopic recovery ramps: 868 -> 944 -> 1,088 ms, and each step is within 20%
of the one before it, so **1,088 ms is accepted** and becomes the reference. Every subsequent
normal beat of ~780 ms is more than 20% below it, nothing is ever accepted again, and the
reference has no mechanism to walk back down. From index 96 the deletion runs unbroken to the end
of the record.

Compare each interval to the immediately preceding *raw* interval instead -- also standard, also
unreported -- and 995 of 1,000 survive.

Same rule. Same 20% threshold. Same data. 94 intervals against 995, decided by an implementation
detail that no paper states. This is the paper's thesis one level down: an unreported degree of
freedom *inside* a single level of a single axis.

`src/preprocess.py` implements the last-accepted variant, which is the one the rule's usual
phrasing describes. It is not changed here. This script measures the consequence.
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

from data import load_cached  # noqa: E402

PCT = 0.20


def keep_last_accepted(rr, pct=PCT):
    """Compare each interval to the last ACCEPTED interval. What `preprocess.ectopic_pct` does."""
    rr = np.asarray(rr, float)
    keep = np.ones(len(rr), bool)
    prev = rr[0]
    for i in range(1, len(rr)):
        if abs(rr[i] - prev) > pct * prev:
            keep[i] = False
        else:
            prev = rr[i]
    return keep, prev


def keep_previous_raw(rr, pct=PCT):
    """Compare each interval to the immediately preceding RAW interval. Also standard."""
    rr = np.asarray(rr, float)
    d = np.abs(np.diff(rr)) / rr[:-1]
    return np.concatenate([[True], d <= pct])


def analyse(rr):
    a, frozen = keep_last_accepted(rr)
    b = keep_previous_raw(rr)
    dropped = np.where(~a)[0]
    d = np.abs(np.diff(rr)) / rr[:-1]
    first = int(dropped.min()) if dropped.size else None

    # Where the terminal run of deletions begins -- i.e. the latch. Distinct from the first
    # deletion, which on this record is a genuine ectopic beat the rule removes correctly.
    run_start = None
    if dropped.size and dropped.max() == len(rr) - 1:
        run_start = int(dropped.max())
        s = set(dropped.tolist())
        while run_start - 1 in s:
            run_start -= 1
    latch_index = (run_start - 1) if run_start else None      # the last interval ever accepted

    return dict(
        n=int(len(rr)),
        frac_successive_over_threshold=float((d > PCT).mean()),
        kept_last_accepted=int(a.sum()),
        kept_previous_raw=int(b.sum()),
        retention_last_accepted=float(a.mean()),
        retention_previous_raw=float(b.mean()),
        first_deleted_index=first,
        first_deleted_value_ms=(float(rr[first]) if first is not None else None),
        deletion_run_starts_at=run_start,
        deletion_run_length=(int(len(rr) - run_start) if run_start is not None else 0),
        deletion_runs_to_end=bool(dropped.size and dropped.max() == len(rr) - 1),
        latch_index=latch_index,
        latch_value_ms=(float(rr[latch_index]) if latch_index is not None else None),
        reference_frozen_at_ms=float(frozen),
        median_rr_after_latch_ms=(float(np.median(rr[run_start:]))
                                  if run_start is not None else None),
    )


def cohort_table(cohort="fantasia", n=1000):
    """Retention under both variants for every record, so the case is placed in context."""
    out = {}
    for rec, rr in sorted(load_cached(cohort).items()):
        x = rr[:n]
        if len(x) < 100:
            continue
        out[rec] = dict(last_accepted=float(keep_last_accepted(x)[0].mean()),
                        previous_raw=float(keep_previous_raw(x).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="fantasia")
    ap.add_argument("--record", default="f2y05")
    ap.add_argument("--n", type=int, default=1000,
                    help="prefix length; the case was found in the ladder's 1,000-interval rung")
    ap.add_argument("--out", default="results/ectopic_case.json")
    a = ap.parse_args()

    data = load_cached(a.cohort)
    if a.record not in data:
        raise SystemExit(f"{a.record} not cached; run src/data.py {a.cohort}")

    case = analyse(data[a.record][:a.n])
    table = cohort_table(a.cohort, a.n)
    worst = sorted(table.items(), key=lambda kv: kv[1]["last_accepted"])[:5]
    ret = np.array([v["last_accepted"] for v in table.values()])

    payload = dict(cohort=a.cohort, record=a.record, prefix=a.n, threshold=PCT,
                   case=case,
                   cohort_retention_last_accepted=dict(
                       n_records=len(table), min=float(ret.min()),
                       median=float(np.median(ret)), mean=float(ret.mean()),
                       max=float(ret.max()),
                       n_below_half=int((ret < 0.5).sum())),
                   worst_five=[dict(record=k, **v) for k, v in worst])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    c = case
    print(f"=== {a.record}, first {a.n:,} RR intervals, {int(PCT*100)}% rule")
    print(f"  successive differences over threshold : {100*c['frac_successive_over_threshold']:.1f}%")
    print(f"  kept, compare to last ACCEPTED        : {c['kept_last_accepted']:,} "
          f"({100*c['retention_last_accepted']:.1f}%)")
    print(f"  kept, compare to previous RAW         : {c['kept_previous_raw']:,} "
          f"({100*c['retention_previous_raw']:.1f}%)")
    print(f"  first deletion (a genuine ectopic)    : index {c['first_deleted_index']}, "
          f"{c['first_deleted_value_ms']:.0f} ms")
    print(f"  the LATCH                             : index {c['latch_index']} accepted at "
          f"{c['latch_value_ms']:.0f} ms and became the reference")
    print(f"  unbroken deletion run                 : index {c['deletion_run_starts_at']} to the "
          f"end, {c['deletion_run_length']:,} intervals")
    print(f"  median RR after the latch             : "
          f"{c['median_rr_after_latch_ms']:.0f} ms, i.e. "
          f"{100*(1-c['median_rr_after_latch_ms']/c['reference_frozen_at_ms']):.0f}% below the "
          f"frozen reference")
    print(f"\ncohort retention under the last-accepted variant (n={len(table)}): "
          f"min {100*ret.min():.1f}%, median {100*np.median(ret):.1f}%")
    print(f"records losing more than half their data: "
          f"{payload['cohort_retention_last_accepted']['n_below_half']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
