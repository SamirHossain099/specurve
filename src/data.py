"""Fetch and cache RR-interval series from PhysioNet.

Cohorts
-------
fantasia : 20 young (f1y*/f2y*) vs 20 old (f1o*/f2o*) healthy subjects. The canonical
           dataset for the aging-reduces-long-range-correlation result, and therefore the
           right place to ask how much of that result is the analysis pipeline.
nsrdb    : 18 normal sinus rhythm subjects (healthy control cohort).
chf2db   : 29 congestive heart failure subjects (pathological contrast).
"""
import os

import numpy as np
import wfdb

CACHE = "data/rr"

COHORTS = {
    "fantasia": dict(pn_dir="fantasia/1.0.0", ann="ecg"),
    "nsrdb": dict(pn_dir="nsrdb/1.0.0", ann="atr"),
    "chf2db": dict(pn_dir="chf2db/1.0.0", ann="ecg"),
}

# Beat symbols treated as normal sinus beats.
NORMAL = {"N"}


def record_list(cohort):
    import requests
    d = COHORTS[cohort]
    r = requests.get(f"https://physionet.org/files/{d['pn_dir']}/RECORDS", timeout=90)
    r.raise_for_status()
    return [x.strip() for x in r.text.split() if x.strip()]


def fantasia_group(rec):
    """f1y01 -> young, f1o01 -> old."""
    return "young" if "y" in rec[:3] else "old"


def rr_from_annotations(rec, cohort):
    d = COHORTS[cohort]
    ann = wfdb.rdann(rec, d["ann"], pn_dir=d["pn_dir"])
    sym = np.asarray(ann.symbol)
    samp = np.asarray(ann.sample)
    keep = np.isin(sym, list(NORMAL))
    beats = samp[keep]
    if len(beats) < 100:
        return None
    rr = np.diff(beats) / ann.fs * 1000.0          # ms
    # Drop physiologically impossible intervals before any analysis choice is applied.
    rr = rr[(rr > 300) & (rr < 2000)]
    return rr if len(rr) >= 500 else None


def cache_cohort(cohort, limit=None):
    os.makedirs(CACHE, exist_ok=True)
    recs = record_list(cohort)
    if limit:
        recs = recs[:limit]
    out = {}
    for i, rec in enumerate(recs, 1):
        path = f"{CACHE}/{cohort}__{rec}.npy"
        if os.path.exists(path):
            out[rec] = np.load(path)
            continue
        try:
            rr = rr_from_annotations(rec, cohort)
        except Exception as e:
            print(f"  [{i}/{len(recs)}] {rec}: FAILED {type(e).__name__}: {e}")
            continue
        if rr is None:
            print(f"  [{i}/{len(recs)}] {rec}: too short / unusable")
            continue
        np.save(path, rr)
        out[rec] = rr
        print(f"  [{i}/{len(recs)}] {rec}: n={len(rr):,} mean={rr.mean():.0f}ms", flush=True)
    return out


def load_cached(cohort):
    out = {}
    if not os.path.isdir(CACHE):
        return out
    for fn in sorted(os.listdir(CACHE)):
        if fn.startswith(f"{cohort}__") and fn.endswith(".npy"):
            rec = fn[len(cohort) + 2:-4]
            out[rec] = np.load(f"{CACHE}/{fn}")
    return out


if __name__ == "__main__":
    import sys
    for c in (sys.argv[1:] or ["fantasia"]):
        print(f"=== {c}")
        d = cache_cohort(c)
        print(f"  cached {len(d)} records, "
              f"median length {np.median([len(v) for v in d.values()]):.0f}")
        if c == "fantasia":
            g = {}
            for r in d:
                g.setdefault(fantasia_group(r), []).append(r)
            print("  groups:", {k: len(v) for k, v in g.items()})
