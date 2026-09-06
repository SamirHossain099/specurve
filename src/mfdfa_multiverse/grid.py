"""The specification-grid runner -- the part of this project that is not about DFA.

A multiverse analysis is the same shape whatever the estimator: enumerate a cartesian product of
analytic choices, evaluate every one against every unit, and keep the result in a long table that
downstream analysis can group by choice. Nothing in this module knows what DFA is. It takes a
mapping of axis name to levels and a callable that turns one specification plus one series into a
dict of estimates.

This is the component the project brief asks to be liftable, and the reason it is worth lifting is
the three properties it enforces, all of which were bugs before they were rules:

* **Nothing defaults silently.** A specification is a dict of every axis; there is no "the rest of
  the parameters were whatever the library does."
* **Failures are recorded, not dropped.** An estimator that cannot evaluate a specification
  returns a row with `ok=False`, so the denominator stays honest. In this project that turned out
  to matter: a specification that silently vanished would have removed a subject from a quarter of
  the grid without appearing anywhere in the results.
* **The grid is frozen before the outcome is computed.** `freeze_hash` gives a stable fingerprint
  of the axes so a pre-registration can name one and a later run can prove it used the same grid.

Example
-------
    from mfdfa_multiverse import grid

    AXES = {"window": [16, 32, 64], "detrend": ["none", "linear"]}

    def estimate(series, spec):
        ...
        return {"stat": value, "ok": True}

    df = grid.run(AXES, {"subject_01": x1, "subject_02": x2}, estimate)
"""
import hashlib
import itertools
import json
import time

import pandas as pd


def build(axes):
    """Every specification in the cartesian product of `axes`, as a list of dicts.

    `axes` maps axis name to an iterable of levels. Insertion order is preserved, so `spec_id`
    (the index into this list) is stable as long as the axes are.
    """
    names = list(axes)
    return [dict(zip(names, combo)) for combo in itertools.product(*[list(axes[k]) for k in names])]


def freeze_hash(axes):
    """A stable fingerprint of the grid, for a pre-registration to name.

    Levels are stringified before hashing so that a float level and its repr do not produce
    different hashes across platforms. Changing a level, adding an axis or reordering the axes all
    change the hash; re-running the same grid does not.
    """
    payload = json.dumps({k: [str(v) for v in axes[k]] for k in axes}, sort_keys=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def summary(axes, is_common=None):
    """Shape of the grid: total size, per-axis level counts, and the fingerprint."""
    g = build(axes)
    out = dict(total=len(g), axis_sizes={k: len(list(axes[k])) for k in axes},
               freeze_hash=freeze_hash(axes))
    if is_common is not None:
        out["common"] = sum(bool(is_common(s)) for s in g)
    return out


def _one_spec(spec_id, spec, names, series, meta, estimate):
    rows = []
    for name in names:
        est = estimate(series[name], spec)
        rows.append(dict(spec_id=spec_id, unit=name, **meta.get(name, {}), **spec, **est))
    return rows


def run(axes, series, estimate, meta=None, is_common=None, n_jobs=None, verbose=5,
        batch_size=16):
    """Evaluate every specification against every unit.

    Parameters
    ----------
    axes : mapping of axis name -> levels. The cartesian product is the grid.
    series : mapping of unit name -> whatever `estimate` accepts.
    estimate : callable(series_for_one_unit, spec) -> dict of estimates. It should catch its own
        errors and return `ok=False` rather than raising; a raise here loses the whole batch.
    meta : optional mapping of unit name -> dict of extra columns (group labels, covariates).
    is_common : optional callable(spec) -> bool, marking a conventional-defaults subgrid. The
        result carries it as `is_common`.
    n_jobs : passed to joblib. None runs in-process, which is the right choice for a fast
        estimator where process startup would dominate.

    Returns a long DataFrame with one row per (specification, unit).
    """
    names = sorted(series)
    meta = meta or {}
    g = build(axes)
    t0 = time.time()

    if n_jobs in (None, 0, 1):
        batches = [_one_spec(i, sp, names, series, meta, estimate) for i, sp in enumerate(g)]
    else:
        from joblib import Parallel, delayed
        # max_nbytes=None keeps joblib from spilling memmaps into TEMP, which fills the disk on
        # long sweeps.
        batches = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=batch_size,
                           max_nbytes=None)(
            delayed(_one_spec)(i, sp, names, series, meta, estimate) for i, sp in enumerate(g))

    df = pd.DataFrame([r for b in batches for r in b])
    if is_common is not None:
        df["is_common"] = [bool(is_common(sp)) for sp in g for _ in names]
    df.attrs["freeze_hash"] = freeze_hash(axes)
    df.attrs["elapsed_s"] = time.time() - t0
    return df


def failure_report(df, ok_col="ok"):
    """How many fits failed, and under which levels. A zero here is a claim worth making."""
    if ok_col not in df.columns:
        return dict(n=len(df), failed=0, frac=0.0, by_axis={})
    bad = df[~df[ok_col].astype(bool)]
    by_axis = {}
    if len(bad):
        for c in df.columns:
            if c in (ok_col, "spec_id", "unit") or df[c].dtype.kind in "fc":
                continue
            counts = bad[c].value_counts()
            if len(counts) < len(df[c].unique()):     # a level that never fails is informative
                by_axis[c] = counts.to_dict()
    return dict(n=len(df), failed=int(len(bad)),
                frac=float(len(bad) / len(df)) if len(df) else 0.0, by_axis=by_axis)
