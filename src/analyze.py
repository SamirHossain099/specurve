"""Specification-curve analysis and variance decomposition.

For each specification we compute the group-level effect (young vs old) on DFA alpha, then:
  1. sort specifications by effect size -> the specification curve
  2. decompose variance in the effect across analytic choices -> what to standardise
  3. permutation-test the MEDIAN effect across specifications, not a cherry-picked one
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

MIN_GROUP_N = 3  # shared by effect_table and pooled_d_matrix; see pooled_d_matrix.

AXES = ["ectopic", "detrend", "normalise", "order", "s_min",
        "s_max_frac", "n_scales", "spacing", "overlap", "q_grid"]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def pooled_d_matrix(M, mask, min_n=MIN_GROUP_N):
    """Row-wise Cohen's *d* over a (spec x record) matrix, vectorised.

    This is the matrix form of `cohens_d` and MUST agree with it: same n-1 weighted pooled
    standard deviation, same minimum group size. It used to use an unweighted pooled SD,
    sqrt((sa**2 + sb**2) / 2), which is algebraically identical only when the two groups are
    the same size. Fantasia is 20 vs 20 so the two agreed there and the discrepancy stayed
    invisible; the replication contrast is 29 vs 18, where the permutation test reported an
    observed median effect of 0.714 against the specification curve's 0.687 -- two numbers for
    one quantity, both of which reached the results file. `tests/test_permutation.py` now pins
    the identity on every cohort.
    """
    a = np.where(mask, M, np.nan)
    b = np.where(~mask, M, np.nan)
    na = np.isfinite(a).sum(axis=1)
    nb = np.isfinite(b).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
        va, vb = np.nanvar(a, axis=1, ddof=1), np.nanvar(b, axis=1, ddof=1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = (ma - mb) / pooled
    d = np.where((na >= min_n) & (nb >= min_n) & (pooled > 0), d, np.nan)
    return d


def effect_table(df, metric="alpha", g1="young", g2="old"):
    """One row per specification: effect size, p-value, group means."""
    from scipy import stats
    rows = []
    for sid, sub in df.groupby("spec_id", sort=True):
        a = sub.loc[sub.group == g1, metric].to_numpy(dtype=float)
        b = sub.loc[sub.group == g2, metric].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < MIN_GROUP_N or len(b) < MIN_GROUP_N:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        r = dict(spec_id=sid, d=cohens_d(a, b), p=float(p),
                 mean_young=float(a.mean()), mean_old=float(b.mean()),
                 n_young=len(a), n_old=len(b),
                 is_common=bool(sub.is_common.iloc[0]))
        for ax in AXES:
            r[ax] = sub[ax].iloc[0]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("d").reset_index(drop=True)


def variance_decomposition(eff):
    """Share of variance in the effect size attributable to each analytic axis.

    Eta-squared from a one-way grouping on each axis, reported side by side. Axes are not
    orthogonal in effect, so these do not sum to 1 -- they rank what matters.
    """
    out = []
    d = eff["d"].to_numpy(dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    grand = d.mean()
    sst = ((d - grand) ** 2).sum()
    for ax in AXES:
        lv = eff.loc[ok, ax].to_numpy()
        ssb = 0.0
        for level in pd.unique(lv):
            m = lv == level
            ssb += m.sum() * (d[m].mean() - grand) ** 2
        out.append(dict(axis=ax, eta_sq=ssb / sst if sst > 0 else np.nan,
                        n_levels=len(pd.unique(lv))))
    return pd.DataFrame(out).sort_values("eta_sq", ascending=False).reset_index(drop=True)


def permutation_test_median(df, metric="alpha", n_perm=1000, seed=0, g1=None):
    """Permute the group labels at the SUBJECT level, recompute the median effect across
    all specifications, and compare to the observed median.

    This is the inferential statement the specification curve needs: not "some
    specification is significant" but "the typical specification's effect exceeds chance".

    `g1` names the positive group. It used to be hardcoded to "young", which silently
    produced an all-False mask (and therefore a NaN effect, and a spurious p = 1/(n_perm+1))
    on any cohort not labelled young/old. Defaults to the first group alphabetically.
    """
    rng = np.random.default_rng(seed)
    subj = df[["record", "group"]].drop_duplicates().reset_index(drop=True)

    groups = sorted(pd.unique(subj.group))
    if len(groups) != 2:
        raise ValueError(f"permutation test needs exactly 2 groups, got {groups}")
    if g1 is None:
        g1 = "young" if "young" in groups else groups[0]
    if g1 not in groups:
        raise ValueError(f"g1={g1!r} not among {groups}")

    piv = df.pivot_table(index="spec_id", columns="record", values=metric)
    recs = list(piv.columns)
    lab_for_rec = dict(zip(subj.record, subj.group))
    y = np.array([lab_for_rec[r] == g1 for r in recs])
    if y.sum() == 0 or (~y).sum() == 0:
        raise ValueError(f"one side of the contrast is empty (g1={g1!r}, groups={groups})")
    M = piv.to_numpy(dtype=float)

    def med_effect(mask):
        return np.nanmedian(pooled_d_matrix(M, mask))

    obs = med_effect(y)
    if not np.isfinite(obs):
        # Never return a p-value for a NaN observation: `np.abs(null) >= nan` is all-False,
        # which yields p = 1/(n_perm+1) and looks like a strong result. It is not.
        return dict(observed_median_d=float("nan"), p_perm=None, n_perm=n_perm,
                    error="observed median effect is NaN; p-value withheld",
                    positive_group=g1, groups=groups)

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = med_effect(rng.permutation(y))
    finite = np.isfinite(null)
    if finite.sum() < n_perm * 0.9:
        return dict(observed_median_d=float(obs), p_perm=None, n_perm=n_perm,
                    error=f"only {int(finite.sum())}/{n_perm} permutations finite; "
                          "p-value withheld", positive_group=g1, groups=groups)

    p = (np.sum(np.abs(null[finite]) >= abs(obs)) + 1) / (finite.sum() + 1)
    return dict(observed_median_d=float(obs), p_perm=float(p),
                null_mean=float(np.nanmean(null)), null_sd=float(np.nanstd(null)),
                n_perm=int(finite.sum()), positive_group=g1, groups=groups)


def summarise(eff):
    d = eff["d"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    sig = eff.p < 0.05
    pos = eff.d > 0
    common = eff[eff.is_common]
    return dict(
        n_specs=len(eff),
        d_median=float(np.median(d)), d_min=float(d.min()), d_max=float(d.max()),
        d_iqr=[float(np.percentile(d, 25)), float(np.percentile(d, 75))],
        frac_p05=float(sig.mean()),
        frac_positive=float(pos.mean()),
        frac_sig_positive=float((sig & pos).mean()),
        frac_sig_negative=float((sig & ~pos).mean()),
        sign_flips=bool(pos.any() and (~pos).any()),
        common_n=len(common),
        common_d_median=float(common.d.median()) if len(common) else np.nan,
        common_d_range=[float(common.d.min()), float(common.d.max())] if len(common) else None,
    )


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid_fantasia.parquet")
    ap.add_argument("--metric", default="alpha")
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()

    df = pd.read_parquet(a.grid)
    print(f"loaded {len(df):,} rows, {df.spec_id.nunique():,} specifications")

    eff = effect_table(df, metric=a.metric)
    eff.to_csv(f"results/effects_{a.metric}.csv", index=False)
    s = summarise(eff)
    print("\n=== SPECIFICATION CURVE SUMMARY ===")
    print(json.dumps(s, indent=2))

    vd = variance_decomposition(eff)
    vd.to_csv(f"results/variance_{a.metric}.csv", index=False)
    print("\n=== VARIANCE DECOMPOSITION (eta^2 on effect size) ===")
    print(vd.to_string(index=False))

    pt = permutation_test_median(df, metric=a.metric, n_perm=a.n_perm)
    print("\n=== PERMUTATION TEST ON THE MEDIAN EFFECT ===")
    print(json.dumps(pt, indent=2))

    with open(f"results/summary_{a.metric}.json", "w") as fh:
        json.dump(dict(summary=s, permutation=pt,
                       variance=vd.to_dict("records")), fh, indent=2)
