"""The specification curve: per-specification effects, joint inference, variance decomposition.

Also estimator-agnostic. It takes the long table `grid.run` produces and answers the three
questions a specification curve exists to answer (Simonsohn, Simmons & Nelson, 2020):

1. what effect does each defensible specification give?
2. does the *typical* specification's effect exceed chance? -- not "is some specification
   significant", which in a grid of thousands is guaranteed and meaningless
3. which analytic choice moves the result most? -- the only actionable output

One rule is enforced here rather than left to the caller: **there is exactly one definition of
the effect size.** `cohens_d` and `pooled_d_matrix` are the scalar and vectorised forms of the
same estimator, and the test suite pins them equal on unbalanced groups. They had drifted apart
in this project's own code, and because the primary cohort was balanced the drift stayed
invisible until an unbalanced replication reported two different numbers for one quantity.
"""
import numpy as np
import pandas as pd

MIN_GROUP_N = 3


def cohens_d(a, b):
    """Cohen's *d* with the n-1 weighted pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def pooled_d_matrix(M, mask, min_n=MIN_GROUP_N):
    """Row-wise Cohen's *d* over a (specification x unit) matrix. Must agree with `cohens_d`.

    The unweighted pooled SD, sqrt((sa**2 + sb**2) / 2), equals this only when the two groups are
    the same size. Using it here is the bug this function exists to prevent.
    """
    a = np.where(mask, M, np.nan)
    b = np.where(~mask, M, np.nan)
    na, nb = np.isfinite(a).sum(axis=1), np.isfinite(b).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
        va, vb = np.nanvar(a, axis=1, ddof=1), np.nanvar(b, axis=1, ddof=1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = (ma - mb) / pooled
    return np.where((na >= min_n) & (nb >= min_n) & (pooled > 0), d, np.nan)


def effect_table(df, metric, axes, g1, g2, group_col="group", unit_col="unit",
                 spec_col="spec_id"):
    """One row per specification: effect size, Welch p-value, group means, axis levels."""
    from scipy import stats
    rows = []
    for sid, sub in df.groupby(spec_col, sort=True):
        a = sub.loc[sub[group_col] == g1, metric].to_numpy(dtype=float)
        b = sub.loc[sub[group_col] == g2, metric].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < MIN_GROUP_N or len(b) < MIN_GROUP_N:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        r = dict(spec_id=sid, d=cohens_d(a, b), p=float(p),
                 mean_g1=float(a.mean()), mean_g2=float(b.mean()),
                 n_g1=len(a), n_g2=len(b))
        if "is_common" in sub.columns:
            r["is_common"] = bool(sub.is_common.iloc[0])
        for ax in axes:
            r[ax] = sub[ax].iloc[0]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("d").reset_index(drop=True)


def variance_decomposition(eff, axes):
    """Share of variance in the effect size attributable to each axis, ranked.

    Eta-squared from a one-way grouping on each axis, side by side. The axes are not orthogonal in
    effect, so these do not sum to 1 -- they rank what matters, which is the question a reporting
    standard needs answered.
    """
    d = eff["d"].to_numpy(dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    grand = d.mean()
    sst = ((d - grand) ** 2).sum()
    out = []
    for ax in axes:
        lv = eff.loc[ok, ax].to_numpy()
        ssb = 0.0
        for level in pd.unique(lv):
            m = lv == level
            ssb += int(m.sum()) * (d[m].mean() - grand) ** 2
        out.append(dict(axis=ax, eta_sq=ssb / sst if sst > 0 else np.nan,
                        n_levels=len(pd.unique(lv))))
    return pd.DataFrame(out).sort_values("eta_sq", ascending=False).reset_index(drop=True)


def permutation_test_median(df, metric, g1, n_perm=1000, seed=0, group_col="group",
                            unit_col="unit", spec_col="spec_id"):
    """Permute group labels at the UNIT level and compare the observed median effect to the null.

    This is step 3 of the specification-curve protocol: joint inference across the whole grid.
    The claim tested is *the typical defensible specification's effect exceeds chance*.

    Two guards, both from bugs:

    * `g1` is a parameter. Hardcoding it produces an all-False mask on any cohort not using the
      expected labels, hence a NaN observation.
    * A non-finite observation never yields a p-value. `np.abs(null) >= nan` is all-False, so a
      p-value built by counting exceedances returns `1/(n_perm+1)` -- maximal significance -- for
      a comparison that is undefined. Guard the observation, not just the null.
    """
    rng = np.random.default_rng(seed)
    subj = df[[unit_col, group_col]].drop_duplicates().reset_index(drop=True)
    groups = sorted(pd.unique(subj[group_col]))
    if len(groups) != 2:
        raise ValueError(f"permutation test needs exactly 2 groups, got {groups}")
    if g1 not in groups:
        raise ValueError(f"g1={g1!r} not among {groups}")

    piv = df.pivot_table(index=spec_col, columns=unit_col, values=metric)
    lab = dict(zip(subj[unit_col], subj[group_col]))
    y = np.array([lab[u] == g1 for u in piv.columns])
    if y.sum() == 0 or (~y).sum() == 0:
        raise ValueError(f"one side of the contrast is empty (g1={g1!r}, groups={groups})")
    M = piv.to_numpy(dtype=float)

    def med(mask):
        return np.nanmedian(pooled_d_matrix(M, mask))

    obs = med(y)
    if not np.isfinite(obs):
        return dict(observed_median_d=float("nan"), p_perm=None, n_perm=n_perm,
                    error="observed median effect is NaN; p-value withheld",
                    positive_group=g1, groups=groups)

    null = np.array([med(rng.permutation(y)) for _ in range(n_perm)])
    finite = np.isfinite(null)
    if finite.sum() < n_perm * 0.9:
        return dict(observed_median_d=float(obs), p_perm=None, n_perm=n_perm,
                    error=f"only {int(finite.sum())}/{n_perm} permutations finite; withheld",
                    positive_group=g1, groups=groups)
    p = (np.sum(np.abs(null[finite]) >= abs(obs)) + 1) / (finite.sum() + 1)
    return dict(observed_median_d=float(obs), p_perm=float(p),
                null_mean=float(np.nanmean(null)), null_sd=float(np.nanstd(null)),
                n_perm=int(finite.sum()), positive_group=g1, groups=groups)


def summarise(eff):
    """The numbers a specification curve is reported by."""
    d = eff["d"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    sig, pos = eff.p < 0.05, eff.d > 0
    common = eff[eff.is_common] if "is_common" in eff.columns else eff.iloc[:0]
    return dict(
        n_specs=len(eff),
        d_median=float(np.median(d)), d_min=float(d.min()), d_max=float(d.max()),
        d_iqr=[float(np.percentile(d, 25)), float(np.percentile(d, 75))],
        frac_p05=float(sig.mean()), frac_positive=float(pos.mean()),
        frac_sig_positive=float((sig & pos).mean()),
        frac_sig_negative=float((sig & ~pos).mean()),
        sign_flips=bool(pos.any() and (~pos).any()),
        common_n=len(common),
        common_d_median=float(common.d.median()) if len(common) else np.nan,
        common_d_range=[float(common.d.min()), float(common.d.max())] if len(common) else None,
    )
