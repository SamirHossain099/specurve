# Pre-Registration: MFDFA Specification Curve

**Fixed:** 2026-08-29, before any group-level effect was computed.
**Provenance:** the grid is defined in `src/specs.py`. That file was written and committed
*before* `src/analyze.py` existed and before any young-vs-old comparison was run. The raw
per-record estimates (`src/run_grid.py`) were produced from the frozen grid.

> The killer objection to this paper is *"you manufactured instability by including
> specifications no competent analyst would use."* This document is the defence: every axis
> level below is justified from published practice, the grid was fixed in advance, and the
> analysis additionally reports a restricted `commonly used` subgrid.

---

## 1. Research question

Across the space of defensible analytic choices in a DFA/MFDFA pipeline, how much does the
reported group-level effect vary -- and is the effect's *sign* or *significance* a function of
choices that papers do not report?

**Primary endpoint:** DFA α, i.e. the generalized Hurst exponent h(q) evaluated at q = 2, on
RR-interval series.
**Primary contrast:** Fantasia young (n = 20) vs old (n = 20).
**Secondary endpoint:** multifractal width Δα from the Legendre transform.

**Direction of the established claim:** healthy aging is reported to *reduce* long-range
correlation in heart-rate dynamics, so α should be lower in the old group (positive Cohen's
*d* for young − old).

## 2. The grid: 9,216 specifications

| Axis | Levels | Justification for each level |
|---|---|---|
| `ectopic` | 4 | `none` (pre-screened recordings) · `pct20_drop` (Malik 20% rule, deletion -- the most common HRV convention) · `pct20_interp` (same detection, linear interpolation -- preserves length) · `mad4_drop` (robust 4-MAD rule on successive differences) |
| `detrend` | 3 | `none` (DFA already removes within-window polynomial trends) · `linear` (global drift) · `quadratic` |
| `normalise` | 2 | `none` (raw ms) · `zscore` (standard when comparing subjects with different mean HR) |
| `order` | 4 | DFA1 (original Peng formulation) · DFA2 (most common default) · DFA3 · DFA4 (strongly nonstationary records) |
| `s_min` | 2 | 16 (common lower bound; below ~10 the polynomial fit is unstable) · 32 (conservative, used when order ≥ 3) |
| `s_max_frac` | 3 | N/20 (conservative) · N/10 (most common) · N/4 (permissive, short-record studies) |
| `n_scales` | 2 | 20 (typical) · 30 (finer sampling) |
| `spacing` | 2 | `log` (near-universal) · `linear` (occasionally used) |
| `overlap` | 2 | non-overlapping (original formulation) · 50% overlap (improves statistics on short records) |
| `q_grid` | 2 | coarse (step 1.0) · fine (step 0.5) |

**Total: 4 × 3 × 2 × 4 × 2 × 3 × 2 × 2 × 2 × 2 = 9,216.**

### The `commonly used` subgrid: 32 specifications

A conventional analyst choosing defaults without deliberation would land in:
`ectopic ∈ {none, pct20_drop}`, `detrend ∈ {none, linear}`, `normalise ∈ {none, zscore}`,
`order ∈ {1, 2}`, `s_min = 16`, `s_max_frac = 0.10`, `n_scales = 20`, `spacing = log`,
`overlap = False`. The curve is reported for this subgrid separately. **If instability appears
only outside this subgrid, we say so plainly** -- that is a weaker but still publishable result.

## 3. Data

Cached from PhysioNet, open access, no credentials.

| Cohort | n | Role |
|---|---|---|
| `fantasia` | 40 (20 young / 20 old) | primary contrast |
| `nsrdb` | 18 | healthy control cohort (secondary) |
| `chf2db` | 29 | pathological contrast (secondary) |

**Pre-analysis exclusions**, applied identically to every specification and *before* any
analytic choice: beats restricted to normal sinus (`N`); RR intervals outside 300–2000 ms
removed as physiologically impossible; records with <500 usable intervals dropped. A
specification producing <200 usable samples or <4 valid scales returns NaN and is recorded as
a failed fit rather than silently dropped.

## 4. Analysis plan

1. **Specification curve.** Effect size (Cohen's *d*, Welch) per specification, sorted; lower
   panel shows which analytic choices are active in each region.
2. **Inference on the curve.** Permutation test on the **median** effect across all
   specifications, permuting group labels at the **subject** level (1,000 permutations). The
   claim tested is "the typical specification's effect exceeds chance," not "some
   specification is significant."
3. **Variance decomposition.** η² of effect size against each axis, ranked. The deliverable is
   *which choice matters most* -- that is what the field should standardise.
4. **Reported regardless of outcome:** fraction of specifications with p < 0.05; fraction with
   each sign; whether any sign flip occurs; the `commonly used` subgrid's range.

## 5. Outcomes and how each is interpreted

| Outcome | Interpretation | Publishable? |
|---|---|---|
| Wide curve, sign flips | A substantial share of published DFA findings are specification-dependent | Yes -- the strong result |
| Wide curve, no sign flips | Magnitude is unstable, direction is robust; effect sizes are not comparable across papers | Yes |
| Narrow curve | **Certification of robustness** -- the field's estimates are stable, and this becomes the citable justification for parameter choices | Yes |
| Instability only outside `commonly used` | Conventional defaults are safe; the literature's variability comes from unconventional choices | Yes, weaker |

**No outcome is a failure.** This is deliberate: the design cannot produce an unpublishable result,
which is why it is the low-risk parallel track.

## 6. Deviations

Any change after this date must be recorded here with a date and reason.

### D1: 2026-08-29: replication cohorts analysed as a disease contrast, not an aging one

`nsrdb` and `chf2db` were named in §3 as secondary cohorts without specifying the contrast. They
have no internal age split, so the contrast run is **chf2db vs nsrdb** (heart failure vs healthy).
It tests whether the *instability* replicates on a different contrast, not whether the aging
effect does. Stated as such wherever it is reported.

### D2: 2026-09-03: record length added as a crossed factor (post hoc)

**Not pre-registered. Added after seeing the primary results, and reported as a control rather
than as a planned analysis.**

Reason: the primary contrast (Fantasia, median 6,977 RR intervals) and the replication contrast
(nsrdb/chf2db, median ~100,000) differ in record length by roughly 14x. The headline of §2b --
that the dominant analytic choice *changes with the population* -- is therefore confounded with
record length on arrival, and no analysis in the pre-registered plan can separate the two.

What was added (`src/length_control.py`):

- **A length-matched rung.** Every record in every cohort truncated to its first 4,731 intervals
  -- the shortest Fantasia record, so no record is ragged and every cohort is matched exactly.
  The population claim is re-tested there.
- **A dose-response ladder** over truncations {1000, 2000, 4000, 4731, 8000, 16000, 32000} plus
  the existing full-length runs, tracking the effect size, the width of the curve and the eta^2
  ranking as length varies.

**The specification grid itself is unchanged.** No axis was added, removed or re-levelled; the
9,216 specifications are the frozen ones and `src/specs.py` is untouched. Length is crossed with
them as an external factor, and every truncated run is a separate parquet named
`grid_<cohort>_trunc<N>.parquet`.

This deviation makes a result *harder* to claim, not easier: it exists to test whether §2b's
headline survives a confound, and it is reported whichever way it comes out.

### D3: 2026-09-03: one estimator for the median effect, not two

`permutation_test_median` computed Cohen's *d* with an unweighted pooled SD,
`sqrt((sa^2 + sb^2)/2)`, while `effect_table` used the n-1 weighted pooled SD. The two are
algebraically identical when the groups are the same size, so Fantasia (20 vs 20) never showed a
discrepancy; the replication contrast (29 vs 18) reported an observed median effect of 0.714
against the specification curve's 0.687 for the same quantity. Both numbers reached
`results/`. The matrix estimator now shares its definition with the scalar one and a test pins
the identity on every shipped result file.

Effect on reported numbers: Fantasia is unchanged (alpha p = 0.170, delta-alpha p = 0.002). The
disease contrast's observed medians become 0.687 (alpha, p = 0.004, unchanged) and 0.487
(delta-alpha, **p = 0.004, was 0.001**).
