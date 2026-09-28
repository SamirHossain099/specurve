"""Every number the manuscript quotes, pinned to the file that produced it.

The rule this file follows: **read `results/`, never hardcode the value**. A test that asserts
`d_median == 0.2966` passes until the day the grid changes and then reports a stale number as
correct. These tests assert relationships and tolerances against the shipped result files, and
assert that the manuscript's text agrees with them.

Habit 7 in CORRECTIONS.md.
"""
import json
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
DRAFT = os.path.join(ROOT, "PAPER-DRAFT.md")

CELLS = {
    "fantasia_alpha": "summary_alpha.json",
    "fantasia_delta_alpha": "summary_delta_alpha.json",
    "disease_alpha": "summary_repl_chf2db_vs_nsrdb_alpha.json",
    "disease_delta_alpha": "summary_repl_chf2db_vs_nsrdb_delta_alpha.json",
}


def load(name):
    path = os.path.join(RESULTS, CELLS[name])
    if not os.path.exists(path):
        pytest.skip(f"{CELLS[name]} not present")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def draft_text():
    if not os.path.exists(DRAFT):
        pytest.skip("PAPER-DRAFT.md not present")
    with open(DRAFT, encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.parametrize("name", list(CELLS))
def test_grid_size_is_the_preregistered_one(name):
    """9,216 specifications, every cell, or the grid is not the frozen one."""
    assert load(name)["summary"]["n_specs"] == 9216


@pytest.mark.parametrize("name", list(CELLS))
def test_common_subgrid_is_32(name):
    assert load(name)["summary"]["common_n"] == 32


@pytest.mark.parametrize("name", list(CELLS))
def test_curve_median_lies_inside_its_own_range(name):
    s = load(name)["summary"]
    assert s["d_min"] <= s["d_median"] <= s["d_max"]
    assert s["d_iqr"][0] <= s["d_median"] <= s["d_iqr"][1]


@pytest.mark.parametrize("name", list(CELLS))
def test_significant_fractions_are_consistent(name):
    """sig-positive plus sig-negative must equal the total significant fraction."""
    s = load(name)["summary"]
    assert s["frac_sig_positive"] + s["frac_sig_negative"] == pytest.approx(s["frac_p05"], abs=1e-12)


@pytest.mark.parametrize("name", list(CELLS))
def test_sign_flips_everywhere(name):
    """The paper says the sign flips in all four cells. If one stops flipping, say so."""
    s = load(name)["summary"]
    assert s["sign_flips"] is True
    assert s["d_min"] < 0 < s["d_max"]


def test_fantasia_common_subgrid_is_uniformly_positive_on_alpha():
    """Section 3.1's reassuring half: the conventional defaults recover the canonical effect."""
    s = load("fantasia_alpha")["summary"]
    assert s["common_d_range"][0] > 0


def test_disease_common_subgrid_flips_sign_on_alpha():
    """Section 3.5: the reassurance is cohort-specific -- the SAME 32 specifications flip."""
    lo, hi = load("disease_alpha")["summary"]["common_d_range"]
    assert lo < 0 < hi


def test_endpoints_disagree_on_fantasia():
    """Section 3.6, the paper's second-strongest claim: alpha is not significant at the median
    specification and delta-alpha is, on the same fits."""
    a = load("fantasia_alpha")["permutation"]
    d = load("fantasia_delta_alpha")["permutation"]
    assert a["p_perm"] > 0.05
    assert d["p_perm"] < 0.05


def test_two_axes_that_theory_says_are_inert_are_inert():
    """Section 3.4. alpha is invariant to affine rescaling, and h(q=2) cannot depend on which
    other q values were evaluated. Both must come back at ~0 or the implementation is wrong."""
    for name in ("fantasia_alpha", "disease_alpha"):
        eta = {v["axis"]: v["eta_sq"] for v in load(name)["variance"]}
        assert eta["normalise"] < 1e-8, f"{name}: normalise is not inert"
        assert eta["q_grid"] < 1e-8, f"{name}: q_grid is not inert"


def test_ectopic_dominates_three_of_four_cells():
    """Section 3.3, the corrected version of the population claim (CORRECTIONS.md C5)."""
    tops = {}
    for name in CELLS:
        v = load(name)["variance"]
        tops[name] = max(v, key=lambda r: r["eta_sq"])["axis"]
    assert sum(t == "ectopic" for t in tops.values()) == 3, tops
    assert tops["fantasia_alpha"] == "s_max_frac", tops


def test_draft_quotes_the_grid_size_and_subgrid_size():
    t = draft_text()
    assert "9,216" in t
    assert "32 specifications" in t or "32-specification" in t


@pytest.mark.parametrize("name,pattern", [
    ("fantasia_alpha", r"p = 0\.170"),
    ("fantasia_delta_alpha", r"p = 0\.002"),
])
def test_draft_permutation_p_values_match_results(name, pattern):
    """The p-values the prose quotes are the ones in results/, to three decimals."""
    p = load(name)["permutation"]["p_perm"]
    assert re.search(pattern, draft_text()), f"{name}: prose does not quote its p-value"
    assert float(re.search(pattern, draft_text()).group().split("=")[1]) == pytest.approx(p, abs=5e-4)


def test_the_width_split_is_described_in_the_direction_the_sign_means():
    """CORRECTIONS.md C9. d is young minus elderly, so a significant NEGATIVE effect on the width
    is a significant INCREASE with age. The submitted abstract had the two words swapped while the
    percentages were right, and no test looked at the words."""
    s = load("fantasia_delta_alpha")["summary"]
    inc, dec = s["frac_sig_negative"], s["frac_sig_positive"]
    assert inc > dec
    t = draft_text()
    ab = t[t.index("## Abstract"):t.index("**Keywords:**")]
    want = re.sub(r"\s+", r"\\s+",
                  rf"{100 * inc:.1f}% of specifications showed a significant increase with age "
                  rf"and {100 * dec:.1f}% a significant decrease")
    assert re.search(want, ab), "abstract no longer states the width split in the right direction"


def test_draft_has_no_unresolved_ladder_placeholders_once_ladder_is_complete():
    """While the length ladder is running the draft carries [LADDER] markers. Once the ladder
    file reports every planned cell, the draft must not still be quoting placeholders."""
    path = os.path.join(RESULTS, "length_matched_verdict.json")
    if not os.path.exists(path):
        pytest.skip("length ladder still running")
    assert "[LADDER]" not in draft_text(), "ladder is complete but the draft still has placeholders"


# ---------------------------------------------------------------------------------------------
# The ectopic case study (section 3.3 of the draft). Reads results/ectopic_case.json.
# ---------------------------------------------------------------------------------------------

def ectopic_case():
    path = os.path.join(RESULTS, "ectopic_case.json")
    if not os.path.exists(path):
        pytest.skip("ectopic_case.json not present; run src/ectopic_case.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_two_implementations_of_one_rule_disagree_by_an_order_of_magnitude():
    """Same rule, same threshold, same record. The claim is the ratio, not the literals."""
    c = ectopic_case()["case"]
    assert c["kept_previous_raw"] > 10 * c["kept_last_accepted"]
    assert c["retention_previous_raw"] > 0.95
    assert c["retention_last_accepted"] < 0.15


def test_the_deletion_is_a_latch_not_a_threshold():
    """0.5% of successive differences exceed 20%, yet most of the record is deleted, and the
    deletion is one unbroken run to the end. That combination is what makes it a latch."""
    c = ectopic_case()["case"]
    assert c["frac_successive_over_threshold"] < 0.02
    assert c["deletion_runs_to_end"] is True
    assert c["deletion_run_length"] > 0.85 * c["n"]
    assert c["latch_index"] < c["deletion_run_starts_at"]


def test_the_latched_reference_is_far_above_the_records_own_beats():
    """The reference the filter froze on is well outside the range of normal beats that follow,
    which is why nothing is ever accepted again."""
    c = ectopic_case()["case"]
    assert c["median_rr_after_latch_ms"] < 0.8 * c["reference_frozen_at_ms"]


def test_the_case_is_an_outlier_not_the_norm():
    """One record in forty. The paper must not imply the rule usually behaves this way."""
    r = ectopic_case()["cohort_retention_last_accepted"]
    assert r["n_below_half"] == 1
    assert r["median"] > 0.95


# ---------------------------------------------------------------------------------------------
# House style. These guard formatting decisions that are easy to undo by accident.
# ---------------------------------------------------------------------------------------------

EM_DASH = "\u2014"


@pytest.mark.parametrize("name", ["PAPER-DRAFT.md", "MANUSCRIPT.md"])
def test_no_em_dashes(name):
    """Em dashes were removed from the paper by rewriting each sentence, not by substituting a
    character. A reintroduced one means a paragraph was edited without that rule in mind."""
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    text = open(path, encoding="utf-8").read()
    hits = [ln for ln in text.split("\n") if EM_DASH in ln]
    assert not hits, f"{len(hits)} line(s) with an em dash, first: {hits[0][:80]}"


def test_abstract_is_within_the_journals_limit():
    """European Journal of Applied Physiology: a structured abstract of 150 to 250 words."""
    text = draft_text()
    ab = text[text.index("## Abstract"):text.index("**Keywords:**")]
    words = len(re.findall(r"\S+", re.sub(r"[*`]", "", ab.replace("## Abstract", ""))))
    assert 150 <= words <= 250, f"abstract is {words} words"


def test_the_abstract_carries_no_bold_run_in_labels():
    """House rule, and the reference layout in RESEARCH.md section 11: the abstract runs as prose.
    It carried Purpose/Methods/Results/Conclusion heads while a venue that wanted them was the
    target."""
    text = draft_text()
    ab = text[text.index("## Abstract"):text.index("**Keywords:**")]
    assert "**" not in ab, "bold label in the abstract"


def test_keywords_and_abbreviations_follow_the_journal():
    """4 to 6 keywords, then an alphabetical abbreviation list."""
    text = draft_text()
    kw = text[text.index("**Keywords:**"):].split("\n\n")[0].replace("**Keywords:**", "")
    assert 4 <= len([k for k in kw.split(";") if k.strip()]) <= 6
    abbr = text[text.index("## Abbreviations"):].split("\n\n")[1]
    terms = [ln[2:].split(":")[0] for ln in abbr.splitlines() if ln.startswith("- ")]
    assert terms and terms == sorted(terms, key=str.lower), terms


def test_abstract_cites_nothing_and_names_no_float():
    """No references, table numbers, figure numbers or equations in the abstract."""
    text = draft_text()
    ab = text[text.index("## Abstract"):text.index("**Keywords:**")]
    assert not re.search(r"\[-?@[a-z0-9_]+", ab), "abstract carries a citation"
    assert not re.search(r"\b(Table|Figure|Eq\.?|Equation)\s+\d", ab), "abstract names a float"


def test_every_section_cross_reference_resolves():
    """Sections were renumbered when 3.4 was inserted; a dangling reference is silent otherwise."""
    text = draft_text()
    body = text[text.index("## Abstract"):text.index("## Drafting notes")]
    heads = {m.group(1) for m in re.finditer(r"^#{2,3} (\d+(?:\.\d+)?)[.\s]", body, re.M)}
    used = set(re.findall(r"\u00a7(\d+(?:\.\d+)?)", body))
    assert not (used - heads), f"dangling: {sorted(used - heads)}"


def test_the_replication_table_has_a_first_column_heading():
    """A headerless first column was how the table read before; the header names what the rows
    are. Matches the column rather than the whole line, which now also carries the
    matched-length column."""
    t = draft_text()
    assert re.search(r"\|\s*Quantity\s*\|.*Fantasia \(aging\), α", t)


def interactions():
    path = os.path.join(RESULTS, "interactions_summary.json")
    if not os.path.exists(path):
        pytest.skip("interactions_summary.json not present; run src/interactions.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_interactions_are_a_material_share_of_variance():
    """Section 3.8's first claim: on the primary cell the two-way terms are a third of the
    explained variance, so a one-way ranking is not describing the design."""
    j = interactions()["fantasia_alpha"]
    assert j["sum_two_way_eta_sq"] > 0.3
    assert j["sum_two_way_eta_sq"] > 0.5 * j["sum_main_eta_sq"]


def test_order_carries_the_largest_interaction_in_most_cells():
    """Section 3.8's second claim, and the reason `order` joins the reporting recommendation."""
    j = interactions()
    involved = sum("order" in v["largest_interaction"] for v in j.values())
    assert involved >= 3, {k: v["largest_interaction"] for k, v in j.items()}


def test_the_obvious_candidate_pair_is_not_the_big_one():
    """We say plainly that ectopic x s_max_frac was the intuitive guess and is small. If that
    ever stops being true the sentence has to go."""
    j = interactions()
    for cell, v in j.items():
        assert v["ectopic_x_s_max_frac"] < 0.02, (cell, v["ectopic_x_s_max_frac"])


def test_q_grid_is_an_identity_for_alpha_but_not_for_delta_alpha():
    """Section 3.5 rests on this distinction: exactly zero for alpha because both grids contain
    q = 2, non-zero for the width because it uses the whole spectrum."""
    a = {v["axis"]: v["eta_sq"] for v in load("fantasia_alpha")["variance"]}
    d = {v["axis"]: v["eta_sq"] for v in load("fantasia_delta_alpha")["variance"]}
    assert a["q_grid"] == 0.0
    assert 0.0 < d["q_grid"] < 1e-4


def test_the_free_axes_claim_holds_including_interactions():
    """Section 5.1 names three free axes. `spacing` was a fourth until interactions were run."""
    import pandas as pd
    path = os.path.join(RESULTS, "interactions.csv")
    if not os.path.exists(path):
        pytest.skip("interactions.csv not present")
    inter = pd.read_csv(path)
    for ax in ("n_scales", "overlap", "normalise"):
        s = inter[(inter.axis_a == ax) | (inter.axis_b == ax)]
        assert s.eta_sq_interaction.max() < 0.002, f"{ax} is no longer free"
    sp = inter[(inter.axis_a == "spacing") | (inter.axis_b == "spacing")]
    assert sp.eta_sq_interaction.max() > 0.02, "spacing looks free again; §5.1 needs revisiting"


def test_the_replication_table_carries_a_matched_length_column():
    """The retraction has to be in the table, not only in the prose under it."""
    t = draft_text()
    assert "Disease, α, matched length" in t


def test_the_scale_range_direction_matches_the_effect_table():
    """An earlier draft said N/20 sat at the negative end of the curve and N/4 at the positive
    end; the effect table says the opposite. The direction is asserted against results/, never
    against the sentence, so the prose cannot drift back."""
    import pandas as pd
    path = os.path.join(RESULTS, "effects_alpha.csv")
    if not os.path.exists(path):
        pytest.skip("effects_alpha.csv not present")
    e = pd.read_csv(path).sort_values("d").reset_index(drop=True)
    means = e.groupby("s_max_frac").d.mean()
    assert means[0.25] < means[0.10] < means[0.05], means.to_dict()
    low = e.head(len(e) // 10).s_max_frac.value_counts(normalize=True)
    assert low.idxmax() == 0.25, low.to_dict()
    t = draft_text()
    assert f"{100 * low[0.25]:.1f}%" in t, "the decile share quoted in 3.1 is not the one in results/"
    assert "N/20 concentrates at the negative end" not in t


# ---------------------------------------------------------------------------------------------
# The synthetic arm (section 3.9). Reads results/synthetic_summary.json and the ectopic CSV.
# ---------------------------------------------------------------------------------------------

def synthetic():
    path = os.path.join(RESULTS, "synthetic_summary.json")
    if not os.path.exists(path):
        pytest.skip("synthetic arm not run; python src/synthetic.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_draft_quotes_the_synthetic_fit_count_and_span():
    s = synthetic()
    t = draft_text()
    assert f"{s['n_fits']:,} fits" in t, s["n_fits"]
    assert f"{s['median_span_within_series']:.3f}" in t, s["median_span_within_series"]


def test_the_unbiasedness_claim_matches_the_summary():
    """The section leads on the estimator being unbiased at conventional settings. If that stops
    being true the sentence has to go, so it is asserted rather than quoted."""
    s = synthetic()
    assert abs(s["conventional_bias"]) < 0.001
    assert f"{s['conventional_bias']:.4f}".lstrip("-") in draft_text().replace("\u2212", "-")


def test_the_ectopic_arm_numbers_in_the_draft_are_the_measured_ones():
    import pandas as pd
    path = os.path.join(RESULTS, "synthetic_ectopic.csv")
    if not os.path.exists(path):
        pytest.skip("ectopic arm not run")
    ect = pd.read_csv(path)
    t = draft_text()
    for rate in (0.01, 0.05):
        m = ect[ect.rate == rate].groupby("ectopic").bias.apply(lambda x: x.abs().mean())
        for level in ("none", "pct20_drop", "pct20_interp", "mad4_drop"):
            assert f"{m[level]:.3f}" in t, (rate, level, m[level])
