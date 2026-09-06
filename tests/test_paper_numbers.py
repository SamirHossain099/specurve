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
    """Physiological Measurement rescinds manuscripts whose abstract exceeds 300 words."""
    text = draft_text()
    ab = text[text.index("## Abstract"):text.index("**Keywords:**")]
    words = len(re.findall(r"\S+", re.sub(r"[*`]", "", ab.replace("## Abstract", ""))))
    assert words <= 300, f"abstract is {words} words"


def test_abstract_is_structured_with_the_journals_headings():
    text = draft_text()
    ab = text[text.index("## Abstract"):text.index("**Keywords:**")]
    for head in ("**Objective.**", "**Approach.**", "**Main results.**", "**Significance.**"):
        assert head in ab, f"missing {head}"


def test_abstract_cites_nothing_and_names_no_float():
    """IOP: no references, table numbers, figure numbers or equations in the abstract."""
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
