from app.agents.editor import find_unattributed_opinions, find_uncited, run_editor
from app.state import Claim, ClaimStatus, Example, Plan, Theme
from tests.fakes import make_fake_deps

CLAIM = Claim(id="c1", task_id="t1", entity="Alpha", dimension="pricing", text="Alpha costs ten dollars per user.",
              source_url="https://a.example/alpha", supporting_quote="Alpha costs ten dollars per user per month.",
              status=ClaimStatus.VERIFIED)
THEME = Theme(id="th1", entity="Alpha", dimension="pricing", label="too expensive", snippet_ids=["s1", "s2"], n_snippets=2,
              n_threads=3, n_positive=0, n_negative=2, n_neutral=0, stance="negative", confidence="medium",
              examples=[Example(url="https://hn/1", quote="way too expensive for us"), Example(url="https://hn/2", quote="price per seat is too high")])


def state(draft):
    return {"draft": draft, "refs": {1: "c1", 2: "th1"}, "claims": [CLAIM], "themes": [THEME], "plan": Plan()}


def test_unattributed_opinions_are_found():
    text = (
        "- Pricing is a problem for the whole team [5].\n"
        "- Users report slow boards across several threads [5].\n"
        "- Costs $10 per seat and users complain about it [1][5].\n"
    )
    assert find_unattributed_opinions(text, {5}) == ["Pricing is a problem for the whole team [5]."]
    assert find_unattributed_opinions(text, set()) == []  # no theme numbers: nothing counts as an opinion


def test_uncited_sentences_are_found_but_tables_and_headings_are_ignored():
    text = "# Heading with no citation at all and plenty of words\n| a | b |\n- This sentence makes a claim without any citation attached.\n"
    assert find_uncited(text) == ["This sentence makes a claim without any citation attached."]


def test_sources_cover_facts_and_community_themes():
    deps = make_fake_deps()
    out = run_editor(
        state(
            "# R\n\n- Alpha costs ten dollars per user according to the page [1].\n"
            "- Users report that Alpha pricing hurts small teams in several threads [2].\n"
        ),
        deps,
    )
    report = out["report"]
    assert "1. https://a.example/alpha" in report
    assert "2. Community theme, 3 threads, negative (medium confidence)" in report
    assert "https://hn/1" in report and "https://hn/2" in report
    assert [(c["n"], c["claim_id"], c["url"]) for c in out["final_citations"]] == [
        (1, "c1", "https://a.example/alpha"), (2, "th1", "https://hn/1"), (2, "th1", "https://hn/2"),
    ]
    assert out["metrics"]["citations"] == 2 and out["metrics"]["theme_citations"] == 1
    assert out["metrics"]["unattributed_opinion_sentences"] == 0
    assert deps.llm.calls == 0  # nothing needed repair, so no LLM call


def test_citations_are_renumbered_in_order_of_first_use():
    deps = make_fake_deps()
    out = run_editor(
        state(
            "- Users report pricing pain across several threads [2].\n"
            "- Alpha costs ten dollars per user according to the page [1].\n"
        ),
        deps,
    )
    lines = out["report"].splitlines()
    assert lines[0].endswith("[1].") and "Users report" in lines[0]  # the theme was cited first, so it becomes [1]
    assert any(l.startswith("1. Community theme") for l in lines)
    assert any(l.startswith("2. https://a.example/alpha") for l in lines)
    assert out["final_citations"][0]["claim_id"] == "th1" and out["final_citations"][0]["n"] == 1


def test_editor_with_no_refs_passes_the_draft_through():
    out = run_editor({"draft": "nothing verified", "refs": {}}, make_fake_deps())
    assert out["report"] == "nothing verified" and out["final_citations"] == []
