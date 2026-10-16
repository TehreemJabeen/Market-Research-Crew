from types import SimpleNamespace

from app.agents.analyst import ClaimThemeLink, LinkSet, fact_status_from_relations, link_evidence, run_analyst
from app.state import Claim, ClaimStatus, Plan, Theme
from tests.fakes import make_fake_deps


def claim(i, entity, dim, text="a fact about the product"):
    return Claim(id=f"c{i}", task_id="t1", entity=entity, dimension=dim, text=text, source_url="u",
                 supporting_quote="q" * 25, status=ClaimStatus.VERIFIED)


def theme(i, entity, dim, stance="negative", n=3):
    return Theme(id=f"th{i}", entity=entity, dimension=dim, label=f"theme {i}", snippet_ids=["s"], n_snippets=n,
                 n_threads=n, n_positive=0, n_negative=n, n_neutral=0, stance=stance, confidence="medium", examples=[])


class LinkLLM:
    def __init__(self, links):
        self.links, self.calls = links, 0

    def structured(self, system, prompt, schema):
        self.calls += 1
        return LinkSet(links=self.links)


def test_fact_status_precedence():
    assert fact_status_from_relations([]) == "no_user_evidence"
    assert fact_status_from_relations(["unrelated"]) == "no_user_evidence"
    assert fact_status_from_relations(["supports"]) == "confirmed"
    assert fact_status_from_relations(["supports", "qualifies"]) == "qualified"
    assert fact_status_from_relations(["supports", "qualifies", "contradicts"]) == "disputed"


def test_link_evidence_drops_invented_cross_entity_and_unknown_relations():
    claim_refs = {1: claim(1, "Alpha", "pricing"), 2: claim(2, "Beta", "pricing")}
    theme_refs = {3: theme(1, "Alpha", "pricing")}
    llm = LinkLLM([
        ClaimThemeLink(claim=1, theme=3, relation=" Qualifies "),  # fine after cleaning
        ClaimThemeLink(claim=2, theme=3, relation="supports"),  # Beta fact vs an Alpha theme: dropped
        ClaimThemeLink(claim=9, theme=3, relation="supports"),  # no such claim: dropped
        ClaimThemeLink(claim=1, theme=3, relation="loves"),  # not a relation: dropped
    ])
    links = link_evidence(claim_refs, theme_refs, SimpleNamespace(llm=llm), [])
    assert [(l.claim_id, l.theme_id, l.relation) for l in links] == [("c1", "th1", "qualifies")]


def test_link_evidence_skips_the_llm_when_nothing_can_be_linked():
    llm = LinkLLM([])
    deps = SimpleNamespace(llm=llm)
    link_evidence({1: claim(1, "Alpha", "pricing")}, {2: theme(1, "Beta", "pricing")}, deps, [])  # different entity
    link_evidence({1: claim(1, "Alpha", "pricing")}, {2: theme(1, "Alpha", "features")}, deps, [])  # different dimension
    assert llm.calls == 0


def test_analyst_flags_unaddressed_concerns_and_gaps():
    deps = make_fake_deps()
    state = {
        "question": "q",
        "plan": Plan(entities=["Alpha", "Beta"], dimensions=["pricing", "features"]),
        "claims": [claim(1, "Beta", "pricing", "Beta starts at $12 per seat.")],
        "themes": [theme(1, "Alpha", "pricing", "negative", 3), theme(2, "Alpha", "features", "positive", 2)],
        "opinion_summaries": [],
    }
    out = run_analyst(state, deps)
    assert out["refs"] == {1: "c1", 2: "th2", 3: "th1"}  # facts first, then themes
    assert out["fact_status"] == {"c1": "no_user_evidence"}
    assert out["unaddressed"] == ["th1"]  # a negative theme no fact addresses; the positive one is not a concern
    assert out["evidence_table"].rows[0][3] == "No user evidence"
    assert "Users discuss features, but no verified facts were found." in out["gaps"]
    assert "No verified information on pricing for Alpha." in out["gaps"]
    assert out["public_opinion"]  # themes exist, so a scoped read of public opinion is written


def test_analyst_without_community_evidence_does_no_opinion_work():
    deps = make_fake_deps(enable_community=False)
    state = {
        "question": "q",
        "plan": Plan(entities=["Alpha"], dimensions=["pricing"]),
        "claims": [claim(1, "Alpha", "pricing")],
        "themes": [theme(1, "Alpha", "pricing")],  # present in state but the branch is switched off
    }
    out = run_analyst(state, deps)
    assert out["refs"] == {1: "c1"} and out["links"] == [] and out["evidence_table"] is None
    assert out["public_opinion"] == [] and out["unaddressed"] == []
