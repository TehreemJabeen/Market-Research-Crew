import pytest

from app.agents.planner import (
    Candidate,
    Candidates,
    QueryAnalysis,
    SubQuestion,
    SubQuestions,
    organize_sub_questions,
    overlaps,
    run_planner,
    score_candidate,
    select_dimensions,
)
from app.config import Settings
from app.deps import Deps
from app.llm import BaseLLM
from tests.fakes import URL_A, URL_B, fake_fetch, fake_search


class ScriptedLLM(BaseLLM):
    """Returns pre-set structured answers per schema, in order; the last answer repeats."""

    def __init__(self, script):
        super().__init__(rpm=0, max_calls=100)
        self.script = {k: list(v) for k, v in script.items()}

    def structured(self, system, prompt, schema, retries=2):
        self._reserve()
        queue = self.script[schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]


ANALYSIS = QueryAnalysis(
    rewritten_question="Compare Alpha and Beta for small teams",
    entities=["Alpha", "Beta"],
    audience="small teams",
    decision="which to adopt",
    scout_queries=["alpha vs beta"],
)


def cand(name, rel=4, diff=4, ans=4, urls=()):
    return Candidate(name=name, why_it_matters=f"{name} matters", relevance=rel, differentiation=diff, answerable=ans, source_urls=list(urls))


def make_deps(script, search=fake_search, **settings):
    script = {QueryAnalysis: [ANALYSIS], **script}
    return Deps(llm=ScriptedLLM(script), search=search, fetch=fake_fetch, settings=Settings(rpm=0, **settings))


def plan_for(deps, **state):
    return run_planner({"question": "Compare Alpha and Beta", **state}, deps)


# ---- selection (pure) --------------------------------------------------------------------------


def test_score_rewards_relevance_and_evidence():
    strong = score_candidate(cand("a", 5, 5, 5), n_sources=3)
    weak = score_candidate(cand("b", 2, 2, 2), n_sources=0)
    assert strong > weak
    assert score_candidate(cand("c", 4, 4, 4), 4) > score_candidate(cand("c", 4, 4, 4), 0)
    assert score_candidate(cand("c", 4, 4, 4), 10) == score_candidate(cand("c", 4, 4, 4), 4)  # bonus is capped
    junk = cand("d", 99, -5, "oops")  # junk ratings are coerced and clamped, never a validation error
    assert (junk.relevance, junk.differentiation, junk.answerable) == (5.0, 1.0, 3.0)
    assert cand("e", "4/5", "high", None).relevance == 4.0


def test_select_keeps_best_and_drops_low_scores():
    s = Settings(min_dimensions=2, max_dimensions=3)
    cands = [cand("pricing", 5, 5, 5), cand("features", 4, 4, 4), cand("integrations", 4, 3, 4), cand("trivia", 1, 1, 1), cand("lore", 1, 1, 1)]
    names = [d.name for d in select_dimensions(cands, set(), [], s)]
    assert names == ["pricing", "features", "integrations"]


def test_select_tops_up_to_minimum_when_few_pass_the_bar():
    s = Settings(min_dimensions=3)
    cands = [cand("pricing", 5, 5, 5), cand("meh one", 1, 1, 1), cand("meh two", 1, 2, 1), cand("meh three", 1, 1, 1)]
    names = [d.name for d in select_dimensions(cands, set(), [], s)]
    assert len(names) == 3 and names[0] == "pricing"


def test_made_up_source_urls_earn_no_evidence_bonus():
    s = Settings(min_dimensions=1)
    cands = [cand("real", urls=[URL_A, URL_B]), cand("fake", urls=["https://made.up/x", "https://made.up/y"])]
    chosen = {d.name: d for d in select_dimensions(cands, {URL_A, URL_B}, [], s)}
    assert chosen["real"].n_sources == 2 and chosen["fake"].n_sources == 0
    assert chosen["real"].score > chosen["fake"].score


def test_overlapping_names_are_deduplicated_keeping_the_higher_score():
    s = Settings(min_dimensions=1)
    cands = [cand("Pricing", 3, 3, 3), cand("Pricing and plans", 5, 5, 5), cand("features")]
    names = [d.name for d in select_dimensions(cands, set(), [], s)]
    assert names == ["Pricing and plans", "features"]
    assert overlaps("pricing", "Pricing & Plans") and not overlaps("api", "rapid")


def test_forced_dimensions_come_first_and_reuse_matching_candidate_notes():
    s = Settings(min_dimensions=1, max_dimensions=3)
    cands = [cand("pricing", 5, 5, 5), cand("Security and compliance", 3, 3, 3)]
    chosen = select_dimensions(cands, set(), ["security", "uptime"], s)
    assert [d.name for d in chosen[:2]] == ["security", "uptime"]
    assert chosen[0].why == "Security and compliance matters" and chosen[0].forced
    assert chosen[1].why == "requested by the user"
    assert "pricing" in [d.name for d in chosen]  # remaining slots still filled by score


# ---- sub-question coverage (pure) ----------------------------------------------------------------


def sq(dim, q):
    return SubQuestion(dimension=dim, question=q)


def test_organize_caps_per_dimension_and_dedupes():
    s = Settings(subq_per_dim_max=2, max_subquestions=8)
    items = [sq("pricing", "p1"), sq("pricing", "p2"), sq("pricing", "p3"), sq("pricing", "P1!"), sq("features", "f1")]
    kept, missing = organize_sub_questions(items, ["pricing", "features"], s)
    assert [(k.dimension, k.question) for k in kept] == [("pricing", "p1"), ("pricing", "p2"), ("features", "f1")]
    assert missing == []


def test_organize_reports_missing_dimensions_and_ignores_unknown_ones():
    kept, missing = organize_sub_questions([sq("pricing", "p1"), sq("weather", "w1")], ["pricing", "features"], Settings())
    assert [k.question for k in kept] == ["p1"] and missing == ["features"]


def test_total_cap_trims_extras_but_never_a_dimensions_only_question():
    s = Settings(subq_per_dim_max=2, max_subquestions=3)
    items = [sq("a", "a1"), sq("a", "a2"), sq("b", "b1"), sq("b", "b2"), sq("c", "c1")]
    kept, missing = organize_sub_questions(items, ["a", "b", "c"], s)
    assert len(kept) == 3 and {k.dimension for k in kept} == {"a", "b", "c"} and missing == []


# ---- the agent with a scripted model ---------------------------------------------------------------


def test_missing_dimension_is_repaired_with_one_extra_call():
    deps = make_deps({
        Candidates: [Candidates(candidates=[cand("pricing", 5, 5, 5), cand("features", 4, 4, 4)])],
        SubQuestions: [SubQuestions(items=[sq("pricing", "pricing of Alpha and Beta")]),
                       SubQuestions(items=[sq("features", "features of Alpha and Beta")])],
    })
    out = plan_for(deps)
    assert {t.dimension for t in out["tasks"]} == {"pricing", "features"}
    assert deps.llm.calls == 4  # analyze, candidates, sub-questions, repair
    assert not any("template" in line for line in out["trace"])


def test_still_missing_after_repair_gets_a_template_question():
    deps = make_deps({
        Candidates: [Candidates(candidates=[cand("pricing", 5, 5, 5), cand("features", 4, 4, 4)])],
        SubQuestions: [SubQuestions(items=[sq("pricing", "pricing of Alpha and Beta")])],  # model never covers 'features'
    })
    out = plan_for(deps)
    by_dim = {t.dimension: t.sub_question for t in out["tasks"]}
    assert set(by_dim) == {"pricing", "features"}
    assert "features" in by_dim["features"] and "Alpha" in by_dim["features"] and "small teams" in by_dim["features"]
    assert any("template" in line for line in out["trace"])


def test_scout_failure_does_not_stop_planning():
    def broken_search(query, n):
        raise RuntimeError("search down")

    deps = make_deps(
        {Candidates: [Candidates(candidates=[cand("pricing", 5, 5, 5, urls=[URL_A])])],
         SubQuestions: [SubQuestions(items=[sq("pricing", "pricing of Alpha and Beta")])]},
        search=broken_search, min_dimensions=1,
    )
    out = plan_for(deps)
    assert out["plan"].dimensions == ["pricing"]
    assert out["plan"].dimension_evidence == {"pricing": 0}  # no pages were read, so no evidence counted
    assert any("search failed" in line for line in out["trace"])


def test_scout_can_be_switched_off():
    def must_not_search(query, n):
        raise AssertionError("scout is disabled, search must not be called")

    deps = make_deps(
        {Candidates: [Candidates(candidates=[cand("pricing", 5, 5, 5)])],
         SubQuestions: [SubQuestions(items=[sq("pricing", "pricing of Alpha and Beta")])]},
        search=must_not_search, enable_scout=False, min_dimensions=1,
    )
    out = plan_for(deps)
    assert out["plan"].dimensions == ["pricing"]
    assert any("disabled" in line for line in out["trace"])


def test_no_candidates_falls_back_to_generic_dimensions():
    deps = make_deps({
        Candidates: [Candidates(candidates=[])],
        SubQuestions: [SubQuestions(items=[])],
    })
    out = plan_for(deps)
    assert out["plan"].dimensions == ["pricing", "features", "user experience"]
    assert len(out["tasks"]) == 3  # template questions cover all of them
    assert all(t.dimension for t in out["tasks"])
