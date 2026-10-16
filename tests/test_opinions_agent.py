import pytest

from app.agents.community import run_community_researcher
from app.agents.opinions import run_opinions
from app.llm import BudgetExceeded
from app.state import Plan, Snippet, Task
from tests.fakes import make_fake_deps

PLAN = Plan(entities=["Alpha", "Beta"], dimensions=["pricing", "features"])


def snippet(i, thread, quote):
    return Snippet(id=f"s{i}", task_id="t1", post_id=f"p{i}", thread_id=thread, url=f"https://x/{i}", source="hn",
                   entity="Alpha", dimension="pricing", quote=quote)


SNIPPETS = [
    snippet(1, "a", "Alpha got way too expensive for our five person team"),
    snippet(2, "b", "the price per seat is too high once you add guests"),
]


class FailingScorer:
    name = "boom"

    def score(self, items):
        raise RuntimeError("model download failed")


class BudgetScorer:
    name = "budget"

    def score(self, items):
        raise BudgetExceeded("out of calls")


def test_scorer_failure_falls_back_to_the_llm_scorer():
    deps = make_fake_deps()
    deps.scorer = FailingScorer()
    out = run_opinions({"plan": PLAN, "snippets": SNIPPETS}, deps)
    assert {s.scorer for s in out["snippets"]} == {"llm"}
    assert any("falling back to the LLM scorer" in line for line in out["trace"])
    assert len(out["themes"]) == 1 and out["opinions_ready"] is True  # the run still produced a theme


def test_budget_errors_are_never_swallowed():
    deps = make_fake_deps()
    deps.scorer = BudgetScorer()
    with pytest.raises(BudgetExceeded):
        run_opinions({"plan": PLAN, "snippets": SNIPPETS}, deps)


def test_no_snippets_still_marks_the_branch_ready():
    out = run_opinions({"plan": PLAN, "snippets": []}, make_fake_deps())
    assert out["opinions_ready"] is True and out["themes"] == [] and out["opinion_summaries"] == []
    assert out["sentiment_table"].rows[0] == ["Alpha", "no data", "no data", "no data"]


def payload():
    return {"task": Task(id="t1", sub_question="q", dimension="pricing"), "question": "q",
            "entities": ["Alpha", "Beta"], "dimensions": ["pricing"]}


def test_community_researcher_survives_a_failing_collector():
    deps = make_fake_deps()

    def broken(entities, dimension, n):
        raise RuntimeError("network down")

    deps.community = broken
    out = run_community_researcher(payload(), deps)
    assert out["snippets"] == [] and "failed: network down" in out["trace"][0]


def test_community_researcher_reraises_budget_errors():
    deps = make_fake_deps()

    def exhausted(entities, dimension, n):
        raise BudgetExceeded("out of calls")

    deps.community = exhausted
    with pytest.raises(BudgetExceeded):
        run_community_researcher(payload(), deps)


def test_community_researcher_without_a_collector():
    deps = make_fake_deps()
    deps.community = None
    out = run_community_researcher(payload(), deps)
    assert out["snippets"] == [] and "no community collector configured" in out["trace"][0]


def test_community_researcher_rejects_invented_quotes_and_records_why():
    out = run_community_researcher(payload(), make_fake_deps())
    quotes = [s.quote for s in out["snippets"]]
    assert "Alpha support ghosted us for weeks" not in quotes  # not in any post
    assert len(quotes) == 4
    assert "1 rejected because the quote is not in the post" in out["trace"][-1]
    assert set(out["sources"]) == {s.url for s in out["snippets"]}  # every cited page's text is kept for verification
