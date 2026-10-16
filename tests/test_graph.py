import pytest

from app.llm import BudgetExceeded
from app.runner import run_crew
from app.state import ClaimStatus
from app.tools.verify import citation_accuracy
from tests.fakes import FABRICATED_A, FABRICATED_B, make_fake_deps


def _run(**overrides):
    deps = make_fake_deps(**overrides)
    return run_crew("Compare Alpha and Beta", deps), deps


def _count(final, prefix):
    return sum(line.startswith(prefix) for line in final["trace"])


def test_full_crew_end_to_end():
    final, deps = _run()
    claims = {c.id: c for c in final["claims"]}

    # Hallucinated quotes are caught by the deterministic check; real ones pass.
    rejected = [c for c in claims.values() if c.status == ClaimStatus.REJECTED]
    assert {c.supporting_quote for c in rejected} == {FABRICATED_A, FABRICATED_B}
    assert all("not found in source" in c.check_note for c in rejected)
    assert sum(c.status == ClaimStatus.VERIFIED for c in claims.values()) == 4

    # The features task had nothing verified on attempt 1, so it was sent back once and then succeeded.
    tasks = {t.id: t for t in final["tasks"]}
    assert tasks["t2"].attempt == 2 and tasks["t1"].attempt == 1
    assert all(t.status == "done" for t in tasks.values())
    assert final["loop_count"] == 1

    # The final report is grounded: facts AND opinion examples are verified against their pages.
    report = final["report"]
    assert "## Sources" in report
    assert "built-in AI" not in report and "99%" not in report
    assert "[99]" not in report  # invalid citation stripped
    assert "300%" not in report  # uncited factual sentence dropped
    assert final["metrics"]["invalid_refs_removed"] == 1
    assert final["metrics"]["uncited_factual_sentences_dropped"] == 1
    assert citation_accuracy(final["final_citations"], final["sources"]) == 1.0
    assert final["metrics"]["citations"] == 7 and final["metrics"]["theme_citations"] == 3
    assert len({c["n"] for c in final["final_citations"]}) == 7
    assert len(final["final_citations"]) == 10  # 4 facts + 2 example quotes for each of 3 themes

    # The comparison table is built from verified facts only.
    table = final["table"]
    assert table.columns == ["Entity", "pricing", "features"]
    assert [r[0] for r in table.rows] == ["Alpha", "Beta"]
    assert "Not found" not in " ".join(c for r in table.rows for c in r)

    assert deps.llm.calls < deps.llm.max_calls


def test_join_waits_for_the_facts_branch_even_when_it_loops():
    final, _ = _run()
    # The opinions branch finishes a round before the facts branch (which retries once). The Analyst must still run
    # exactly once, and only after the retried facts arrived.
    assert _count(final, "analyst:") == 1 and _count(final, "writer:") == 1 and _count(final, "editor:") == 1
    fact_refs = [rid for rid in final["refs"].values() if not rid.startswith("th")]
    assert len(fact_refs) == 4  # includes the two claims that only exist after the re-research round


def test_opinions_track_end_to_end():
    final, _ = _run()
    snippets = final["snippets"]

    # promotional and duplicate posts never produce snippets; the invented quote was rejected
    assert {s.post_id for s in snippets} == {"hn-1", "hn-2", "rd-3", "hn-6", "hn-7", "hn-8", "rd-9", "hn-10"}
    assert not any("ghosted" in s.quote for s in snippets)

    # hybrid scoring: the local scorer for plain snippets, the LLM for the comparative one
    scorer = {s.post_id: s.scorer for s in snippets}
    assert scorer["hn-6"] == "llm" and all(v == "fake" for k, v in scorer.items() if k != "hn-6")
    assert next(s for s in snippets if s.post_id == "hn-6").polarity == "positive"
    assert next(s for s in snippets if s.post_id == "hn-6").comparative

    # themes: statistics come from code; single-thread themes are dropped as anecdotal
    themes = {t.label: t for t in final["themes"]}
    assert set(themes) == {"too expensive for small teams", "good value and a generous free tier", "kanban and timeline views work well"}
    expensive = themes["too expensive for small teams"]
    assert (expensive.entity, expensive.stance, expensive.n_threads, expensive.confidence) == ("Alpha", "negative", 2, "low")
    assert themes["good value and a generous free tier"].stance == "positive"
    assert all(len(t.examples) == 2 for t in themes.values())

    # per-entity sentiment is thread-weighted
    summaries = {s.entity: s for s in final["opinion_summaries"]}
    assert (summaries["Alpha"].sentiment.label, summaries["Alpha"].sentiment.n_threads) == ("negative", 3)
    assert (summaries["Beta"].sentiment.label, summaries["Beta"].sentiment.n_threads, summaries["Beta"].sentiment.confidence) == ("positive", 5, "high")
    sentiment = final["sentiment_table"]
    assert sentiment.columns == ["Entity", "pricing", "features", "Overall"]
    assert sentiment.rows[0] == ["Alpha", "negative (2 threads)", "neutral (1 thread)", "negative (3 threads)"]
    assert sentiment.rows[1] == ["Beta", "positive (2 threads)", "positive (3 threads)", "positive (5 threads)"]


def test_facts_are_mapped_to_user_evidence_and_the_report_keeps_voices_apart():
    final, _ = _run()
    claims = {c.id: c for c in final["claims"]}
    status = {claims[cid].text: s for cid, s in final["fact_status"].items()}
    assert status["Alpha costs $10 per user per month for teams."] == "qualified"
    assert status["Beta starts at $12 per seat monthly."] == "confirmed"
    assert status["Beta includes kanban boards and timeline views."] == "confirmed"
    assert status["Alpha offers a free plan for individuals."] == "no_user_evidence"
    assert final["evidence_table"].columns == ["Entity", "Fact", "User evidence", "Status"]
    assert len(final["evidence_table"].rows) == 4
    assert final["public_opinion"] and final["unaddressed"] == []

    report = final["report"]
    assert "Community theme" in report  # themes are listed as such in Sources
    assert final["metrics"]["unattributed_opinion_sentences"] == 1  # the Editor detected the unattributed opinion


def test_planner_output_flows_into_tasks_and_plan():
    final, _ = _run()
    plan = final["plan"]
    assert plan.dimensions == ["pricing", "features"]  # ordered by computed score
    assert plan.rewritten_question.startswith("Compare Alpha and Beta")
    assert plan.audience == "small teams"
    assert plan.dimension_evidence == {"pricing": 2, "features": 1}  # scouted pages that really compare on it
    assert plan.dimension_notes["pricing"]
    assert {t.id: t.dimension for t in final["tasks"]} == {"t1": "pricing", "t2": "features"}
    assert plan.outline == ["Summary", "pricing", "features", "Limitations"]
    assert any(line.startswith("scout:") for line in final["trace"])


def test_user_dimensions_are_forced_into_the_plan():
    deps = make_fake_deps()
    final = run_crew("Compare Alpha and Beta", deps, dimensions=["security"])
    assert final["plan"].dimensions[0] == "security"  # user's dimension comes first
    assert "security" in {t.dimension for t in final["tasks"]}  # and gets its own sub-question (template fallback here)
    assert final["plan"].dimension_notes["security"] == "requested by the user"


def test_facts_only_run_skips_the_opinions_branch():
    deps = make_fake_deps(enable_community=False)

    def must_not_collect(*args):
        raise AssertionError("community collection is disabled")

    deps.community = must_not_collect
    final = run_crew("Compare Alpha and Beta", deps)
    assert not final.get("themes") and not final.get("snippets")
    assert _count(final, "analyst:") == 1  # the gate does not wait for a branch that was switched off
    assert "Community theme" not in final["report"]
    assert len(final["final_citations"]) == 4
    assert final.get("evidence_table") is None


def test_ablation_without_factchecker_lets_hallucinations_through():
    final, _ = _run(enable_factchecker=False)
    acc = citation_accuracy(final["final_citations"], final["sources"])
    assert acc < 1.0  # fabricated quotes reach the report: this is what the fact-checker prevents


def test_no_verifiable_evidence_gives_limitations_only_report():
    # A threshold nothing can meet rejects every claim and every community quote.
    final, _ = _run(fuzzy_threshold=1.01, max_loops=1)
    assert "No claims could be verified" in final["report"]
    assert final["final_citations"] == []
    assert _count(final, "analyst:") == 1


def test_llm_call_budget_is_enforced():
    deps = make_fake_deps()
    deps.llm.max_calls = 2  # analyze + candidates, then the guardrail trips at the sub-question call
    with pytest.raises(BudgetExceeded):
        run_crew("Compare Alpha and Beta", deps)
