from app.state import Claim, ClaimStatus, Example, Link, Theme
from evals.run_evals import coverage, domains, summarize
from evals.sentiment_eval import accuracy, evaluate, macro_f1
from evals.sentiment_eval import summarize as summarize_sentiment
from evals.theme_review import review_rows, score_review
from tests.fakes import FakeScorer


def test_coverage_and_domains():
    assert coverage("Notion has a free plan", ["Notion", "free", "Coda"]) == 2 / 3
    assert coverage("anything", []) == 0.0
    cites = [{"url": "https://a.com/x"}, {"url": "https://a.com/y"}, {"url": "https://b.org/z"}, {}]
    assert domains(cites) == 2


def test_run_summary_handles_failures_and_unrun_modes():
    ok = {"id": "q", "mode": "crew", "citation_accuracy": 1.0, "citations": 10, "coverage": 0.8, "themes": 3,
          "domains": 5, "llm_calls": 50, "tokens": 9000, "seconds": 120.0}
    table = summarize([ok, {"id": "q2", "mode": "crew", "error": "boom"}])
    assert "| crew | 1 (1 failed) | 1.0 | 0.8 | 10 | 3 | 5 | 50 | 9000 | 120.0 |" in table
    assert "| single_agent | 0 ok / 0 failed |" in table


def test_sentiment_metrics():
    gold = ["positive", "negative", "neutral"]
    assert accuracy(gold, gold) == 1.0 and macro_f1(gold, gold) == 1.0
    always_positive = ["positive"] * 3
    assert accuracy(gold, always_positive) == 0.333
    assert macro_f1(gold, always_positive) == 0.167  # only the 'positive' class scores (f1 0.5 of 3 classes)
    assert accuracy([], []) == 0.0


def test_evaluate_splits_plain_and_comparative_snippets():
    items = [
        {"entity": "Alpha", "dimension": "pricing", "text": "way too expensive for us", "label": "negative"},
        {"entity": "Beta", "dimension": "pricing", "text": "Switched from Alpha to Beta and it is great", "label": "positive"},
        {"entity": "Alpha", "dimension": "features", "text": "Alpha supports lists", "label": "neutral"},
    ]
    result = evaluate(FakeScorer(), items, ["Alpha", "Beta"])
    assert result["n"] == 3 and result["n_comparative"] == 1
    assert result["accuracy"] == 1.0 and result["accuracy_plain"] == 1.0 and result["accuracy_comparative"] == 1.0
    assert "| fake | 3 | 1.0 |" in summarize_sentiment({"fake": result})


def test_theme_review_rows_and_scoring():
    claim = Claim(id="c1", task_id="t", entity="Alpha", dimension="pricing", text="Alpha costs $10.", source_url="u",
                  supporting_quote="q" * 25, status=ClaimStatus.VERIFIED)
    theme = Theme(id="th1", entity="Alpha", dimension="pricing", label="too pricey", snippet_ids=["s1"], n_snippets=2,
                  n_threads=2, n_positive=0, n_negative=2, n_neutral=0, stance="negative", confidence="low",
                  examples=[Example(url="https://hn/1", quote="so pricey")])
    links = [Link(claim_id="c1", theme_id="th1", relation="qualifies", note="caveat"),
             Link(claim_id="c1", theme_id="th1", relation="unrelated")]
    rows = review_rows({"claims": [claim], "themes": [theme], "links": links})
    assert [r["kind"] for r in rows] == ["theme", "link"]  # 'unrelated' links are not worth reviewing
    assert rows[0]["model_says"] == "negative, 2 threads" and '"so pricey" (https://hn/1)' in rows[0]["evidence"]
    assert rows[1]["model_says"] == "qualifies (caveat)"

    rows[0]["human_y_or_n"] = "Y"
    rows += [{"kind": "theme", "item": "x", "evidence": "", "model_says": "", "human_y_or_n": "n"},
             {"kind": "theme", "item": "y", "evidence": "", "model_says": "", "human_y_or_n": ""}]  # unanswered: ignored
    result = score_review(rows)
    assert result["theme"] == {"reviewed": 2, "agree": 1, "rate": 0.5}
    assert result["link"] == {"reviewed": 0, "agree": 0, "rate": None}
