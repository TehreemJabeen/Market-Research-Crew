from evals.planner_eval import dimension_scores, matches, summarize


def test_synonyms_and_word_overlap_match():
    assert matches("cost", "pricing")
    assert matches("Pricing and plans", "pricing")
    assert matches("usability", "ease of use")
    assert matches("performance and latency", "speed")
    assert not matches("security", "pricing")
    assert not matches("", "pricing")


def test_recall_and_precision():
    s = dimension_scores(["pricing", "security", "emoji support"], ["cost", "ease of use", "security", "integrations"])
    assert s["recall"] == 0.5  # found cost and security, missed ease of use and integrations
    assert s["precision"] == round(2 / 3, 3)
    assert s["missed"] == ["ease of use", "integrations"]
    assert dimension_scores([], ["a"]) == {"recall": 0.0, "precision": 0.0, "missed": ["a"]}


def test_summarize_handles_failures_and_empty_modes():
    rows = [
        {"id": "q1", "mode": "scouted", "recall": 0.8, "precision": 0.6, "n_dims": 5, "llm_calls": 4, "seconds": 3.0},
        {"id": "q2", "mode": "scouted", "error": "boom"},
    ]
    table = summarize(rows)
    assert "| scouted | 1 (1 failed) | 0.8 | 0.6 | 5 | 4 | 3.0 |" in table
    assert "| llm_only | 0 ok / 0 failed |" in table
