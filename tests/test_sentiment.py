import importlib.util
import re

import pytest

from app.config import Settings
from app.tools.sentiment import (
    NEUTRAL,
    HFScorer,
    LLMScorer,
    Polarity,
    ScoreItem,
    ScoreOut,
    ScoresOut,
    label_to_polarity,
    make_scorer,
)


class StubLLM:
    def __init__(self, answer=None):
        self.prompts = []
        self.answer = answer

    def structured(self, system, prompt, schema):
        self.prompts.append(prompt)
        if self.answer is not None:
            return self.answer
        n = len(re.findall(r"^\[\d+\]", prompt, re.M))
        return ScoresOut(items=[ScoreOut(id=i + 1, label="negative", confidence=0.5) for i in range(n)])


def items(n):
    return [ScoreItem("Alpha", "pricing", f"text {i}") for i in range(n)]


def test_label_vocabularies_map_to_polarity():
    assert label_to_polarity("positive", 0.9) == Polarity("positive", 0.9)
    assert label_to_polarity("NEGATIVE", 0.8) == Polarity("negative", -0.8)
    assert label_to_polarity("LABEL_0", 0.7) == Polarity("negative", -0.7)
    assert label_to_polarity("LABEL_1", 0.7) == Polarity("neutral", 0.0)
    assert label_to_polarity("LABEL_2", 0.6) == Polarity("positive", 0.6)
    assert label_to_polarity("1 star", 0.9).label == "negative"
    assert label_to_polarity("5 stars", 0.9).label == "positive"
    assert label_to_polarity("3 stars", 0.9).label == "neutral"
    assert label_to_polarity("mystery", 0.9) == NEUTRAL
    assert label_to_polarity("positive", 7).score == 1.0  # confidence is clamped


def test_hf_scorer_with_an_injected_pipeline():
    seen = {}

    def pipe(texts, **kwargs):
        seen.update(kwargs, n=len(texts))
        return [{"label": "negative", "score": 0.9} for _ in texts]

    scorer = HFScorer("some/model", pipeline_fn=pipe)
    assert scorer.score(items(2)) == [Polarity("negative", -0.9)] * 2
    assert seen["truncation"] is True and seen["max_length"] == 512 and seen["n"] == 2

    nested = HFScorer("m", pipeline_fn=lambda texts, **kw: [[{"label": "POSITIVE", "score": 0.8}] for _ in texts])
    assert nested.score(items(1)) == [Polarity("positive", 0.8)]  # list-of-candidates shape

    def never(texts, **kw):
        raise AssertionError("no items, so the model must not be called")

    assert HFScorer("m", pipeline_fn=never).score([]) == []


def test_hf_scorer_needs_its_libraries_and_make_scorer_falls_back(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError):
        HFScorer("some/model")
    scorer = make_scorer(Settings(sentiment_backend="hf"), StubLLM())
    assert isinstance(scorer, LLMScorer) and "hf scorer unavailable" in scorer.note
    plain = make_scorer(Settings(sentiment_backend="llm"), StubLLM())
    assert isinstance(plain, LLMScorer) and not hasattr(plain, "note")


def test_llm_scorer_batches_and_defaults_missing_ids_to_neutral():
    llm = StubLLM()
    out = LLMScorer(llm).score(items(45))
    assert len(llm.prompts) == 3  # 20 + 20 + 5
    assert out == [Polarity("negative", -0.5)] * 45

    partial = LLMScorer(StubLLM(ScoresOut(items=[ScoreOut(id=1, label="positive", confidence=0.9)])))
    assert partial.score(items(2)) == [Polarity("positive", 0.9), NEUTRAL]


def test_score_out_tolerates_messy_model_output():
    messy = ScoreOut(id=1, label="Mixed", confidence="very")
    assert messy.label == "neutral" and messy.confidence == 0.7
    assert ScoreOut(id=1, label=" Positive ", confidence=5).confidence == 1.0
