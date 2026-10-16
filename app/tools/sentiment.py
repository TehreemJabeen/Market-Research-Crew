"""Polarity scoring for opinion snippets: the "hybrid" design.

The LLM decides WHAT people are talking about (snippets, themes). A scorer decides HOW positive or negative
each snippet is. Code aggregates. The scorer is swappable:

  HFScorer   a local Hugging Face model: free, deterministic, uses no LLM quota. It scores the text, not a target:
             it cannot tell who "X is way better than Y" favours, so comparative snippets go to the LLM scorer.
  LLMScorer  asks the LLM for the stance toward a named entity on a dimension.
"""
import importlib.util
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True)
class ScoreItem:
    entity: str
    dimension: str
    text: str


@dataclass(frozen=True)
class Polarity:
    label: str  # positive | negative | neutral
    score: float  # -1..1: the label's sign times the scorer's confidence


NEUTRAL = Polarity("neutral", 0.0)


class SentimentScorer(Protocol):
    name: str

    def score(self, items: list[ScoreItem]) -> list[Polarity]: ...


def label_to_polarity(label: str, confidence: float) -> Polarity:
    """Map the label vocabularies of common models onto positive / negative / neutral.
    Handles 'positive' / 'NEGATIVE', '1 star' ... '5 stars', and LABEL_0/1/2 (negative/neutral/positive,
    the convention of the cardiffnlp sentiment models). Check your model's card if you swap it."""
    text = label.strip().lower()
    conf = max(0.0, min(1.0, float(confidence)))
    stars = re.match(r"(\d)\s*star", text)
    if stars:
        sign = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}.get(int(stars.group(1)), 0)
    elif text in ("label_0",):
        sign = -1
    elif text in ("label_1",):
        sign = 0
    elif text in ("label_2",):
        sign = 1
    elif "pos" in text:
        sign = 1
    elif "neg" in text:
        sign = -1
    else:
        sign = 0
    name = {-1: "negative", 0: "neutral", 1: "positive"}[sign]
    return Polarity(name, round(sign * conf, 3))


class HFScorer:
    name = "hf"

    def __init__(self, model: str, pipeline_fn=None):
        self.model = model
        self._pipe = pipeline_fn  # injectable so tests never download a model
        if pipeline_fn is None and (importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None):
            raise ImportError("transformers and torch are not installed (pip install -r requirements-ml.txt)")

    def _pipeline(self):
        if self._pipe is None:  # loaded on first use, so starting the app stays fast
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model)
        return self._pipe

    def score(self, items: list[ScoreItem]) -> list[Polarity]:
        if not items:
            return []
        raw = self._pipeline()([i.text for i in items], truncation=True, max_length=512, batch_size=16)
        out = []
        for result in raw:
            if isinstance(result, list):  # some versions return a list of candidates per input
                result = result[0] if result else {}
            out.append(label_to_polarity(result.get("label", ""), result.get("score", 0.0)) if result else NEUTRAL)
        return out


class ScoreOut(BaseModel):
    id: int
    label: str = "neutral"
    confidence: float = 0.7

    @field_validator("label", mode="before")
    @classmethod
    def _label(cls, value):
        value = str(value).strip().lower()
        return value if value in ("positive", "negative") else "neutral"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.7


class ScoresOut(BaseModel):
    items: list[ScoreOut] = Field(default_factory=list)


SCORER_SYSTEM = (
    "You judge the stance of short community comments toward a named product on one dimension. "
    "Answer only from the text of each comment."
)


class LLMScorer:
    name = "llm"
    BATCH = 20

    def __init__(self, llm):
        self.llm = llm

    def score(self, items: list[ScoreItem]) -> list[Polarity]:
        out: list[Polarity] = []
        for start in range(0, len(items), self.BATCH):
            batch = items[start : start + self.BATCH]
            lines = "\n".join(
                f"[{i + 1}] entity={it.entity} | dimension={it.dimension} | text: {it.text}" for i, it in enumerate(batch)
            )
            result = self.llm.structured(
                SCORER_SYSTEM,
                "For each comment give label (positive, negative or neutral) and confidence 0-1 for the stance toward "
                "the NAMED entity on the named dimension. If the comment compares products, judge only the named one. "
                "Facts without a judgement are neutral. Return one item per id.\n\n" + lines,
                ScoresOut,
            )
            by_id = {r.id: r for r in result.items}
            for i in range(len(batch)):
                r = by_id.get(i + 1)
                if r is None or r.label == "neutral":
                    out.append(NEUTRAL)
                else:
                    out.append(Polarity(r.label, round((1 if r.label == "positive" else -1) * r.confidence, 3)))
        return out


def make_scorer(settings, llm):
    """The configured scorer. If 'hf' is chosen but transformers/torch are missing, fall back to the LLM scorer
    (and say so in `.note`) so a light install still works."""
    if settings.sentiment_backend == "hf":
        try:
            return HFScorer(settings.sentiment_model)
        except ImportError as exc:
            scorer = LLMScorer(llm)
            scorer.note = f"hf scorer unavailable ({exc}); using the LLM scorer"
            return scorer
    return LLMScorer(llm)
