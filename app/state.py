from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONFLICTING = "conflicting"


class Plan(BaseModel):
    rewritten_question: str = ""
    audience: str = ""  # who is deciding
    decision: str = ""  # what they are trying to decide
    entities: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    dimension_notes: dict[str, str] = Field(default_factory=dict)  # why each dimension matters
    sub_questions: list[str] = Field(default_factory=list)
    outline: list[str] = Field(default_factory=list)


class Task(BaseModel):
    id: str
    sub_question: str
    dimension: str = ""  # which dimension this sub-question serves
    status: Literal["pending", "done", "needs_more"] = "pending"
    attempt: int = 0  # how many research rounds have run for this task
    feedback: str = ""  # why the previous attempt fell short (fed to the researcher)


class Claim(BaseModel):
    id: str
    task_id: str
    entity: str = ""
    dimension: str = ""
    text: str
    source_url: str
    supporting_quote: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    check_note: str = ""
    source_type: str = "web"  # facts come from the web track; opinions never become Claims (see Theme)


class Table(BaseModel):
    columns: list[str]
    rows: list[list[str]]


# ---- opinions track -------------------------------------------------------------------------------


class Snippet(BaseModel):
    """One short opinion pulled from a community post. Its quote is verified against the post in code."""

    id: str
    task_id: str
    post_id: str
    thread_id: str  # snippets from one discussion share a thread: we count threads, not comments
    url: str
    source: str = ""  # "hn" | "reddit"
    entity: str
    dimension: str
    quote: str
    created: int | None = None
    comparative: bool = False  # compares entities, so a text-only sentiment model can't be trusted on it
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    polarity_score: float = 0.0  # -1..1: signed scorer confidence
    scorer: str = ""  # which scorer produced the polarity ("hf", "llm", ...)


class Example(BaseModel):
    url: str
    quote: str
    source: str = ""


class Theme(BaseModel):
    """A recurring point people make about one entity on one dimension. Every number is computed in code."""

    id: str
    entity: str
    dimension: str
    label: str
    snippet_ids: list[str]
    n_snippets: int
    n_threads: int
    n_positive: int
    n_negative: int
    n_neutral: int
    stance: Literal["positive", "negative", "mixed", "neutral"]
    confidence: Literal["low", "medium", "high"]
    examples: list[Example] = Field(default_factory=list)


class SentimentStat(BaseModel):
    label: str  # positive | negative | mixed | neutral | no data
    score: float  # -1..1, mean of per-thread means
    n_threads: int
    confidence: str


class OpinionSummary(BaseModel):
    entity: str
    sentiment: SentimentStat
    n_snippets: int
    by_dimension: dict[str, SentimentStat] = Field(default_factory=dict)
    top_praise: list[str] = Field(default_factory=list)  # theme ids
    top_complaints: list[str] = Field(default_factory=list)  # theme ids


class Link(BaseModel):
    """How a verified fact relates to a user-opinion theme (same entity, same dimension)."""

    claim_id: str
    theme_id: str
    relation: Literal["supports", "contradicts", "qualifies", "unrelated"]
    note: str = ""


# ---- helpers shared by agents -------------------------------------------------


def normalize_label(value: str, allowed: list[str]) -> str:
    """Map a free-text entity/dimension onto the planner's canonical list, else 'other'."""
    v = (value or "").strip().casefold()
    if not v:
        return "other"
    for a in allowed:
        if v == a.casefold():
            return a
    for a in allowed:
        al = a.casefold()
        if al in v or v in al:
            return a
    return "other"


def build_refs(claims: list[Claim], themes: list["Theme"] | None = None) -> dict[int, str]:
    """Number the verified claims 1..N, then the opinion themes N+1... The writer may only cite these numbers.
    Values are claim ids or theme ids ("th1", ...)."""
    verified = [c for c in claims if c.status == ClaimStatus.VERIFIED]
    verified.sort(key=lambda c: (c.entity.casefold(), c.dimension.casefold(), c.id))
    refs = {i + 1: c.id for i, c in enumerate(verified)}
    ordered = sorted(themes or [], key=lambda t: (t.entity.casefold(), t.dimension.casefold(), -t.n_threads, t.id))
    for j, t in enumerate(ordered):
        refs[len(verified) + j + 1] = t.id
    return refs


def table_to_markdown(table: Table) -> str:
    def cell(x: str) -> str:
        return x.replace("|", "/").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = ["| " + " | ".join(cell(c) for c in row) + " |" for row in table.rows]
    return "\n".join([head, sep, *body])