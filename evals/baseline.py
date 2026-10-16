"""Single-agent baseline: one search, one LLM call. The crew has to beat this to justify its cost."""
from pydantic import BaseModel, Field

from app.deps import Deps

SYSTEM = (
    "You are a market research analyst. Using ONLY the sources provided, write a Markdown report that answers the "
    "question. Cite sources as [n]. For every citation, list the source url and a word-for-word quote that supports it."
)


class BaselineCitation(BaseModel):
    n: int
    url: str
    quote: str


class BaselineOut(BaseModel):
    report: str
    citations: list[BaselineCitation] = Field(default_factory=list)


def run_single_agent(question: str, deps: Deps) -> dict:
    s = deps.settings
    sources: dict[str, str] = {}
    for r in deps.search(question, 5):
        text = deps.fetch(r.url) or r.snippet
        if text:
            sources[r.url] = text[: s.source_chars]
    blocks = "\n\n".join(f"=== SOURCE url={u} ===\n{t}" for u, t in sources.items())
    out = deps.llm.structured(SYSTEM, f"Question: {question}\n\n{blocks}", BaselineOut)
    return {
        "report": out.report,
        "final_citations": [{"n": c.n, "claim_id": "", "url": c.url, "quote": c.quote} for c in out.citations],
        "sources": sources,
        "usage": deps.llm.usage(),
    }
