from pydantic import BaseModel, Field

from app.llm import BudgetExceeded
from app.state import Claim, normalize_label

RESEARCHER_SYSTEM = (
    "You are a Researcher agent in a market-research crew. You extract specific, checkable facts "
    "from the web sources you are given. You never use outside knowledge."
)
QUERY_SYSTEM = "You write web search queries for a researcher whose previous attempt found too little usable evidence."


class ExtractedClaim(BaseModel):
    entity: str = ""
    dimension: str = ""
    text: str
    source_url: str
    supporting_quote: str


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class QueryList(BaseModel):
    queries: list[str] = Field(default_factory=list)


def _dimension_label(raw: str, dimensions: list[str], task_dimension: str) -> str:
    """Prefer the model's label if it names a planned dimension; otherwise fall back to the task's own dimension."""
    label = normalize_label(raw, dimensions)
    return task_dimension if label == "other" and task_dimension else label


def _done(task, attempt: int, claims=None, sources=None, note: str = "") -> dict:
    return {
        "tasks": [task.model_copy(update={"status": "done", "attempt": attempt, "feedback": ""})],
        "claims": claims or [],
        "sources": sources or {},
        "trace": [f"researcher {task.id} (attempt {attempt}): {note}"],
    }


def run_researcher(payload, deps) -> dict:
    """Runs once per sub-question, in parallel with its siblings."""
    task = payload["task"]
    entities, dimensions = payload["entities"], payload["dimensions"]
    s = deps.settings
    attempt = task.attempt + 1
    try:
        queries = [task.sub_question]
        if task.attempt > 0:  # retry: ask the model for different queries given what went wrong
            out = deps.llm.structured(
                QUERY_SYSTEM,
                f"Sub-question: {task.sub_question}\nWhat went wrong last time: {task.feedback}\n"
                "Write 2 different search queries likely to surface authoritative sources "
                "(official pages, documentation, pricing pages, reputable reviews).",
                QueryList,
            )
            queries = [q for q in out.queries if q.strip()][:2] or queries

        results, seen = [], set()
        for q in queries:
            for r in deps.search(q, s.max_sources_per_task + 2):
                if r.url not in seen:
                    seen.add(r.url)
                    results.append(r)

        sources: dict[str, str] = {}
        for r in results[: s.max_sources_per_task]:
            text = deps.fetch(r.url) or r.snippet
            if text:
                sources[r.url] = text[: s.source_chars]
        if not sources:
            return _done(task, attempt, note="no readable sources found")

        blocks = "\n\n".join(f"=== SOURCE url={u} ===\n{t}" for u, t in sources.items())
        prompt = (
            f"SUB-QUESTION: {task.sub_question}\n"
            f"ATTEMPT: {attempt}\n"
            f"Entities (use one of these exactly, else leave blank): {entities}\n"
            f"Dimensions (use one of these exactly, else leave blank): {dimensions}\n"
            + (f"This sub-question serves the dimension: {task.dimension}\n" if task.dimension else "")
            + (f"Previous problems to avoid: {task.feedback}\n" if task.feedback else "")
            + "\nExtract up to 8 factual claims that help answer the sub-question. Rules:\n"
            "- Each claim must be self-contained, specific (numbers, names, features, prices), and about one entity.\n"
            "- supporting_quote MUST be copied word-for-word from the source text (1-2 sentences). Do not paraphrase it.\n"
            "- source_url MUST be one of the source urls below.\n"
            "- If the sources do not contain a fact, do not invent it. Return fewer claims instead.\n\n"
            f"{blocks}"
        )
        out = deps.llm.structured(RESEARCHER_SYSTEM, prompt, ExtractedClaims)
        claims = [
            Claim(
                id=f"{task.id}-a{attempt}-c{i + 1}",
                task_id=task.id,
                entity=normalize_label(ec.entity, entities),
                dimension=_dimension_label(ec.dimension, dimensions, task.dimension),
                text=ec.text.strip(),
                source_url=ec.source_url,
                supporting_quote=ec.supporting_quote.strip(),
            )
            for i, ec in enumerate(out.claims[:8])
            if ec.source_url in sources and ec.text.strip()
        ]
        return _done(task, attempt, claims, sources, f"{len(claims)} claims from {len(sources)} sources")
    except BudgetExceeded:
        raise  # an exhausted call budget must stop the run, not hide as a quiet empty result
    except Exception as exc:  # one failed researcher must not sink the whole run
        return _done(task, attempt, note=f"failed: {exc}")
