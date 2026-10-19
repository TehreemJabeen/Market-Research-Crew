"""Web researcher: reads each entity's OFFICIAL website and extracts claims. One LLM call per task, no retries.

Code does the finding: for every entity the sub-question is about, search that entity's official domain(s) only,
fetch the top pages. The model only reads those pages and writes claims, each with a verbatim quote.
An entity with no known official site is skipped (strict official-only), and the trace says so.
"""
from pydantic import BaseModel, Field

from app.llm import BudgetExceeded
from app.state import Claim, ClaimStatus, normalize_label
from app.tools.domains import name_key, on_domain

RESEARCHER_SYSTEM = (
    "You are a Researcher agent in a market-research crew. You extract specific, checkable facts "
    "from the official websites you are given. You never use outside knowledge."
)


class ExtractedClaim(BaseModel):
    entity: str = ""
    dimension: str = ""
    text: str
    source_url: str
    supporting_quote: str


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


def _dimension_label(raw: str, dimensions: list[str], task_dimension: str) -> str:
    """Prefer the model's label if it names a planned dimension; otherwise fall back to the task's own dimension."""
    label = normalize_label(raw, dimensions)
    return task_dimension if label == "other" and task_dimension else label


def entities_for(sub_question: str, entities: list[str]) -> list[str]:
    """The entities this sub-question is about (named in it); all of them if it names none."""
    text = name_key(sub_question)
    named = [e for e in entities if name_key(e) and name_key(e) in text]
    return named or list(entities)


def _done(task, claims=None, sources=None, note: str = "") -> dict:
    return {
        "tasks": [task.model_copy(update={"status": "done"})],
        "claims": claims or [],
        "sources": sources or {},
        "trace": [f"researcher {task.id}: {note}"],
    }


def run_researcher(payload, deps) -> dict:
    """Runs once per sub-question, in parallel with its siblings."""
    task = payload["task"]
    entities, dimensions = payload["entities"], payload["dimensions"]
    official = payload.get("official_sites") or {}
    s = deps.settings
    notes: list[str] = []
    try:
        sources: dict[str, str] = {}
        owner: dict[str, str] = {}  # url -> the entity whose official site it is
        for entity in entities_for(task.sub_question, entities):
            domains = official.get(entity)
            if not domains:
                notes.append(f"{entity}: no official site known, skipped")
                continue
            try:
                results = deps.search(task.sub_question, s.max_sources_per_entity, domains)
            except Exception as exc:
                notes.append(f"{entity}: search failed ({exc})")
                continue
            # Enforced in code, whatever the search tool returned: only pages on the official domain(s).
            urls = [r for r in results if on_domain(r.url, domains) and r.url not in sources]
            for r in urls[: s.max_sources_per_entity]:
                text = deps.fetch(r.url) or r.snippet
                if text:
                    sources[r.url] = text[: s.source_chars]
                    owner[r.url] = entity
            if not any(o == entity for o in owner.values()):
                notes.append(f"{entity}: no readable pages on {', '.join(domains)}")
        skipped = f" ({'; '.join(notes)})" if notes else ""
        if not sources:
            return _done(task, note=f"no official pages to read{skipped}")

        blocks = "\n\n".join(f"=== SOURCE url={u} | official website of {owner[u]} ===\n{t}" for u, t in sources.items())
        prompt = (
            f"SUB-QUESTION: {task.sub_question}\n"
            f"Entities (use one of these exactly, else leave blank): {entities}\n"
            f"Dimensions (use one of these exactly, else leave blank): {dimensions}\n"
            + (f"This sub-question serves the dimension: {task.dimension}\n" if task.dimension else "")
            + "\nThe sources below are the entities' OWN websites, so they are marketing and documentation pages: "
            "report what they state, not whether it is the best.\n"
            "Extract up to 8 factual claims that help answer the sub-question. Rules:\n"
            "- Each claim must be self-contained, specific (numbers, names, features, prices), and about one entity.\n"
            "- supporting_quote MUST be copied word-for-word from the source text (1-2 sentences). Do not paraphrase it.\n"
            "- source_url MUST be one of the source urls below.\n"
            "- If the sources do not contain a fact, do not invent it. Return fewer claims instead.\n\n"
            f"{blocks}"
        )
        out = deps.llm.structured(RESEARCHER_SYSTEM, prompt, ExtractedClaims)
        claims = []
        for ec in out.claims[:8]:
            if ec.source_url not in sources or not ec.text.strip():
                continue
            entity = normalize_label(ec.entity, entities)
            claims.append(
                Claim(
                    id=f"{task.id}-c{len(claims) + 1}",
                    task_id=task.id,
                    entity=owner[ec.source_url] if entity == "other" else entity,
                    dimension=_dimension_label(ec.dimension, dimensions, task.dimension),
                    text=ec.text.strip(),
                    source_url=ec.source_url,
                    supporting_quote=ec.supporting_quote.strip(),
                    # there is no fact-checker any more: claims go to the analyst exactly as the model wrote them
                    status=ClaimStatus.VERIFIED,
                    check_note="not fact-checked",
                )
            )
        return _done(task, claims, sources, f"{len(claims)} claims from {len(sources)} official pages{skipped}")
    except BudgetExceeded:
        raise  # an exhausted call budget must stop the run, not hide as a quiet empty result
    except Exception as exc:  # one failed researcher must not sink the whole run
        return _done(task, note=f"failed: {exc}")