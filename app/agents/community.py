"""Community researcher: one run per dimension.

Collect posts (Hacker News + Reddit), filter them in code, have the LLM pull out short opinion snippets with
word-for-word quotes, and verify every quote against its post in code. It does NOT judge sentiment (the scorer's
job) and it never produces Claims: what people say is evidence about opinion, not a fact.
"""
from pydantic import BaseModel, Field

from app.llm import BudgetExceeded
from app.opinion import filter_posts, is_comparative
from app.state import Snippet, normalize_label
from app.tools.verify import quote_score

COMMUNITY_SYSTEM = (
    "You are a Community Researcher agent. You pick out what real people say about products from community posts. "
    "You quote them exactly and never invent or paraphrase a quote."
)


class ExtractedSnippet(BaseModel):
    post_id: str
    entity: str = ""
    quote: str


class ExtractedSnippets(BaseModel):
    snippets: list[ExtractedSnippet] = Field(default_factory=list)


def _result(trace: list[str], snippets=None, sources=None) -> dict:
    return {"snippets": snippets or [], "sources": sources or {}, "trace": trace}


def run_community_researcher(payload, deps) -> dict:
    task = payload["task"]
    entities = payload["entities"]
    s = deps.settings
    dimension = task.dimension or task.sub_question
    label = f"community[{dimension}]"
    if deps.community is None:
        return _result([f"{label}: no community collector configured"])
    try:
        posts, notes = deps.community(entities, dimension, s.max_posts_per_task * 2)
        kept, stats = filter_posts(posts, s)
        trace = [f"{label}: {len(posts)} posts collected, {len(kept)} kept" + (f" (dropped: {stats})" if stats else "")]
        trace += [f"{label}: {n}" for n in notes]
        if not kept:
            return _result(trace)

        by_id = {p.id: p for p in kept}
        blocks = "\n\n".join(f"=== POST id={p.id} ===\n{p.text[: s.post_chars]}" for p in kept)
        prompt = (
            f"DIMENSION: {dimension}\n"
            f"Entities (use one exactly): {entities}\n"
            f"Context: {task.sub_question}\n\n"
            f"From the posts below, extract up to {s.max_snippets_per_dimension} opinion snippets about the listed "
            "entities on this dimension. Rules:\n"
            "- quote: copied word-for-word from ONE post (one or two sentences). Do not paraphrase or merge.\n"
            "- post_id: the id of the post the quote comes from.\n"
            "- entity: which listed entity the opinion is about, exactly as listed. Skip posts about none of them.\n"
            "- Keep only personal experiences and opinions ('we switched because...', 'it is slow on big tables'), "
            "not marketing or neutral descriptions.\n"
            "- Do not judge or summarise sentiment.\n\n" + blocks
        )
        out = deps.llm.structured(COMMUNITY_SYSTEM, prompt, ExtractedSnippets)

        snippets: list[Snippet] = []
        sources: dict[str, str] = {}
        fabricated = off_topic = 0
        for es in out.snippets[: s.max_snippets_per_dimension]:
            post = by_id.get(es.post_id)
            if post is None:
                fabricated += 1
                continue
            text = post.text[: s.post_chars]
            if quote_score(es.quote, text) < s.fuzzy_threshold:  # code checks the AI: the quote must really be there
                fabricated += 1
                continue
            entity = normalize_label(es.entity, entities)
            if entity == "other":
                off_topic += 1
                continue
            snippets.append(
                Snippet(
                    id=f"{task.id}-s{len(snippets) + 1}", task_id=task.id, post_id=post.id, thread_id=post.thread_id,
                    url=post.url, source=post.source, entity=entity, dimension=task.dimension or "other",
                    quote=es.quote.strip(), created=post.created, comparative=is_comparative(es.quote, entities),
                )
            )
            sources[post.url] = text
        trace.append(
            f"{label}: {len(snippets)} snippets ({fabricated} rejected because the quote is not in the post, "
            f"{off_topic} off-topic)"
        )
        return _result(trace, snippets, sources)
    except BudgetExceeded:
        raise
    except Exception as exc:  # one failed dimension must not sink the whole run
        return _result([f"{label}: failed: {exc}"])
