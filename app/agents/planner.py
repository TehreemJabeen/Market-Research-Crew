"""Planner v2.

1. analyze   (LLM)   rewrite the question, infer who is deciding what, propose scouting searches
2. scout     (code)  search + read real comparison pages
3. propose   (LLM)   candidate dimensions, each rated 1-5 on relevance / differentiation / answerability,
                     with the scouted pages that really compare on it
4. select    (code)  score, dedupe, apply user-forced dimensions, keep the best 3-6
5. questions (LLM)   >=1 sub-question per chosen dimension
6. validate  (code)  coverage check, one repair call, then a template fallback
"""
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from app.agents.scout import run_scout
from app.llm import BudgetExceeded, LLMError
from app.state import Plan, Task, normalize_label

ANALYZE_SYSTEM = (
    "You are the Planner agent in a market-research crew. You work out what the person is really trying to decide "
    "so the research can focus on what matters to them. You do not answer the question yourself."
)
CRITERIA_SYSTEM = (
    "You are the Planner agent. You choose the comparison criteria (dimensions) that actually matter for a decision, "
    "using evidence from real comparison pages where possible."
)
SUBQ_SYSTEM = "You are the Planner agent. You turn comparison dimensions into specific, searchable research questions."


class QueryAnalysis(BaseModel):
    rewritten_question: str
    entities: list[str] = Field(default_factory=list)
    category: str = ""
    audience: str = ""
    decision: str = ""
    scout_queries: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    name: str
    why_it_matters: str = ""
    relevance: float = 3
    differentiation: float = 3
    answerable: float = 3
    source_urls: list[str] = Field(default_factory=list)

    @field_validator("relevance", "differentiation", "answerable", mode="before")
    @classmethod
    def _tolerant_rating(cls, value):
        # A messy rating ('4/5', 'high', 99) must not cost a whole retry: coerce and clamp to 1-5 here.
        return clamp(value)


class Candidates(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)


class SubQuestion(BaseModel):
    dimension: str
    question: str


class SubQuestions(BaseModel):
    items: list[SubQuestion] = Field(default_factory=list)


@dataclass
class ChosenDimension:
    name: str
    why: str
    score: float
    n_sources: int
    forced: bool = False


# ---- pure helpers (unit-tested without any LLM) -----------------------------------------------


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _tokens(text: str) -> set[str]:
    return set(_key(text).split())


def overlaps(a: str, b: str) -> bool:
    """True if one name's words are all contained in the other's ('pricing' vs 'pricing and plans')."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def clamp(value, lo: float = 1.0, hi: float = 5.0) -> float:
    """Turn whatever the model wrote into a number in [lo, hi]; unparseable text becomes the neutral 3."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        found = re.search(r"-?\d+(?:\.\d+)?", str(value))  # '4/5' -> 4
        number = float(found.group()) if found else 3.0
    return max(lo, min(hi, number))


def score_candidate(c: Candidate, n_sources: int) -> float:
    """0.4*relevance + 0.3*differentiation + 0.3*answerability (each 1-5), plus an evidence bonus of 0.25 per
    scouted page that compares on it (max +1.0). The model rates; code does the arithmetic and the choosing."""
    base = 0.4 * clamp(c.relevance) + 0.3 * clamp(c.differentiation) + 0.3 * clamp(c.answerable)
    return round(base + 0.25 * min(n_sources, 4), 2)


def select_dimensions(candidates: list[Candidate], scouted_urls: set[str], forced: list[str], settings) -> list[ChosenDimension]:
    scored: list[ChosenDimension] = []
    for c in candidates:
        name = c.name.strip()
        if not name:
            continue
        # Only pages we really fetched count as evidence: a made-up URL earns nothing.
        n_sources = len({u for u in c.source_urls if u in scouted_urls})
        scored.append(ChosenDimension(name, c.why_it_matters.strip(), score_candidate(c, n_sources), n_sources))
    scored.sort(key=lambda d: d.score, reverse=True)

    chosen: list[ChosenDimension] = []

    def add(d: ChosenDimension) -> None:
        if not any(overlaps(d.name, x.name) for x in chosen):
            chosen.append(d)

    for name in forced:  # the user's own dimensions always come first
        match = next((d for d in scored if overlaps(d.name, name)), None)
        add(ChosenDimension(
            name,
            match.why if match else "requested by the user",
            match.score if match else 0.0,
            match.n_sources if match else 0,
            forced=True,
        ))
    limit = max(settings.max_dimensions, len(chosen))
    for d in scored:
        if len(chosen) >= limit:
            break
        if d.score >= settings.min_dimension_score:
            add(d)
    for d in scored:  # too few passed the bar: top up with the best of the rest
        if len(chosen) >= min(settings.min_dimensions, limit):
            break
        add(d)
    return chosen


def organize_sub_questions(items: list[SubQuestion], dims: list[str], settings) -> tuple[list[SubQuestion], list[str]]:
    """Group by dimension, drop duplicates, cap per dimension and overall (never a dimension's first question).
    Returns (kept questions in dimension order, dimensions that still have none)."""
    by_dim: dict[str, list[SubQuestion]] = {d: [] for d in dims}
    seen: set[str] = set()
    for item in items:
        dim = normalize_label(item.dimension, dims)
        question = item.question.strip()
        key = _key(question)
        if dim == "other" or not question or key in seen:
            continue
        seen.add(key)
        if len(by_dim[dim]) < settings.subq_per_dim_max:
            by_dim[dim].append(SubQuestion(dimension=dim, question=question))
    missing = [d for d in dims if not by_dim[d]]

    limit = max(settings.max_subquestions, len(dims))
    while sum(len(v) for v in by_dim.values()) > limit:
        for d in reversed(dims):
            if len(by_dim[d]) > 1:
                by_dim[d].pop()
                break
        else:
            break
    return [sq for d in dims for sq in by_dim[d]], missing


def fallback_question(dimension: str, entities: list[str], audience: str) -> str:
    subject = ", ".join(entities) if entities else "the options"
    return f"How do {subject} compare on {dimension}" + (f" for {audience}" if audience else "") + "?"


# ---- the agent ---------------------------------------------------------------------------------


def run_planner(state, deps) -> dict:
    s = deps.settings
    question = state["question"]
    forced = [d.strip() for d in (state.get("user_dimensions") or []) if d and d.strip()]
    trace: list[str] = []

    # 1. analyze
    qa = deps.llm.structured(
        ANALYZE_SYSTEM,
        f"Research question: {question}\n\nProduce:\n"
        "- rewritten_question: one clear, self-contained question that keeps the audience and constraints\n"
        "- entities: the companies/products/segments to compare (1-6 short names; if the question asks for "
        "'alternatives' or 'the best', name the most relevant real ones)\n"
        "- category: what kind of thing is being compared\n"
        "- audience: who is making the decision\n"
        "- decision: what they are trying to decide or achieve\n"
        "- scout_queries: 2-3 web search queries that would find comparison articles, buyer's guides or "
        "reviews for this decision (queries for guides, not for the answers)",
        QueryAnalysis,
    )
    rewritten = qa.rewritten_question.strip() or question
    entities = [e.strip() for e in qa.entities if e.strip()][:6]
    audience = qa.audience.strip()

    # 2. scout
    sources: dict[str, str] = {}
    if s.enable_scout:
        sources = run_scout([q for q in qa.scout_queries if q.strip()] or [rewritten], deps, trace)
    else:
        trace.append("scout: disabled (model knowledge only)")

    # 3. propose candidate dimensions
    source_block = (
        "\n\n".join(f"=== SOURCE url={u} ===\n{t}" for u, t in sources.items())
        if sources
        else "(no web sources available: propose from your own knowledge and leave source_urls empty)"
    )
    forced_note = f"\nThe user REQUIRES these dimensions; include and rate them: {forced}\n" if forced else ""
    candidates = deps.llm.structured(
        CRITERIA_SYSTEM,
        f"Question: {rewritten}\nAudience: {audience}\nDecision: {qa.decision}\nCategory: {qa.category}\n"
        f"Entities: {entities}\n{forced_note}\n"
        "List 6-10 candidate comparison criteria. For each give:\n"
        "- name: 1-3 words, generic enough to apply to every entity\n"
        "- why_it_matters: one sentence about THIS audience's decision\n"
        "- relevance 1-5: how much it matters to this decision\n"
        "- differentiation 1-5: do the entities actually differ on it? (5 = big differences)\n"
        "- answerable 1-5: can it be answered from public web pages? (1 = needs private data)\n"
        "- source_urls: only urls from the SOURCES below whose page really compares on this criterion; "
        "empty if you added it from your own knowledge\n"
        "Include criteria the sources emphasise even if the question never mentioned them. "
        "No filler like 'overview' or 'general'.\n\n" + source_block,
        Candidates,
    )

    # 4. select (code)
    chosen = select_dimensions(candidates.candidates, set(sources), forced, s)
    if not chosen:
        trace.append("planner: no usable dimensions proposed; falling back to generic ones")
        chosen = [ChosenDimension(n, "generic fallback", 0.0, 0) for n in ("pricing", "features", "user experience")]
    dims = [d.name for d in chosen]
    trace.append(
        "planner: dimensions = "
        + ", ".join(f"{d.name} ({'user' if d.forced else f'{d.score:.2f}'}, {d.n_sources} src)" for d in chosen)
    )
    if len(chosen) < s.min_dimensions:
        trace.append(f"planner: only {len(chosen)} dimensions available (wanted {s.min_dimensions})")

    # 5. sub-questions, 6. validate coverage
    dim_lines = "\n".join(f"- {d.name}: {d.why}" for d in chosen)
    items = deps.llm.structured(
        SUBQ_SYSTEM,
        f"Question: {rewritten}\nAudience: {audience}\nEntities: {entities}\n\nDimensions:\n{dim_lines}\n\n"
        "For EVERY dimension write 1-2 sub-questions answerable by web search. Each must be about exactly one "
        "dimension, name the entities it covers, and be specific. Return items with `dimension` (exactly as listed) "
        "and `question`.",
        SubQuestions,
    ).items
    kept, missing = organize_sub_questions(items, dims, s)
    if missing:
        trace.append(f"planner: no sub-question for {missing}; asking once more")
        try:
            extra = deps.llm.structured(
                SUBQ_SYSTEM,
                f"Question: {rewritten}\nEntities: {entities}\n"
                f"These dimensions still have no sub-question: {missing}. Write 1 specific, searchable "
                "sub-question for each. Return items with `dimension` (exactly as listed) and `question`.",
                SubQuestions,
            ).items
        except BudgetExceeded:
            raise
        except LLMError as exc:
            extra = []
            trace.append(f"planner: repair call failed ({exc})")
        kept, missing = organize_sub_questions([*kept, *extra], dims, s)
    for dim in missing:
        trace.append(f"planner: used a template sub-question for '{dim}'")
        kept.append(SubQuestion(dimension=dim, question=fallback_question(dim, entities, audience)))
    kept.sort(key=lambda sq: dims.index(sq.dimension))

    tasks = [Task(id=f"t{i + 1}", sub_question=sq.question, dimension=sq.dimension) for i, sq in enumerate(kept)]
    plan = Plan(
        rewritten_question=rewritten,
        audience=audience,
        decision=qa.decision.strip(),
        entities=entities,
        dimensions=dims,
        dimension_notes={d.name: d.why for d in chosen},
        dimension_scores={d.name: d.score for d in chosen},
        dimension_evidence={d.name: d.n_sources for d in chosen},
        sub_questions=[t.sub_question for t in tasks],
        outline=["Summary", *dims, "Limitations"],
    )
    trace.append(f"planner: {len(tasks)} sub-questions across {len(dims)} dimensions; entities={entities}")
    return {"plan": plan, "tasks": tasks, "trace": trace}
