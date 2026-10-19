"""Planner: three LLM calls, nothing else.

1. rewrite     (LLM)   rewrite the question, infer who is deciding what
2. dimensions  (LLM)   3-6 comparison dimensions that matter for that decision, most important first
3. questions   (LLM)   1-2 sub-questions per dimension
Code between the calls: keep the user's numbers in the rewrite, merge in forced dimensions, drop duplicates,
trim to the limit, and make sure every dimension ends up with at least one sub-question (template fallback).
The dimensions come from the model's own knowledge: there is no web search in the planner.
"""
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.state import Plan, Task, normalize_label
from app.tools.appstore import APP_DIMENSION, app_enabled

REWRITE_SYSTEM = (
    "You are the Planner agent in a market-research crew. You work out what the person is really trying to decide "
    "so the research can focus on what matters to them. You do not answer the question yourself."
)
DIMENSIONS_SYSTEM = (
    "You are the Planner agent. You choose the comparison criteria (dimensions) that actually matter for a decision."
)
SUBQ_SYSTEM = "You are the Planner agent. You turn comparison dimensions into specific, searchable research questions."

REWRITE_EXAMPLE = (
    "Example\n"
    "Question: notion or coda, we are 8 people and want to stay under $100/month\n"
    '{"rewritten_question": "Which of Notion and Coda is the better workspace and documentation tool for an '
    '8-person team with a budget under $100 per month?", "entities": ["Notion", "Coda"], '
    '"category": "team workspace tools", "audience": "an 8-person team", '
    '"decision": "which tool to adopt within a $100 monthly budget", "is_app": false}'
)


class QueryRewrite(BaseModel):
    rewritten_question: str
    entities: list[str] = Field(default_factory=list)
    category: str = ""
    audience: str = ""
    decision: str = ""
    is_app: bool = False


class DimensionChoice(BaseModel):
    name: str
    why: str = ""


class DimensionList(BaseModel):
    dimensions: list[DimensionChoice] = Field(default_factory=list)


class SubQuestion(BaseModel):
    dimension: str
    question: str


class SubQuestions(BaseModel):
    items: list[SubQuestion] = Field(default_factory=list)


@dataclass
class ChosenDimension:
    name: str
    why: str
    forced: bool = False


# ---- pure helpers (unit-tested without any LLM) -----------------------------------------------

NUMBER_RE = re.compile(r"[$€£]?\d[\d,.]*[kKmM%]?")


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _tokens(text: str) -> set[str]:
    return set(_key(text).split())


def overlaps(a: str, b: str) -> bool:
    """True if one name's words are all contained in the other's ('pricing' vs 'pricing and plans')."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def preserve_constraints(original: str, rewritten: str) -> tuple[str, list[str]]:
    """Every number, price or percentage in the user's question must survive the rewrite. Any that was lost is
    appended as an explicit constraint. Returns (question to use, the numbers that had been lost)."""
    wanted = [m.group().rstrip(".,") for m in NUMBER_RE.finditer(original)]
    lost = [n for n in dict.fromkeys(wanted) if n and not re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", rewritten, re.I)]
    if not lost:
        return rewritten, []
    return f"{rewritten.rstrip()} (Constraints from the user: {', '.join(lost)})", lost


def tidy_dimensions(items: list[DimensionChoice], forced: list[str], settings) -> list[ChosenDimension]:
    """The user's own dimensions come first, then the model's in the order it gave them (most important first).
    Overlapping names are dropped and the list is trimmed to `max_dimensions`."""
    chosen: list[ChosenDimension] = []

    def add(d: ChosenDimension) -> None:
        if d.name and not any(overlaps(d.name, x.name) for x in chosen):
            chosen.append(d)

    for name in forced:
        match = next((c for c in items if overlaps(c.name, name)), None)
        add(ChosenDimension(name, match.why.strip() if match else "requested by the user", forced=True))
    limit = max(settings.max_dimensions, len(chosen))
    for c in items:
        if len(chosen) >= limit:
            break
        add(ChosenDimension(c.name.strip(), c.why.strip()))
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

    # Call 1: rewrite the question
    qa = deps.llm.structured(
        REWRITE_SYSTEM,
        f"Research question: {question}\n\nRewrite and analyze it. Return:\n"
        "- rewritten_question: one clear, self-contained question. Keep EVERY number, budget, region, team size and "
        "deadline from the original, word for word.\n"
        "- entities: the companies/products/segments to compare (1-6 short names). Use the ones named in the "
        "question; if it asks for 'alternatives' or 'the best', name the most relevant real ones.\n"
        "- category: what kind of thing is being compared\n"
        "- audience: who is making the decision. If the question does not state or clearly imply it, leave this "
        "empty instead of guessing.\n"
        "- decision: what they are trying to decide or achieve (empty if unclear)\n"
        "- is_app: true only if the things being compared are mobile or consumer apps that people install from an app "
        "store (for example Duolingo vs Babbel); false for web software, platforms, companies or services\n\n" + REWRITE_EXAMPLE,
        QueryRewrite,
    )
    rewritten, lost = preserve_constraints(question, qa.rewritten_question.strip() or question)
    if lost:
        trace.append(f"planner: the rewrite had dropped {lost}; added them back as constraints")
    entities = [e.strip() for e in qa.entities if e.strip()][:6]
    audience = qa.audience.strip()
    decision = qa.decision.strip()

    # Call 2: dimensions
    forced_note = f"\nThe user REQUIRES these dimensions, include them: {forced}\n" if forced else ""
    dims_out = deps.llm.structured(
        DIMENSIONS_SYSTEM,
        f"Question: {rewritten}\nAudience: {audience or '(not stated)'}\nDecision: {decision or '(not stated)'}\n"
        f"Category: {qa.category}\nEntities: {entities}\n{forced_note}\n"
        f"List {s.min_dimensions}-{s.max_dimensions} comparison dimensions, most important first. A good dimension:\n"
        "- would change the decision for THIS audience\n"
        "- is one where the entities actually differ\n"
        "- can be answered from public web pages (not private data)\n"
        "- has a short name (1-3 words) that applies to every entity\n"
        "Include criteria this audience would care about even if the question never mentioned them. "
        "No filler like 'overview' or 'general'.\n"
        "For each give `name` and `why` (one sentence about this audience's decision).",
        DimensionList,
    )
    chosen = tidy_dimensions(dims_out.dimensions, forced, s)
    if not chosen:
        trace.append("planner: no usable dimensions proposed; falling back to generic ones")
        chosen = [ChosenDimension(n, "generic fallback") for n in ("pricing", "features", "user experience")]
    dims = [d.name for d in chosen]
    trace.append("planner: dimensions = " + ", ".join(f"{d.name}{' (user)' if d.forced else ''}" for d in chosen))
    if len(chosen) < s.min_dimensions:
        trace.append(f"planner: only {len(chosen)} dimensions available (wanted {s.min_dimensions})")

    # Call 3: sub-questions, then a code check that every dimension is covered
    dim_lines = "\n".join(f"- {d.name}: {d.why}" for d in chosen)
    items = deps.llm.structured(
        SUBQ_SYSTEM,
        f"Question: {rewritten}\nAudience: {audience or '(not stated)'}\nEntities: {entities}\n\n"
        f"Dimensions:\n{dim_lines}\n\n"
        "For EVERY dimension write 1-2 sub-questions answerable by web search. Each must be about exactly one "
        "dimension, name the entities it covers, and be specific. Return items with `dimension` (exactly as listed) "
        "and `question`.",
        SubQuestions,
    ).items
    kept, missing = organize_sub_questions(items, dims, s)
    for dim in missing:
        trace.append(f"planner: used a template sub-question for '{dim}'")
        kept.append(SubQuestion(dimension=dim, question=fallback_question(dim, entities, audience)))
    kept.sort(key=lambda sq: dims.index(sq.dimension))

    use_app = app_enabled(s, qa.is_app)  # App Store facts get their own report section (not a planned dimension)
    tasks = [Task(id=f"t{i + 1}", sub_question=sq.question, dimension=sq.dimension) for i, sq in enumerate(kept)]
    plan = Plan(
        rewritten_question=rewritten,
        audience=audience,
        decision=decision,
        is_app=qa.is_app,
        entities=entities,
        dimensions=dims,
        dimension_notes={d.name: d.why for d in chosen},
        sub_questions=[t.sub_question for t in tasks],
        outline=["Summary", *dims, *([APP_DIMENSION] if use_app else []), "Limitations"],
    )
    trace.append(f"planner: {len(tasks)} sub-questions across {len(dims)} dimensions; entities={entities}")
    return {"plan": plan, "tasks": tasks, "trace": trace}