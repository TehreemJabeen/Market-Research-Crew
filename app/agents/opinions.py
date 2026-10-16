"""Opinion Processor: where the opinions track joins up.

1. polarity   hybrid: the configured scorer (a local Hugging Face model by default) scores plain snippets; the LLM
              scores comparative ones, where it matters who a comparison favours
2. themes     one LLM call per dimension that only GROUPS snippets into recurring points
3. aggregate  stance, counts, confidence and per-entity sentiment are computed in code
"""
from collections import defaultdict

from pydantic import BaseModel, Field

from app.llm import BudgetExceeded, LLMError
from app.opinion import build_themes, sentiment_table, summarize_entities
from app.tools.sentiment import NEUTRAL, LLMScorer, ScoreItem

THEME_SYSTEM = (
    "You are the Opinion Processor. You group short community comments into recurring points. "
    "You never judge sentiment and never invent ids."
)


class ThemeGroup(BaseModel):
    entity: str
    label: str
    snippet_ids: list[str] = Field(default_factory=list)


class ThemeGroups(BaseModel):
    themes: list[ThemeGroup] = Field(default_factory=list)


def _score(scorer, items: list[ScoreItem], deps, trace: list[str]):
    """Score with `scorer`; on failure fall back to the LLM scorer, then to neutral. Budget errors always stop the run."""
    try:
        return scorer.score(items), scorer.name
    except BudgetExceeded:
        raise
    except Exception as exc:
        trace.append(f"opinions: {scorer.name} scorer failed ({exc})")
    if not isinstance(scorer, LLMScorer):
        trace.append("opinions: falling back to the LLM scorer")
        try:
            return LLMScorer(deps.llm).score(items), "llm"
        except BudgetExceeded:
            raise
        except Exception as exc:
            trace.append(f"opinions: LLM scorer failed too ({exc}); snippets left neutral")
    return [NEUTRAL] * len(items), "unscored"


def run_opinions(state, deps) -> dict:
    s = deps.settings
    plan = state["plan"]
    entities, dimensions = plan.entities, plan.dimensions
    snippets = list(state.get("snippets", []))
    empty_table = sentiment_table([], entities, dimensions)
    if not snippets:
        return {
            "themes": [], "opinion_summaries": [], "sentiment_table": empty_table, "opinions_ready": True,
            "trace": ["opinions: no community evidence was collected"],
        }

    trace: list[str] = []
    scorer = deps.scorer or LLMScorer(deps.llm)
    note = getattr(scorer, "note", "")
    if note:
        trace.append(f"opinions: {note}")

    # 1. polarity
    plain = [sn for sn in snippets if not sn.comparative]
    comparative = [sn for sn in snippets if sn.comparative]
    llm_scorer = scorer if isinstance(scorer, LLMScorer) else LLMScorer(deps.llm)
    scored: list = []
    names: set[str] = set()
    for group, group_scorer in ((plain, scorer), (comparative, llm_scorer)):
        if not group:
            continue
        results, name = _score(group_scorer, [ScoreItem(sn.entity, sn.dimension, sn.quote) for sn in group], deps, trace)
        names.add(name)
        scored += [
            sn.model_copy(update={"polarity": p.label, "polarity_score": p.score, "scorer": name})
            for sn, p in zip(group, results)
        ]

    # 2. themes, one call per dimension
    by_dim: dict[str, list] = defaultdict(list)
    for sn in scored:
        by_dim[sn.dimension].append(sn)
    themes, anecdotal, number = [], 0, 1
    for dim in [d for d in dimensions if d in by_dim] + [d for d in by_dim if d not in dimensions]:
        group = by_dim[dim][: s.max_snippets_for_themes]
        if len({sn.thread_id for sn in group}) < s.theme_min_threads:
            anecdotal += 1  # not enough distinct discussions in this dimension to form any theme
            continue
        lines = "\n".join(f"[{sn.id}] entity={sn.entity} | {sn.quote}" for sn in group)
        try:
            out = deps.llm.structured(
                THEME_SYSTEM,
                f"Dimension: {dim}\nBelow are short opinion snippets from community discussions, one per line as "
                "[id] entity=... | quote.\nGroup them into themes: recurring points people make about the SAME entity. "
                "For each theme give `entity` (exactly as in the lines), a short neutral `label` (max 8 words, stating "
                "the point, e.g. 'slows down on large databases') and the `snippet_ids` that make that point. "
                "A theme is about one entity; a snippet belongs to at most one theme; skip snippets that do not fit a "
                "recurring point.\n\n" + lines,
                ThemeGroups,
            )
        except BudgetExceeded:
            raise
        except LLMError as exc:
            trace.append(f"opinions: theme grouping failed for '{dim}' ({exc})")
            continue
        built, dropped = build_themes([(g.entity, g.label, g.snippet_ids) for g in out.themes], group, entities, s, number)
        themes += built
        anecdotal += dropped
        number += len(built)

    # 3. aggregate (code)
    summaries = summarize_entities(scored, themes, entities, dimensions)
    trace.append(
        f"opinions: {len(scored)} snippets scored by {sorted(names)}; {len(themes)} themes kept, "
        f"{anecdotal} dropped as anecdotal (< {s.theme_min_threads} threads)"
    )
    return {
        "snippets": scored, "themes": themes, "opinion_summaries": summaries,
        "sentiment_table": sentiment_table(summaries, entities, dimensions),
        "opinions_ready": True, "trace": trace,
    }
