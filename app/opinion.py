"""Pure helpers for the opinions track: no LLM, no network.

Everything the report says about "how many" or "how positive" is computed here, never by a model:
filtering, comparative detection, thread-weighted sentiment, theme statistics, the sentiment table.
"""
import re
import time
from collections import defaultdict

from app.state import Example, OpinionSummary, SentimentStat, Snippet, Table, Theme, normalize_label
from app.tools.verify import normalize

MIN_POST_CHARS = 40

PROMO_RE = re.compile(
    r"(referral|affiliate|promo code|discount code|use my (?:code|link)|sign up (?:here|now)|check out my|"
    r"i(?:'m| am) the (?:founder|ceo|creator|author) of|full disclosure|we(?:'re| are) hiring|dm me)",
    re.I,
)
URL_RE = re.compile(r"https?://\S+")
COMPARE_CUE_RE = re.compile(
    r"\b(?:switch(?:ed|ing)? (?:from|to)|moved (?:from|to)|migrat\w* (?:from|to)|instead of|ditched|replaced|"
    r"better than|worse than|versus|compared (?:to|with))\b|\bvs\.?\b",
    re.I,
)


def looks_promotional(text: str) -> bool:
    return bool(PROMO_RE.search(text)) or len(URL_RE.findall(text)) >= 3


def filter_posts(posts, settings, now: float | None = None):
    """Drop too-short, promotional, stale and duplicate posts; cap the rest (best-scored first).
    Returns (kept posts, {reason: count})."""
    now = now if now is not None else time.time()
    max_age = settings.community_max_age_days * 86400
    stats: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    kept = []
    for post in posts:
        text = " ".join(post.text.split())
        if len(text) < MIN_POST_CHARS:
            stats["too_short"] += 1
        elif looks_promotional(text):
            stats["promotional"] += 1
        elif post.created and now - post.created > max_age:
            stats["too_old"] += 1
        else:
            key = normalize(text)[:120]
            if key in seen:
                stats["duplicate"] += 1
            else:
                seen.add(key)
                kept.append(post)
    kept.sort(key=lambda p: -(p.score or 0))  # stable: ties keep their order
    return kept[: settings.max_posts_per_task], dict(stats)


def is_comparative(text: str, entities: list[str]) -> bool:
    """True if the text names two or more of the entities, or uses a switching/comparison phrase. Such snippets
    are routed to the LLM scorer, because a text-only model can't tell who the comparison favours."""
    low = text.casefold()
    mentioned = [e for e in entities if e and e.casefold() in low]
    return len(mentioned) >= 2 or bool(COMPARE_CUE_RE.search(text))


def confidence_for(n_threads: int) -> str:
    return "high" if n_threads >= 5 else "medium" if n_threads >= 3 else "low"


def stance_from_counts(pos: int, neg: int, neu: int) -> str:
    n = pos + neg + neu
    if n == 0:
        return "neutral"
    net = (pos - neg) / n
    if net >= 0.34:
        return "positive"
    if net <= -0.34:
        return "negative"
    return "mixed" if pos and neg else "neutral"


def thread_scores(snippets: list[Snippet]) -> list[float]:
    """One score per discussion thread (the mean of its snippets), so one long argument counts once."""
    by_thread: dict[str, list[float]] = defaultdict(list)
    for s in snippets:
        by_thread[s.thread_id].append(s.polarity_score)
    return [sum(v) / len(v) for v in by_thread.values()]


def sentiment_stat(snippets: list[Snippet]) -> SentimentStat:
    scores = thread_scores(snippets)
    if not scores:
        return SentimentStat(label="no data", score=0.0, n_threads=0, confidence="low")
    mean = sum(scores) / len(scores)
    has_pos = any(x >= 0.2 for x in scores)
    has_neg = any(x <= -0.2 for x in scores)
    label = "positive" if mean >= 0.2 else "negative" if mean <= -0.2 else ("mixed" if has_pos and has_neg else "neutral")
    return SentimentStat(label=label, score=round(mean, 2), n_threads=len(scores), confidence=confidence_for(len(scores)))


def build_themes(groups, snippets: list[Snippet], entities: list[str], settings, start: int = 1):
    """groups: [(entity, label, [snippet ids])] as proposed by the LLM. The LLM only groups: stance, counts and
    confidence are computed here. Themes seen in fewer than `theme_min_threads` threads are dropped as anecdotal.
    Returns (themes, number dropped as anecdotal)."""
    by_id = {s.id: s for s in snippets}
    used: set[str] = set()
    themes: list[Theme] = []
    anecdotal = 0
    number = start
    for entity_raw, label, ids in groups:
        entity = normalize_label(entity_raw, entities)
        members = [by_id[i] for i in ids if i in by_id and i not in used and by_id[i].entity == entity]
        if entity == "other" or not members or not label.strip():
            continue
        used.update(m.id for m in members)
        threads = {m.thread_id for m in members}
        if len(threads) < settings.theme_min_threads:
            anecdotal += 1
            continue
        pos = sum(m.polarity == "positive" for m in members)
        neg = sum(m.polarity == "negative" for m in members)
        neu = len(members) - pos - neg
        examples, seen_threads = [], set()
        for m in sorted(members, key=lambda m: -abs(m.polarity_score)):
            if m.thread_id not in seen_threads:
                seen_threads.add(m.thread_id)
                examples.append(Example(url=m.url, quote=m.quote, source=m.source))
            if len(examples) == 2:
                break
        themes.append(
            Theme(
                id=f"th{number}", entity=entity, dimension=members[0].dimension, label=label.strip(),
                snippet_ids=[m.id for m in members], n_snippets=len(members), n_threads=len(threads),
                n_positive=pos, n_negative=neg, n_neutral=neu, stance=stance_from_counts(pos, neg, neu),
                confidence=confidence_for(len(threads)), examples=examples,
            )
        )
        number += 1
    return themes, anecdotal


def summarize_entities(snippets: list[Snippet], themes: list[Theme], entities: list[str], dimensions: list[str]) -> list[OpinionSummary]:
    out = []
    for entity in entities:
        mine = [s for s in snippets if s.entity == entity]
        if not mine:
            continue
        by_dim = {d: sentiment_stat([s for s in mine if s.dimension == d]) for d in dimensions if any(s.dimension == d for s in mine)}
        its = [t for t in themes if t.entity == entity]
        praise = [t.id for t in sorted((t for t in its if t.stance == "positive"), key=lambda t: -t.n_threads)][:3]
        complaints = [t.id for t in sorted((t for t in its if t.stance == "negative"), key=lambda t: -t.n_threads)][:3]
        out.append(
            OpinionSummary(
                entity=entity, sentiment=sentiment_stat(mine), n_snippets=len(mine), by_dimension=by_dim,
                top_praise=praise, top_complaints=complaints,
            )
        )
    return out


def _cell(stat: SentimentStat | None) -> str:
    if stat is None or stat.n_threads == 0:
        return "no data"
    plural = "" if stat.n_threads == 1 else "s"
    return f"{stat.label} ({stat.n_threads} thread{plural})"


def sentiment_table(summaries: list[OpinionSummary], entities: list[str], dimensions: list[str]) -> Table:
    by_entity = {s.entity: s for s in summaries}
    rows = []
    for entity in entities:
        summary = by_entity.get(entity)
        row = [entity] + [_cell(summary.by_dimension.get(d) if summary else None) for d in dimensions]
        row.append(_cell(summary.sentiment if summary else None))
        rows.append(row)
    return Table(columns=["Entity", *dimensions, "Overall"], rows=rows)
