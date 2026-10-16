"""Analyst: three jobs.

1. Comparison table from verified facts only (built in code: nothing can be invented).
2. Map each verified fact to the user evidence about it, and flag user concerns nobody addressed.
   The LLM judges how a fact and a theme relate; code decides the status from those judgements.
3. Cited insights plus a scoped read of public opinion, written by the LLM from numbers computed in code.
"""
from collections import defaultdict

from pydantic import BaseModel, Field

from app.llm import BudgetExceeded
from app.state import ClaimStatus, Link, Table, build_refs

ANALYST_SYSTEM = (
    "You are the Analyst agent. You compare entities using ONLY the numbered verified facts (CLAIM) and the numbered "
    "community opinion themes (THEME) you are given. Every statement must cite the numbers it relies on, like [3] or "
    "[3][5]. Facts and opinions are different: state facts plainly; attribute opinions ('users report...') and mention "
    "how many threads. Never state a THEME as a fact about the product."
)
LINK_SYSTEM = (
    "You compare verified facts about a product with what community members say about it. Judge only from the text."
)

STATUS_LABEL = {
    "confirmed": "Confirmed by users",
    "disputed": "Disputed by users",
    "qualified": "Qualified by users",
    "no_user_evidence": "No user evidence",
}
USER_RELATIONS = ("supports", "contradicts", "qualifies")
ALL_RELATIONS = (*USER_RELATIONS, "unrelated")


class ClaimThemeLink(BaseModel):
    claim: int  # the [n] of a CLAIM line
    theme: int  # the [m] of a THEME line
    relation: str = "unrelated"
    note: str = ""


class LinkSet(BaseModel):
    links: list[ClaimThemeLink] = Field(default_factory=list)


class AnalysisOut(BaseModel):
    insights: list[str] = Field(default_factory=list)
    public_opinion: list[str] = Field(default_factory=list)


def fact_status_from_relations(relations: list[str]) -> str:
    """A fact users contradict is 'disputed'; one they complicate is 'qualified'; one they back up is 'confirmed'."""
    if "contradicts" in relations:
        return "disputed"
    if "qualifies" in relations:
        return "qualified"
    if "supports" in relations:
        return "confirmed"
    return "no_user_evidence"


def link_evidence(claim_refs, theme_refs, deps, trace: list[str]) -> list[Link]:
    """claim_refs / theme_refs: {citation number: Claim / Theme}. One LLM call per dimension that has both."""
    links: list[Link] = []
    shared = {c.dimension for c in claim_refs.values()} & {t.dimension for t in theme_refs.values()}
    for dim in sorted(shared):
        claims = {n: c for n, c in claim_refs.items() if c.dimension == dim}
        themes = {n: t for n, t in theme_refs.items() if t.dimension == dim}
        if not any(c.entity == t.entity for c in claims.values() for t in themes.values()):
            continue
        claim_lines = "\n".join(f"CLAIM [{n}] ({c.entity}): {c.text}" for n, c in claims.items())
        theme_lines = "\n".join(
            f"THEME [{n}] ({t.entity}, {t.n_threads} threads, {t.stance}): {t.label}" for n, t in themes.items()
        )
        try:
            out = deps.llm.structured(
                LINK_SYSTEM,
                f"Dimension: {dim}\n\n{claim_lines}\n\n{theme_lines}\n\n"
                "For each CLAIM and THEME about the SAME entity that are related, return a link with relation:\n"
                "- supports: users' experience matches the fact\n"
                "- contradicts: users report the opposite\n"
                "- qualifies: users add a caveat or condition the fact leaves out\n"
                "- unrelated: skip these (do not return them)\n"
                "Use the numbers in the brackets for `claim` and `theme`. Add a short `note`.",
                LinkSet,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            trace.append(f"analyst: linking failed for '{dim}' ({exc})")
            continue
        for item in out.links:
            claim, theme = claims.get(item.claim), themes.get(item.theme)
            relation = item.relation.strip().lower()
            if claim is None or theme is None or claim.entity != theme.entity or relation not in ALL_RELATIONS:
                continue  # ignore links the model invented or that cross entities
            links.append(Link(claim_id=claim.id, theme_id=theme.id, relation=relation, note=item.note.strip()))
    return links


def build_evidence_table(claim_refs, theme_num: dict[str, int], themes_by_id, links, fact_status) -> Table:
    by_claim: dict[str, list[Link]] = defaultdict(list)
    for link in links:
        if link.relation in USER_RELATIONS:
            by_claim[link.claim_id].append(link)
    rows = []
    for n, claim in claim_refs.items():
        evidence = "; ".join(
            f"{themes_by_id[l.theme_id].label} [{theme_num[l.theme_id]}] "
            f"({themes_by_id[l.theme_id].n_threads} threads, {l.relation})"
            for l in by_claim.get(claim.id, [])
        ) or "none found"
        rows.append([claim.entity, f"{claim.text} [{n}]", evidence, STATUS_LABEL[fact_status[claim.id]]])
    return Table(columns=["Entity", "Fact", "User evidence", "Status"], rows=rows)


def run_analyst(state, deps) -> dict:
    s = deps.settings
    plan = state["plan"]
    claims = [c for c in state.get("claims", []) if c.status == ClaimStatus.VERIFIED]
    themes = list(state.get("themes", [])) if s.enable_community else []
    summaries = state.get("opinion_summaries", []) if s.enable_community else []
    refs = build_refs(claims, themes)
    by_claim = {c.id: c for c in claims}
    by_theme = {t.id: t for t in themes}
    claim_refs = {n: by_claim[i] for n, i in refs.items() if i in by_claim}
    theme_refs = {n: by_theme[i] for n, i in refs.items() if i in by_theme}
    theme_num = {t.id: n for n, t in theme_refs.items()}
    trace: list[str] = []

    # 1. comparison table from verified facts (code)
    entities = list(plan.entities) or ["Overall"]
    dimensions = list(plan.dimensions) or ["findings"]
    if any(c.entity == "other" for c in claims):
        entities.append("other")
    if any(c.dimension == "other" for c in claims):
        dimensions.append("other")
    cells: dict[tuple[str, str], list[str]] = {}
    for n, c in claim_refs.items():
        ent = c.entity if c.entity in entities else entities[0]
        dim = c.dimension if c.dimension in dimensions else dimensions[0]
        cells.setdefault((ent, dim), []).append(f"{c.text} [{n}]")
    rows, gaps = [], []
    for ent in entities:
        row = [ent]
        for dim in dimensions:
            items = cells.get((ent, dim), [])
            if items:
                row.append("; ".join(items[:3]))
            else:
                row.append("Not found")
                gaps.append(f"No verified information on {dim} for {ent}.")
        rows.append(row)
    table = Table(columns=["Entity", *dimensions], rows=rows)

    # 2. facts <-> user evidence
    links: list[Link] = []
    fact_status: dict[str, str] = {}
    unaddressed: list[str] = []
    evidence_table = None
    if s.enable_community:
        if claim_refs and theme_refs:
            links = link_evidence(claim_refs, theme_refs, deps, trace)
        relations: dict[str, list[str]] = defaultdict(list)
        for link in links:
            relations[link.claim_id].append(link.relation)
        fact_status = {c.id: fact_status_from_relations(relations.get(c.id, [])) for c in claims}
        linked_themes = {l.theme_id for l in links if l.relation in USER_RELATIONS}
        unaddressed = [t.id for t in themes if t.stance in ("negative", "mixed") and t.id not in linked_themes]
        if claim_refs:
            evidence_table = build_evidence_table(claim_refs, theme_num, by_theme, links, fact_status)
        fact_dims, theme_dims = {c.dimension for c in claims}, {t.dimension for t in themes}
        for dim in dimensions:
            if dim in fact_dims and dim not in theme_dims:
                gaps.append(f"No user evidence on {dim} (facts only).")
            if dim in theme_dims and dim not in fact_dims:
                gaps.append(f"Users discuss {dim}, but no verified facts were found.")
        for summary in summaries:
            if summary.sentiment.confidence == "low":
                gaps.append(f"Public sentiment on {summary.entity} rests on only {summary.sentiment.n_threads} thread(s).")
        if not themes:
            gaps.append("No recurring user themes passed the minimum-threads bar.")

    # 3. insights and a scoped read of public opinion
    insights: list[str] = []
    public_opinion: list[str] = []
    if refs:
        fact_lines = "\n".join(f"CLAIM [{n}] ({c.entity} / {c.dimension}): {c.text}" for n, c in claim_refs.items())
        theme_lines = "\n".join(
            f"THEME [{n}] ({t.entity} / {t.dimension}, {t.n_threads} threads, {t.stance}, {t.confidence} confidence): {t.label}"
            for n, t in theme_refs.items()
        ) or "(no community themes)"
        status_lines = "\n".join(f"CLAIM [{n}]: {STATUS_LABEL[fact_status[c.id]]}" for n, c in claim_refs.items()) if fact_status else "(not computed)"
        sentiment_lines = "\n".join(
            f"SENTIMENT {x.entity}: {x.sentiment.label} (score {x.sentiment.score}, {x.sentiment.n_threads} threads, "
            f"{x.sentiment.confidence} confidence); praised: {[theme_num[i] for i in x.top_praise if i in theme_num]}; "
            f"complaints: {[theme_num[i] for i in x.top_complaints if i in theme_num]}"
            for x in summaries
        ) or "(none)"
        unaddressed_lines = ", ".join(f"THEME [{theme_num[i]}]" for i in unaddressed) or "(none)"
        try:
            out = deps.llm.structured(
                ANALYST_SYSTEM,
                f"Research question: {plan.rewritten_question or state['question']}\n"
                f"Audience: {plan.audience}; decision: {plan.decision}\n\n"
                f"VERIFIED FACTS:\n{fact_lines}\n\nCOMMUNITY OPINION THEMES:\n{theme_lines}\n\n"
                f"HOW FACTS HOLD UP AGAINST USER EVIDENCE:\n{status_lines}\n\n"
                f"PUBLIC SENTIMENT (computed from sampled threads):\n{sentiment_lines}\n\n"
                f"USER CONCERNS NO FACT ADDRESSES: {unaddressed_lines}\n\n"
                "Return:\n- insights: 3-5 comparative insights (similarities, differences, who each option suits), "
                "each citing numbers. Where a fact and user evidence disagree, say so.\n"
                "- public_opinion: 2-4 sentences on what the sampled community says overall, scoped to the sample "
                "(say 'in the threads we sampled'), citing THEME numbers. Return an empty list if there are no themes.",
                AnalysisOut,
            )
            insights = [i.strip() for i in out.insights if i.strip()][:6]
            public_opinion = [i.strip() for i in out.public_opinion if i.strip()][:4] if themes else []
        except BudgetExceeded:
            raise
        except Exception as exc:
            trace.append(f"analyst: insights unavailable ({exc})")

    trace.append(
        f"analyst: {len(claim_refs)} verified facts, {len(theme_refs)} themes, {len(links)} links, "
        f"{len(unaddressed)} unaddressed concerns, {len(gaps)} gaps"
    )
    return {
        "refs": refs, "table": table, "evidence_table": evidence_table, "links": links, "fact_status": fact_status,
        "unaddressed": unaddressed, "insights": insights, "public_opinion": public_opinion, "gaps": gaps,
        "trace": trace,
    }
