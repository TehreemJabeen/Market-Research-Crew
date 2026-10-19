import re

from app.state import table_to_markdown

WRITER_SYSTEM = (
    "You are the Writer agent. You write a clear market-research report in Markdown using ONLY the numbered items "
    "provided. There are two kinds of evidence and you must keep their voices apart:\n"
    "- CLAIM items are verified facts from web pages. State them plainly and cite them, e.g. 'Alpha costs $10 per "
    "user per month [3].'\n"
    "- THEME items are what some community members said (Hacker News, Reddit). They are opinions, not facts. Always "
    "attribute them ('Users report...', 'Commenters on Hacker News and Reddit say...'), mention the thread count, and "
    "cite the THEME number, e.g. 'Users report slowdowns on large boards (4 threads) [7].' Never present a theme as a "
    "fact about the product, and never cite a CLAIM number for an opinion or a THEME number for a fact.\n"
    "Put each citation before the final period. Never cite a number that is not listed. Never add facts from your own "
    "knowledge. If evidence is missing, say so under 'Limitations' instead of guessing."
)


REF_RE = re.compile(r"\[(\d+)\]")


def citations_for(report: str, refs: dict, by_claim: dict, by_theme: dict) -> list[dict]:
    """The sources behind the [n] markers the report really uses (used by the evals and the UI)."""
    used = sorted({int(n) for n in REF_RE.findall(report)} & set(refs))
    out = []
    for n in used:
        rid = refs[n]
        if rid in by_claim:
            c = by_claim[rid]
            out.append({"n": n, "claim_id": c.id, "url": c.source_url, "quote": c.supporting_quote.strip()})
        elif rid in by_theme:
            out += [{"n": n, "claim_id": rid, "url": e.url, "quote": e.quote.strip()} for e in by_theme[rid].examples]
    return out


def run_writer(state, deps) -> dict:
    refs = state.get("refs", {})
    question = state["question"]
    if not refs:
        draft = (
            f"# {question}\n\nNo claims could be verified against their sources and no recurring user themes were "
            "found, so no findings are reported. Try again with a more specific question or different sources."
        )
        return {"report": draft, "final_citations": [], "trace": ["writer: no evidence; wrote a limitations-only report"]}

    by_claim = {c.id: c for c in state.get("claims", [])}
    by_theme = {t.id: t for t in state.get("themes", [])}
    claim_lines = "\n".join(
        f"CLAIM [{n}] ({by_claim[i].entity} / {by_claim[i].dimension}): {by_claim[i].text}"
        for n, i in refs.items() if i in by_claim
    ) or "(none)"
    theme_lines = "\n".join(
        f"THEME [{n}] ({t.entity} / {t.dimension}, {t.n_threads} threads, {t.stance}, {t.confidence} confidence): {t.label}"
        for n, i in refs.items() if (t := by_theme.get(i))
    )
    plan = state["plan"]
    insights = "\n".join(f"- {i}" for i in state.get("insights", [])) or "(none)"
    opinion_notes = "\n".join(f"- {i}" for i in state.get("public_opinion", [])) or "(none)"
    gap_list = list(state.get("gaps", []))
    if any(c.source_type == "appstore" for c in by_claim.values()):
        gap_list.append("App Store data has no install counts or in-app purchase details (Apple does not publish them).")
    gaps = "\n".join(f"- {g}" for g in gap_list) or "(none)"
    unaddressed = [by_theme[i].label + f" ({by_theme[i].entity})" for i in state.get("unaddressed", []) if i in by_theme]

    sections = [
        f"Research question: {plan.rewritten_question or question}",
        f"Audience: {plan.audience}; decision: {plan.decision}",
        f"Report outline (follow it): {plan.outline}",
        f"VERIFIED FACTS (the only facts you may state; cite as [n]):\n{claim_lines}",
    ]
    if theme_lines:
        sections.append(f"COMMUNITY OPINION THEMES (attribute them; cite as [m]):\n{theme_lines}")
    sections.append(f"COMPARISON TABLE (include verbatim under a '## Comparison' heading):\n{table_to_markdown(state['table'])}")
    if theme_lines and state.get("sentiment_table"):
        sections.append(
            "PUBLIC SENTIMENT TABLE (include verbatim under a '## What users say' heading, followed by a short "
            f"scoped summary; say it covers only the sampled threads):\n{table_to_markdown(state['sentiment_table'])}"
        )
    if theme_lines and state.get("evidence_table"):
        sections.append(
            "CLAIMS VS USER EVIDENCE TABLE (include verbatim under a '## Claims vs user evidence' heading, then "
            f"discuss disputed or qualified facts):\n{table_to_markdown(state['evidence_table'])}"
        )
    if unaddressed:
        sections.append("USER CONCERNS NO FACT ADDRESSES (mention under '## What users say'): " + "; ".join(unaddressed))
    sections += [
        f"KEY INSIGHTS (already cited, reuse them):\n{insights}",
        f"PUBLIC OPINION NOTES (already cited, reuse them):\n{opinion_notes}",
        f"KNOWN GAPS (mention in Limitations, and note the community sample is small and self-selected):\n{gaps}",
    ]
    report = deps.llm.complete(WRITER_SYSTEM, "\n\n".join(sections))
    return {
        "report": report,
        "final_citations": citations_for(report, refs, by_claim, by_theme),
        "trace": [f"writer: drafted {len(report.split())} words"],
    }