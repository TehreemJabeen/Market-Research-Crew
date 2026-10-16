"""Editor: enforces the citation contract mostly in code, with one LLM repair pass.

Checks: citations must exist, factual sentences must be cited, and an opinion (a sentence citing only community
themes) must say whose opinion it is."""
import re

from app.llm import BudgetExceeded

EDITOR_SYSTEM = (
    "You are the Editor agent. You fix problems in a Markdown report. For each uncited sentence, either add the "
    "correct citation number from the valid list or delete the sentence. For each opinion sentence that cites only "
    "community themes but does not say who holds the opinion, rewrite it to attribute it ('Users report...', "
    "'Commenters on Hacker News and Reddit say...'). Do not add new facts. Return the full revised report inside "
    "<report></report> tags."
)

REF_RE = re.compile(r"\[(\d+)\]")
GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
BULLET_RE = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)?(.*)$")
ATTRIB_RE = re.compile(
    r"\b(?:users?|commenters?|community|reddit|hacker news|hn|threads?|reviewers?|people|posters?|report(?:s|ed)?)\b", re.I
)


def normalize_ref_groups(text: str) -> str:
    return GROUP_RE.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text)


def strip_invalid_refs(text: str, valid: set[int]) -> tuple[str, int]:
    removed = 0

    def repl(m):
        nonlocal removed
        if int(m.group(1)) in valid:
            return m.group(0)
        removed += 1
        return ""

    return REF_RE.sub(repl, text), removed


def _is_content_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith(("#", "|", ">", "```", "---"))


def _sentences(text: str):
    for line in text.splitlines():
        if _is_content_line(line):
            yield from SENT_SPLIT.split(BULLET_RE.match(line).group(2))


def find_uncited(text: str) -> list[str]:
    return [s.strip() for s in _sentences(text) if len(s.strip()) > 40 and not REF_RE.search(s)]


def find_unattributed_opinions(text: str, theme_nums: set[int]) -> list[str]:
    """Sentences whose citations are all community themes but that never say whose opinion it is."""
    out = []
    for sent in _sentences(text):
        refs = {int(n) for n in REF_RE.findall(sent)}
        if refs and refs <= theme_nums and not ATTRIB_RE.search(sent):
            out.append(sent.strip())
    return out


def drop_uncited_factual(text: str) -> tuple[str, int]:
    """Remove sentences that look factual (contain a digit) but carry no citation."""
    lines, dropped = [], 0
    for line in text.splitlines():
        if not _is_content_line(line):
            lines.append(line)
            continue
        m = BULLET_RE.match(line)
        prefix, body = m.group(1) or "", m.group(2)
        kept = []
        for sent in SENT_SPLIT.split(body):
            if len(sent.strip()) > 40 and not REF_RE.search(sent) and re.search(r"\d", sent):
                dropped += 1
            else:
                kept.append(sent)
        if kept:
            lines.append(prefix + " ".join(kept))
    return "\n".join(lines), dropped


def renumber(text: str) -> tuple[str, dict[int, int]]:
    order: list[int] = []
    for m in REF_RE.finditer(text):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
    mapping = {old: i + 1 for i, old in enumerate(order)}
    return REF_RE.sub(lambda m: f"[{mapping[int(m.group(1))]}]", text), mapping


def _short(quote: str, limit: int = 200) -> str:
    quote = quote.strip()
    return quote if len(quote) <= limit else quote[:limit] + "…"


def run_editor(state, deps) -> dict:
    draft = state.get("draft", "")
    refs = state.get("refs", {})
    if not refs:
        return {"report": draft, "final_citations": [], "metrics": {"citations": 0}, "trace": ["editor: nothing to check"]}

    by_claim = {c.id: c for c in state.get("claims", [])}
    by_theme = {t.id: t for t in state.get("themes", [])}
    theme_nums = {n for n, rid in refs.items() if rid in by_theme}
    text = normalize_ref_groups(draft)
    text, invalid_removed = strip_invalid_refs(text, set(refs))

    uncited = find_uncited(text)
    unattributed = find_unattributed_opinions(text, theme_nums)
    if uncited or unattributed:
        listing = "\n".join(
            f"[{n}] {by_claim[rid].text}" if rid in by_claim
            else f"[{n}] (community theme, {by_theme[rid].n_threads} threads) {by_theme[rid].label}"
            for n, rid in refs.items()
        )
        try:
            revised = deps.llm.complete(
                EDITOR_SYSTEM,
                "UNCITED SENTENCES:\n" + ("\n".join(f"- {s}" for s in uncited) or "(none)")
                + "\n\nOPINION SENTENCES WITHOUT ATTRIBUTION:\n" + ("\n".join(f"- {s}" for s in unattributed) or "(none)")
                + f"\n\nVALID CITATIONS:\n{listing}\n\n<report>\n{text}\n</report>",
            )
            m = re.search(r"<report>(.*?)</report>", revised, re.S)
            candidate = (m.group(1) if m else revised).strip()
            candidate, extra_invalid = strip_invalid_refs(normalize_ref_groups(candidate), set(refs))
            invalid_removed += extra_invalid
            if candidate:
                text = candidate
        except BudgetExceeded:
            raise
        except Exception:
            pass  # fall through to the deterministic cleanup below

    text, dropped = drop_uncited_factual(text)
    still_uncited = len(find_uncited(text))
    still_unattributed = len(find_unattributed_opinions(text, theme_nums))
    text, mapping = renumber(text)

    sources = ["", "## Sources"]
    citations = []
    themes_cited = 0
    for old, new in sorted(mapping.items(), key=lambda kv: kv[1]):
        rid = refs[old]
        if rid in by_claim:
            c = by_claim[rid]
            sources.append(f'{new}. {c.source_url} — "{_short(c.supporting_quote)}"')
            citations.append({"n": new, "claim_id": c.id, "url": c.source_url, "quote": c.supporting_quote.strip()})
        else:
            t = by_theme[rid]
            themes_cited += 1
            quotes = "; ".join(f'"{_short(e.quote)}" ({e.url})' for e in t.examples)
            sources.append(
                f"{new}. Community theme, {t.n_threads} threads, {t.stance} ({t.confidence} confidence): {quotes}"
            )
            for e in t.examples:
                citations.append({"n": new, "claim_id": t.id, "url": e.url, "quote": e.quote.strip()})
    report = text.rstrip() + "\n" + "\n".join(sources) + "\n"

    metrics = {
        "citations": len(mapping),
        "theme_citations": themes_cited,
        "invalid_refs_removed": invalid_removed,
        "uncited_factual_sentences_dropped": dropped,
        "uncited_sentences_remaining": still_uncited,
        "unattributed_opinion_sentences": still_unattributed,
    }
    return {"report": report, "final_citations": citations, "metrics": metrics, "trace": [f"editor: {metrics}"]}
