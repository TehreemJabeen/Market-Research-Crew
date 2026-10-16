"""Deterministic grounding checks. No LLM involved: code checks the AI's work."""
from difflib import SequenceMatcher

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
MIN_QUOTE_CHARS = 20


def normalize(text: str) -> str:
    text = text.translate(_QUOTE_MAP).replace("​", "")
    return " ".join(text.lower().split())


def quote_score(quote: str, source_text: str) -> float:
    """1.0 if the quote appears verbatim (after whitespace/quote normalisation);
    otherwise the best fuzzy ratio over word windows of similar length."""
    nq, nt = normalize(quote), normalize(source_text)
    if len(nq) < MIN_QUOTE_CHARS or not nt:
        return 0.0
    if nq in nt:
        return 1.0
    words = nt.split()
    qlen = len(nq.split())
    best = 0.0
    for size in {max(1, qlen - 1), qlen, qlen + 1}:
        for i in range(0, len(words) - size + 1):
            window = " ".join(words[i : i + size])
            sm = SequenceMatcher(None, nq, window, autojunk=False)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            best = max(best, sm.ratio())
    return best


def citation_accuracy(citations: list[dict], sources: dict[str, str], threshold: float = 0.9) -> float:
    """Share of citations whose quote is actually found in the cited source."""
    if not citations:
        return 0.0
    ok = sum(1 for c in citations if quote_score(c.get("quote", ""), sources.get(c.get("url", ""), "")) >= threshold)
    return ok / len(citations)
