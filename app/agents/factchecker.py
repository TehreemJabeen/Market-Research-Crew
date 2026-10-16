from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from app.state import ClaimStatus
from app.tools.verify import quote_score

FACTCHECK_SYSTEM = (
    "You are the Fact-Checker agent. You are skeptical. For each claim you judge ONLY whether its quote "
    "supports it, and whether it contradicts another claim in the list. Be strict."
)
BATCH = 12


class Verdict(BaseModel):
    claim_id: str
    verdict: Literal["supported", "unsupported", "conflicting"]
    note: str = ""


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


def run_factchecker(state, deps) -> dict:
    s = deps.settings
    claims = state.get("claims", [])
    sources = state.get("sources", {})
    pending = [c for c in claims if c.status == ClaimStatus.UNVERIFIED]
    updated = {}
    trace = []

    if not s.enable_factchecker:  # ablation: everything passes
        for c in pending:
            updated[c.id] = c.model_copy(update={"status": ClaimStatus.VERIFIED, "check_note": "fact-check disabled"})
        trace.append(f"factchecker: DISABLED, passed {len(pending)} claims unchecked")
    else:
        # Pass 1: deterministic. Does the quote exist in the source we actually fetched?
        survivors = []
        for c in pending:
            score = quote_score(c.supporting_quote, sources.get(c.source_url, ""))
            if score < s.fuzzy_threshold:
                updated[c.id] = c.model_copy(
                    update={"status": ClaimStatus.REJECTED, "check_note": f"quote not found in source (score {score:.2f})"}
                )
            else:
                survivors.append(c)

        # Pass 2: LLM. Does the quote support the claim? Contradictions? Grouped by entity.
        groups = defaultdict(list)
        for c in survivors:
            groups[c.entity.casefold()].append(c)
        for group in groups.values():
            for i in range(0, len(group), BATCH):
                chunk = group[i : i + BATCH]
                listing = "\n".join(
                    f"id={c.id} | entity={c.entity} | claim: {c.text} | quote: {c.supporting_quote}" for c in chunk
                )
                try:
                    out = deps.llm.structured(
                        FACTCHECK_SYSTEM,
                        "For every claim return a verdict: 'supported' (the quote clearly supports the claim), "
                        "'unsupported' (quote is vague, off-topic, or says something different), or 'conflicting' "
                        "(it contradicts another claim in this list). Include one verdict per id.\n\n" + listing,
                        Verdicts,
                    )
                    vmap = {v.claim_id: v for v in out.verdicts}
                except Exception as exc:
                    vmap = {}
                    trace.append(f"factchecker: semantic check unavailable ({exc}); keeping quote-verified claims")
                    for c in chunk:
                        updated[c.id] = c.model_copy(
                            update={"status": ClaimStatus.VERIFIED, "check_note": "quote verified; semantic check unavailable"}
                        )
                    continue
                for c in chunk:
                    v = vmap.get(c.id)
                    if v is None:
                        updated[c.id] = c.model_copy(update={"status": ClaimStatus.REJECTED, "check_note": "no verdict returned"})
                    elif v.verdict == "supported":
                        updated[c.id] = c.model_copy(update={"status": ClaimStatus.VERIFIED, "check_note": v.note})
                    elif v.verdict == "conflicting":
                        updated[c.id] = c.model_copy(update={"status": ClaimStatus.CONFLICTING, "check_note": v.note})
                    else:
                        updated[c.id] = c.model_copy(update={"status": ClaimStatus.REJECTED, "check_note": v.note or "unsupported"})
        counts = defaultdict(int)
        for c in updated.values():
            counts[c.status.value] += 1
        trace.append(f"factchecker: {dict(counts)}")

    # Decide which tasks need another research round (the re-research loop).
    merged = {c.id: updated.get(c.id, c) for c in claims}
    loop = state.get("loop_count", 0)
    task_updates, retry = [], False
    for t in state.get("tasks", []):
        mine = [c for c in merged.values() if c.task_id == t.id]
        verified = [c for c in mine if c.status == ClaimStatus.VERIFIED]
        if len(verified) < s.min_verified_per_task and loop < s.max_loops:
            bad = [f"'{c.text[:80]}' ({c.check_note})" for c in mine if c.status != ClaimStatus.VERIFIED][:5]
            feedback = "rejected claims: " + "; ".join(bad) if bad else "no usable claims were found"
            task_updates.append(t.model_copy(update={"status": "needs_more", "feedback": feedback}))
            retry = True
    if retry:
        trace.append(f"factchecker: sending {len(task_updates)} task(s) back for re-research (round {loop + 1}/{s.max_loops})")
    return {
        "claims": list(updated.values()),
        "tasks": task_updates,
        "loop_count": loop + 1 if retry else loop,
        "facts_ready": not retry,  # the gate before the Analyst waits for this
        "trace": trace,
    }
