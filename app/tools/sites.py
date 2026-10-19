"""Finds each entity's official website, once per run, before the researchers start.

Mostly code: one Tavily search per entity ("Notion official website"), drop sites that only write ABOUT products
(G2, Wikipedia, Reddit...), and keep the domain whose name matches the entity. A model is asked only when no
result matches by name, and it may only choose among domains that the search actually returned.
The user can always fix a wrong or missing domain with `--sites notion=notion.com`.
"""
from pydantic import BaseModel

from app.llm import BudgetExceeded
from app.state import normalize_label
from app.tools.domains import host_of, is_third_party, name_match, registered_domain

PICK_SYSTEM = "You identify which website is a company's or product's own official site. You never guess."
MAX_DOMAINS_PER_ENTITY = 2  # e.g. notion.so and notion.com


class SitePick(BaseModel):
    domain: str = ""


def pick_domains(entity: str, results) -> tuple[list[str], list[str]]:
    """Returns (domains whose name matches the entity, all non-aggregator candidate domains in rank order)."""
    candidates: list[str] = []
    for r in results:
        d = registered_domain(r.url)
        if d and not is_third_party(d) and d not in candidates:
            candidates.append(d)
    best = max((name_match(entity, d) for d in candidates), default=0)
    if best == 0:
        return [], candidates
    exact = [d for d in candidates if name_match(entity, d) == best]
    return exact[: MAX_DOMAINS_PER_ENTITY if best == 2 else 1], candidates


def _clean(domain: str) -> str:
    return registered_domain(host_of(domain))


def run_sites(state, deps) -> dict:
    plan = state["plan"]
    entities = plan.entities
    overrides: dict[str, str] = {}
    trace: list[str] = []
    for key, domain in (state.get("site_overrides") or {}).items():
        entity = normalize_label(key, entities)
        if entity == "other" or not _clean(domain):
            trace.append(f"sites: ignored override '{key}={domain}' (no such entity or bad domain)")
        else:
            overrides[entity] = _clean(domain)

    found: dict[str, list[str]] = {}
    for entity in entities:
        if entity in overrides:
            found[entity] = [overrides[entity]]
            trace.append(f"sites: {entity} -> {overrides[entity]} (set by you)")
            continue
        try:
            results = deps.search(f"{entity} official website", 6, None)
        except Exception as exc:
            trace.append(f"sites: search failed for {entity}: {exc}")
            continue
        domains, candidates = pick_domains(entity, results)
        how = "name match"
        if not domains and candidates:
            how = "model pick"
            try:
                shown = "\n".join(f"- {r.url} | {r.title}" for r in results if registered_domain(r.url) in candidates)
                out = deps.llm.structured(
                    PICK_SYSTEM,
                    f"Which of these results is the official website of '{entity}'? Answer with its domain exactly as "
                    f"shown, or an empty string if none of them is the official site.\n\n{shown}",
                    SitePick,
                )
                pick = _clean(out.domain)
                domains = [pick] if pick in candidates else []
            except BudgetExceeded:
                raise
            except Exception as exc:
                trace.append(f"sites: could not choose a site for {entity}: {exc}")
        if domains:
            found[entity] = domains
            trace.append(f"sites: {entity} -> {', '.join(domains)} ({how})")
        else:
            trace.append(f"sites: no official site found for {entity}; it gets no web claims (set one with --sites)")
    return {"plan": plan.model_copy(update={"official_sites": found}), "trace": trace}