"""Scout: find real comparison / buyer's-guide pages so the Planner picks dimensions from evidence,
not just from the model's memory. Plain code: search, fetch, truncate. No LLM."""


def run_scout(queries: list[str], deps, trace: list[str]) -> dict[str, str]:
    """Return {url: page text} for up to `max_scout_sources` pages. Failures are logged and skipped."""
    s = deps.settings
    sources: dict[str, str] = {}
    for query in queries[: s.max_scout_queries]:
        if len(sources) >= s.max_scout_sources:
            break
        try:
            results = deps.search(query, 5)
        except Exception as exc:  # scouting is a bonus; the planner can still work without it
            trace.append(f"scout: search failed for '{query}': {exc}")
            continue
        for r in results:
            if r.url in sources:
                continue
            if len(sources) >= s.max_scout_sources:
                break
            text = deps.fetch(r.url) or r.snippet
            if text:
                sources[r.url] = text[: s.scout_chars]
    trace.append(f"scout: read {len(sources)} pages from {min(len(queries), s.max_scout_queries)} queries")
    return sources
