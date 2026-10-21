import threading

import httpx
from pydantic import BaseModel

_lock = threading.Lock()
_calls = 0  # Tavily requests made since the last reset (a basic search costs 1 credit)


def search_calls() -> int:
    return _calls


def reset_search_calls() -> None:
    global _calls
    with _lock:
        _calls = 0


class SearchResult(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""


def tavily_search(
    query: str, api_key: str, max_results: int = 5, include_domains: list[str] | None = None
) -> list[SearchResult]:
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set (free tier: https://tavily.com)")
    global _calls
    with _lock:
        _calls += 1
    payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
    if include_domains:
        payload["include_domains"] = include_domains
    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return [
        SearchResult(title=r.get("title", ""), url=r["url"], snippet=r.get("content", ""))
        for r in resp.json().get("results", [])
        if r.get("url")
    ]
