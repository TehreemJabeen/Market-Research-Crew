"""Community collectors: what real people say, from sources that can be read legitimately.

- Hacker News: the free public Algolia search API (no key).
- Reddit: through Tavily with a domain filter, so no scraping and no extra credentials. Results are thread
  snippets, not full comment trees. (Reddit's own API needs an OAuth app; add it here if you have one.)

Sites that block bots or forbid scraping in their terms (G2, Capterra, ...) are deliberately not used.
"""
import hashlib

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from app.tools.search import tavily_search

HN_URL = "https://hn.algolia.com/api/v1/search"


class CommunityPost(BaseModel):
    id: str
    thread_id: str  # posts in the same discussion share it
    url: str
    text: str
    source: str  # "hn" | "reddit"
    score: int | None = None
    created: int | None = None  # epoch seconds, if known


def strip_html(html: str) -> str:
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())


def hn_search(query: str, n: int, timeout: float = 15.0) -> list[CommunityPost]:
    resp = httpx.get(HN_URL, params={"query": query, "tags": "comment", "hitsPerPage": n}, timeout=timeout)
    resp.raise_for_status()
    posts = []
    for hit in resp.json().get("hits", []):
        text = strip_html(hit.get("comment_text") or "")
        object_id = hit.get("objectID")
        if not text or not object_id:
            continue
        posts.append(
            CommunityPost(
                id=f"hn-{object_id}",
                thread_id=f"hn-{hit.get('story_id') or object_id}",
                url=f"https://news.ycombinator.com/item?id={object_id}",
                text=text,
                source="hn",
                score=hit.get("points"),
                created=hit.get("created_at_i"),
            )
        )
    return posts


def reddit_search(query: str, n: int, api_key: str) -> list[CommunityPost]:
    posts = []
    for r in tavily_search(query, api_key, n, include_domains=["reddit.com"]):
        text = " ".join(f"{r.title}. {r.snippet}".split())
        thread = r.url.split("?")[0].rstrip("/")
        posts.append(
            CommunityPost(
                id="rd-" + hashlib.md5(r.url.encode()).hexdigest()[:10],
                thread_id=thread,
                url=r.url,
                text=text,
                source="reddit",
            )
        )
    return posts


def collect_posts(entities: list[str], dimension: str, n: int, tavily_key: str) -> tuple[list[CommunityPost], list[str]]:
    """Gather posts about each entity on one dimension. One source failing never stops the other.
    Returns (posts, notes about what failed or was skipped)."""
    posts: list[CommunityPost] = []
    notes: list[str] = []
    ents = entities[:4] or [""]
    per_entity = max(3, n // len(ents))
    for entity in ents:
        query = f"{entity} {dimension}".strip()
        try:
            posts += hn_search(query, per_entity)
        except Exception as exc:
            notes.append(f"hacker news search failed for '{query}': {exc}")

    reddit_query = f"{' '.join(entities[:3])} {dimension} experience".strip()
    if tavily_key:
        try:
            posts += reddit_search(reddit_query, min(n, 10), tavily_key)
        except Exception as exc:
            notes.append(f"reddit search failed for '{reddit_query}': {exc}")
    else:
        notes.append("reddit skipped: TAVILY_API_KEY is not set")

    unique: dict[str, CommunityPost] = {}
    for p in posts:
        unique.setdefault(p.id, p)
    return list(unique.values()), notes
