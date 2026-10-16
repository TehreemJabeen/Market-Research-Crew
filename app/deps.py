#Bundles the tools (LLM, search, fetch, community, scorer) into one object that gets passed to every agent. Tests swap in fakes here.
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings
from app.llm import BaseLLM, make_llm
from app.tools.community import CommunityPost, collect_posts
from app.tools.fetch import fetch_text
from app.tools.search import SearchResult, tavily_search
from app.tools.sentiment import make_scorer


@dataclass
class Deps:
    """Everything the agents need. Injected so tests can swap in fakes."""

    llm: BaseLLM
    search: Callable[[str, int], list[SearchResult]]
    fetch: Callable[[str], str]
    settings: Settings
    # (entities, dimension, n) -> (posts, notes about failures)
    community: Callable[[list[str], str, int], tuple[list[CommunityPost], list[str]]] | None = None
    scorer: Any = None  # a SentimentScorer (see app/tools/sentiment.py)


def build_deps(settings: Settings) -> Deps:
    llm = make_llm(settings)
    return Deps(
        llm=llm,
        search=lambda query, n: tavily_search(query, settings.tavily_api_key, n),
        fetch=fetch_text,
        settings=settings,
        community=lambda entities, dimension, n: collect_posts(entities, dimension, n, settings.tavily_api_key),
        scorer=make_scorer(settings, llm),
    )
