import httpx

from app.tools import community
from app.tools.community import CommunityPost, collect_posts, hn_search, reddit_search, strip_html


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_strip_html():
    assert strip_html("<p>Hello <b>big</b>   world &amp; more</p>") == "Hello big world & more"
    assert strip_html("") == ""


def test_hn_search_parses_hits(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params)
        return FakeResponse({"hits": [
            {"objectID": "11", "comment_text": "<p>Notion is <i>slow</i> on big tables &amp; sync.</p>",
             "story_id": 5, "points": None, "created_at_i": 1700000000},
            {"objectID": "12", "comment_text": "", "story_id": 5},  # empty text: skipped
            {"comment_text": "no id", "story_id": 5},  # no id: skipped
        ]})

    monkeypatch.setattr(httpx, "get", fake_get)
    posts = hn_search("notion slow", 5)
    assert seen["params"] == {"query": "notion slow", "tags": "comment", "hitsPerPage": 5}
    assert len(posts) == 1
    p = posts[0]
    assert (p.id, p.thread_id, p.source, p.created) == ("hn-11", "hn-5", "hn", 1700000000)
    assert p.url == "https://news.ycombinator.com/item?id=11"
    assert p.text == "Notion is slow on big tables & sync."


def test_reddit_search_goes_through_tavily_with_a_domain_filter(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json=json)
        return FakeResponse({"results": [{
            "title": "Coda vs Notion", "url": "https://www.reddit.com/r/x/comments/abc/coda_vs_notion/?utm=1",
            "content": "I prefer Coda for databases.",
        }]})

    monkeypatch.setattr(httpx, "post", fake_post)
    posts = reddit_search("coda notion databases", 5, "key")
    assert captured["json"]["include_domains"] == ["reddit.com"]
    p = posts[0]
    assert p.source == "reddit" and p.id.startswith("rd-")
    assert p.thread_id == "https://www.reddit.com/r/x/comments/abc/coda_vs_notion"  # query string and slash removed
    assert p.text == "Coda vs Notion. I prefer Coda for databases."


def make_post(pid, source="hn"):
    return CommunityPost(id=pid, thread_id=f"t-{pid}", url=f"https://x/{pid}", text="x" * 50, source=source)


def test_collect_posts_queries_each_entity_and_survives_a_failing_source(monkeypatch):
    queries = []

    def fake_hn(query, n):
        queries.append((query, n))
        if "Beta" in query:
            raise RuntimeError("boom")
        return [make_post("hn-1")]

    monkeypatch.setattr(community, "hn_search", fake_hn)
    monkeypatch.setattr(community, "reddit_search", lambda q, n, key: [make_post("rd-1", "reddit"), make_post("hn-1")])
    posts, notes = collect_posts(["Alpha", "Beta"], "pricing", 12, "key")
    assert queries == [("Alpha pricing", 6), ("Beta pricing", 6)]
    assert sorted(p.id for p in posts) == ["hn-1", "rd-1"]  # duplicate id collapsed
    assert len(notes) == 1 and "hacker news search failed for 'Beta pricing'" in notes[0]


def test_collect_posts_without_tavily_key_skips_reddit(monkeypatch):
    queries = []
    monkeypatch.setattr(community, "hn_search", lambda q, n: queries.append(q) or [])
    posts, notes = collect_posts([], "pricing", 10, "")
    assert posts == [] and queries == ["pricing"]
    assert notes == ["reddit skipped: TAVILY_API_KEY is not set"]
