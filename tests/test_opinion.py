from app.config import Settings
from app.opinion import (
    build_themes,
    confidence_for,
    filter_posts,
    is_comparative,
    looks_promotional,
    sentiment_stat,
    sentiment_table,
    stance_from_counts,
    summarize_entities,
)
from app.state import Snippet
from app.tools.community import CommunityPost

LONG = "This is a perfectly ordinary comment that is long enough to keep around."


def post(i, text, score=None, created=None):
    return CommunityPost(id=f"p{i}", thread_id=f"t{i}", url=f"https://x/{i}", text=text, source="hn", score=score, created=created)


def snip(i, thread, entity="Alpha", dim="pricing", pol="negative", score=-0.8):
    return Snippet(
        id=f"s{i}", task_id="t1", post_id=f"p{i}", thread_id=thread, url=f"https://x/{i}", source="hn",
        entity=entity, dimension=dim, quote=f"quote {i}", polarity=pol, polarity_score=score,
    )


def test_promotional_detection():
    assert looks_promotional("Use my referral code ABC123 for a discount")
    assert looks_promotional("see https://a.com and https://b.com and https://c.com for details")
    assert not looks_promotional(LONG)


def test_filter_posts_drops_each_kind_and_prefers_better_scored():
    now = 2_000_000_000
    posts = [
        post(1, "too short"),
        post(2, "Use my referral code ABC for a discount. " + LONG),
        post(3, LONG + " old", created=now - 4 * 365 * 86400),
        post(4, LONG + " fresh", created=now - 86400),
        post(5, LONG + " fresh"),  # same text as post 4
        post(6, LONG + " best", score=50),
    ]
    kept, stats = filter_posts(posts, Settings(), now=now)
    assert [p.id for p in kept] == ["p6", "p4"]  # higher score first
    assert stats == {"too_short": 1, "promotional": 1, "too_old": 1, "duplicate": 1}
    capped, _ = filter_posts(posts, Settings(max_posts_per_task=1), now=now)
    assert [p.id for p in capped] == ["p6"]


def test_comparative_detection():
    ents = ["Alpha", "Beta"]
    assert is_comparative("Alpha beats Beta on speed", ents)  # two entities
    assert is_comparative("We switched from Alpha last year", ents)  # switching phrase
    assert is_comparative("alpha is great vs. spreadsheets", ents)
    assert not is_comparative("Alpha is too slow", ents)


def test_stance_and_confidence_thresholds():
    assert stance_from_counts(2, 0, 0) == "positive"
    assert stance_from_counts(0, 3, 1) == "negative"
    assert stance_from_counts(2, 2, 0) == "mixed"
    assert stance_from_counts(0, 0, 3) == "neutral"
    assert stance_from_counts(1, 0, 2) == "neutral"  # a weak lean is not a stance
    assert stance_from_counts(0, 0, 0) == "neutral"
    assert [confidence_for(n) for n in (1, 2, 3, 4, 5, 9)] == ["low", "low", "medium", "medium", "high", "high"]


def test_sentiment_is_weighted_by_thread_not_by_comment():
    # one long argument (5 negative comments in one thread) against a single positive thread
    snippets = [snip(i, "big-argument") for i in range(5)] + [snip(9, "small", pol="positive", score=0.8)]
    stat = sentiment_stat(snippets)
    assert stat.n_threads == 2 and stat.score == 0.0 and stat.label == "mixed"
    assert sentiment_stat([]).label == "no data"


def test_build_themes_counts_in_code_and_drops_anecdotes():
    settings = Settings(theme_min_threads=2)
    snippets = [
        snip(1, "a"), snip(2, "b"), snip(3, "b"),
        snip(4, "c", entity="Beta"),
        snip(5, "d", pol="positive", score=0.9),
    ]
    groups = [
        ("alpha", "too pricey", ["s1", "s2", "s3", "s4"]),  # label normalised; s4 is Beta's, so excluded
        ("Beta", "lonely", ["s4"]),  # one thread: anecdotal
        ("Alpha", "reuse", ["s1"]),  # already used by another theme
        ("Gamma", "nobody", ["s5"]),  # not a known entity
        ("Alpha", "fans", ["s5"]),  # one thread: anecdotal
    ]
    themes, anecdotal = build_themes(groups, snippets, ["Alpha", "Beta"], settings, start=7)
    assert anecdotal == 2 and len(themes) == 1
    t = themes[0]
    assert (t.id, t.entity, t.n_snippets, t.n_threads, t.stance, t.confidence) == ("th7", "Alpha", 3, 2, "negative", "low")
    assert t.snippet_ids == ["s1", "s2", "s3"]
    assert [e.url for e in t.examples] == ["https://x/1", "https://x/2"]  # examples come from distinct threads


def test_summaries_and_sentiment_table():
    snippets = [snip(1, "a"), snip(2, "b"), snip(3, "c", dim="features", pol="positive", score=0.9)]
    summaries = summarize_entities(snippets, [], ["Alpha", "Beta"], ["pricing", "features"])
    assert [s.entity for s in summaries] == ["Alpha"]  # no snippets about Beta, so no summary
    table = sentiment_table(summaries, ["Alpha", "Beta"], ["pricing", "features"])
    assert table.columns == ["Entity", "pricing", "features", "Overall"]
    assert table.rows == [
        ["Alpha", "negative (2 threads)", "positive (1 thread)", "negative (3 threads)"],
        ["Beta", "no data", "no data", "no data"],
    ]
