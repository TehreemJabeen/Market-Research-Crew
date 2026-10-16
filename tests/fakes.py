"""Scripted stand-ins for the LLM, search, community collectors and sentiment scorer so the whole graph runs offline."""
import re

from app.agents.analyst import AnalysisOut, ClaimThemeLink, LinkSet
from app.agents.community import ExtractedSnippet, ExtractedSnippets
from app.agents.factchecker import Verdict, Verdicts
from app.agents.opinions import ThemeGroup, ThemeGroups
from app.agents.planner import Candidate, Candidates, QueryAnalysis, SubQuestion, SubQuestions
from app.agents.researcher import ExtractedClaim, ExtractedClaims, QueryList
from app.config import Settings
from app.deps import Deps
from app.llm import BaseLLM
from app.tools.community import CommunityPost
from app.tools.search import SearchResult
from app.tools.sentiment import Polarity, ScoreOut, ScoresOut

URL_A = "https://a.example/alpha"
URL_B = "https://b.example/beta"
PAGES = {
    URL_A: "Alpha costs ten dollars per user per month for teams. Alpha offers a free plan for individuals.",
    URL_B: "Beta includes kanban boards and timeline views for planning. Beta pricing starts at twelve dollars per seat monthly.",
}
FABRICATED_A = "Alpha has built-in AI that writes all your documents automatically."
FABRICATED_B = "Beta guarantees ninety nine percent uptime backed by a signed contract."


# ---- community data ---------------------------------------------------------------------------------


def hn(n, thread, text):
    return CommunityPost(id=f"hn-{n}", thread_id=f"hn-{thread}", url=f"https://news.ycombinator.com/item?id={n}", text=text, source="hn")


def rd(n, text):
    url = f"https://www.reddit.com/r/work/comments/{n}"
    return CommunityPost(id=f"rd-{n}", thread_id=url, url=url, text=text, source="reddit")


POSTS = {
    "pricing": [
        hn(1, "T1", "Alpha got way too expensive for our five person team, we are dropping it next month."),
        hn(2, "T2", "We pay for Alpha and honestly the price per seat is too high once you add guests."),
        rd(3, "Beta's free tier is generous and the paid plan is cheap for what it offers, happy with it."),
        hn(4, "T4", "Use my referral code ALPHA10 to get a discount on Alpha, I am the founder of a tool that helps with this."),
        hn(5, "T5b", "Alpha got way too expensive for our five person team, we are dropping it next month."),  # duplicate of hn-1
        hn(6, "T5", "Switched from Alpha to Beta and the pricing is much better."),
    ],
    "features": [
        hn(7, "T6", "Beta kanban boards and timeline views are solid, we use them daily on the team."),
        hn(8, "T7", "The timeline view in Beta is great for planning, really happy with the timeline view."),
        rd(9, "Beta mobile app is laggy and crashes when opening big boards."),
        hn(10, "T9", "Alpha has nice templates but nothing special."),
    ],
}
# What the fake LLM "extracts" per dimension: (post id, entity, quote). The last pricing one is invented.
SNIPPETS = {
    "pricing": [
        ("hn-1", "Alpha", "Alpha got way too expensive for our five person team"),
        ("hn-2", "Alpha", "the price per seat is too high once you add guests"),
        ("rd-3", "Beta", "Beta's free tier is generous and the paid plan is cheap"),
        ("hn-6", "Beta", "Switched from Alpha to Beta and the pricing is much better"),
        ("hn-2", "Alpha", "Alpha support ghosted us for weeks"),  # fabricated: not in any post
    ],
    "features": [
        ("hn-7", "Beta", "Beta kanban boards and timeline views are solid"),
        ("hn-8", "Beta", "The timeline view in Beta is great for planning"),
        ("rd-9", "Beta", "Beta mobile app is laggy and crashes when opening big boards"),
        ("hn-10", "Alpha", "Alpha has nice templates but nothing special"),
    ],
}


def fake_community(entities, dimension, n):
    return list(POSTS.get(dimension, [])), []


class FakeScorer:
    """A keyword scorer standing in for the Hugging Face model."""

    name = "fake"
    NEG = ("expensive", "too high", "laggy", "crashes")
    POS = ("generous", "solid", "great", "happy")

    def score(self, items):
        out = []
        for item in items:
            text = item.text.lower()
            if any(w in text for w in self.NEG):
                out.append(Polarity("negative", -0.8))
            elif any(w in text for w in self.POS):
                out.append(Polarity("positive", 0.8))
            else:
                out.append(Polarity("neutral", 0.0))
        return out


def theme_label(text):
    if "expensive" in text or "price per seat" in text:
        return "too expensive for small teams"
    if "free tier" in text or "pricing is much better" in text:
        return "good value and a generous free tier"
    if "kanban" in text or "timeline" in text:
        return "kanban and timeline views work well"
    if "laggy" in text:
        return "mobile app lags"
    if "templates" in text:
        return "templates are fine"
    return None


# ---- the fake LLM -----------------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    def __init__(self):
        super().__init__(rpm=0, max_calls=1000)

    def structured(self, system, prompt, schema, retries=2):
        self._reserve()
        if schema is QueryAnalysis:
            return QueryAnalysis(
                rewritten_question="Compare Alpha and Beta for small teams",
                entities=["Alpha", "Beta"],
                category="work tools",
                audience="small teams",
                decision="which tool to adopt",
                scout_queries=["alpha vs beta comparison"],
            )
        if schema is Candidates:
            return Candidates(candidates=[
                Candidate(name="pricing", why_it_matters="Budget is tight for small teams.",
                          relevance=5, differentiation=4, answerable=5, source_urls=[URL_A, URL_B]),
                Candidate(name="features", why_it_matters="Teams need the core workflow covered.",
                          relevance=4, differentiation=4, answerable=4, source_urls=[URL_A]),
            ])
        if schema is SubQuestions:
            return SubQuestions(items=[
                SubQuestion(dimension="pricing", question="pricing of Alpha and Beta"),
                SubQuestion(dimension="features", question="features of Alpha and Beta"),
            ])
        if schema is QueryList:
            return QueryList(queries=["alpha beta features official"])
        if schema is ExtractedClaims:
            return self._extract(prompt)
        if schema is Verdicts:
            return Verdicts(verdicts=[Verdict(claim_id=i, verdict="supported") for i in re.findall(r"id=(\S+)", prompt)])
        if schema is ExtractedSnippets:
            dim = re.search(r"DIMENSION: (.*)", prompt).group(1).strip()
            return ExtractedSnippets(snippets=[ExtractedSnippet(post_id=p, entity=e, quote=q) for p, e, q in SNIPPETS.get(dim, [])])
        if schema is ScoresOut:  # the LLM scorer, used for comparative snippets
            items = [
                ScoreOut(id=int(m.group(1)), label="positive" if "much better" in m.group(2) else "neutral", confidence=0.9)
                for m in re.finditer(r"^\[(\d+)\] entity=.*? \| dimension=.*? \| text: (.*)$", prompt, re.M)
            ]
            return ScoresOut(items=items)
        if schema is ThemeGroups:
            groups: dict[tuple[str, str], list[str]] = {}
            for m in re.finditer(r"^\[(\S+)\] entity=(\S+) \| (.*)$", prompt, re.M):
                sid, entity, text = m.groups()
                label = theme_label(text)
                if label:
                    groups.setdefault((entity, label), []).append(sid)
            return ThemeGroups(themes=[ThemeGroup(entity=e, label=l, snippet_ids=ids) for (e, l), ids in groups.items()])
        if schema is LinkSet:
            claims = re.findall(r"^CLAIM \[(\d+)\] \((.*?)\): (.*)$", prompt, re.M)
            themes = re.findall(r"^THEME \[(\d+)\] \((.*?),", prompt, re.M)
            return LinkSet(links=[
                ClaimThemeLink(claim=int(cn), theme=int(tn), relation="qualifies" if "$10" in text else "supports", note="fake")
                for cn, cent, text in claims for tn, tent in themes if cent == tent
            ])
        if schema is AnalysisOut:
            c = re.search(r"^CLAIM \[(\d+)\] \(", prompt, re.M)
            t = re.search(r"^THEME \[(\d+)\] \(", prompt, re.M)
            return AnalysisOut(
                insights=[f"Alpha and Beta differ on price and features [{c.group(1)}]."] if c else [],
                public_opinion=[f"In the threads we sampled, users report pricing frustration with Alpha [{t.group(1)}]."] if t else [],
            )
        raise AssertionError(f"unexpected schema {schema}")

    def _extract(self, prompt):
        subq = re.search(r"SUB-QUESTION: (.*)", prompt).group(1)
        attempt = int(re.search(r"ATTEMPT: (\d+)", prompt).group(1))
        if "pricing" in subq:
            return ExtractedClaims(claims=[
                ExtractedClaim(entity="Alpha", dimension="pricing", text="Alpha costs $10 per user per month for teams.",
                               source_url=URL_A, supporting_quote="Alpha costs ten dollars per user per month for teams."),
                ExtractedClaim(entity="Beta", dimension="pricing", text="Beta starts at $12 per seat monthly.",
                               source_url=URL_B, supporting_quote="Beta pricing starts at twelve dollars per seat monthly."),
                ExtractedClaim(entity="Alpha", dimension="features", text="Alpha writes documents with built-in AI.",
                               source_url=URL_A, supporting_quote=FABRICATED_A),  # hallucinated quote
            ])
        if attempt == 1:  # first try for the features task: only a hallucinated claim
            return ExtractedClaims(claims=[
                ExtractedClaim(entity="Beta", dimension="features", text="Beta guarantees 99% uptime.",
                               source_url=URL_B, supporting_quote=FABRICATED_B),
            ])
        return ExtractedClaims(claims=[  # retry succeeds
            ExtractedClaim(entity="Beta", dimension="features", text="Beta includes kanban boards and timeline views.",
                           source_url=URL_B, supporting_quote="Beta includes kanban boards and timeline views for planning."),
            ExtractedClaim(entity="Alpha", dimension="features", text="Alpha offers a free plan for individuals.",
                           source_url=URL_A, supporting_quote="Alpha offers a free plan for individuals."),
        ])

    def complete(self, system, prompt, json_mode=False):
        self._reserve()
        if "Writer agent" in system:
            items = re.findall(r"^(CLAIM|THEME) \[(\d+)\]", prompt, re.M)
            lines = ["# Report", "", "## Findings"]
            first_theme = None
            for kind, n in items:
                if kind == "CLAIM":
                    lines.append(f"- Finding number {n} is described in the verified evidence for this item [{n}].")
                else:
                    first_theme = first_theme or n
                    lines.append(f"- Users report on Hacker News and Reddit that theme {n} came up across several threads [{n}].")
            if first_theme:  # an opinion with no attribution: the Editor must notice it
                lines.append(f"- Pricing is a real problem for this product [{first_theme}].")
            lines += [
                "- Revenue grew 300% last year according to nobody at all in the sources.",  # uncited + factual -> dropped
                "- Another claim cites a missing source right here [99].",  # invalid ref -> stripped
            ]
            return "\n".join(lines)
        if "Editor agent" in system:  # an editor that fixes nothing; the deterministic cleanup must still work
            return re.search(r"<report>(.*?)</report>", prompt, re.S).group(0)
        raise AssertionError("unexpected complete() call")


def fake_search(query, n):
    return [SearchResult(title="A", url=URL_A, snippet=PAGES[URL_A]), SearchResult(title="B", url=URL_B, snippet=PAGES[URL_B])]


def fake_fetch(url):
    return PAGES.get(url, "")


def make_fake_deps(**settings_overrides) -> Deps:
    settings = Settings(rpm=0, **settings_overrides)
    return Deps(
        llm=FakeLLM(), search=fake_search, fetch=fake_fetch, settings=settings,
        community=fake_community, scorer=FakeScorer(),
    )
