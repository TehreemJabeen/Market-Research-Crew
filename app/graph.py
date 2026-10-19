"""The LangGraph workflow.

planner -> sites -+-> research_web -----------+
                  +-> research_app (optional) +-> gate -> analyst -> writer
                  +-> research_community -> opinions -+

`sites` finds each entity's official website once. Every task then gets a web (facts) researcher that reads only
those official sites; every entity gets an App Store lookup when the products are apps; every dimension gets one
community (opinions) researcher. There is no fact-checker, no editor and no retry loop.
The gate is the join: the Analyst runs exactly once, after the opinions track has finished (it is one step
longer than the others), or straight after the facts tracks when the community track is switched off.
"""
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.analyst import run_analyst
from app.agents.appstore import AppInput, run_appstore
from app.agents.community import run_community_researcher
from app.agents.opinions import run_opinions
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.sites import run_sites
from app.agents.writer import run_writer
from app.deps import Deps
from app.tools.appstore import app_enabled
from app.state import Claim, OpinionSummary, Plan, Snippet, Table, Task, Theme


def merge_by_id(old, new):
    """Reducer: later updates replace earlier ones with the same id (claims get re-labelled by the fact-checker)."""
    merged = {x.id: x for x in (old or [])}
    for x in new or []:
        merged[x.id] = x
    return list(merged.values())


def merge_sources(old, new):
    """Reducer for url -> page text. If two branches saw the same url, keep the longer text."""
    merged = dict(old or {})
    for url, text in (new or {}).items():
        if len(text) >= len(merged.get(url, "")):
            merged[url] = text
    return merged


class GraphState(TypedDict, total=False):
    question: str
    user_dimensions: list[str]  # optional: dimensions the user insists on
    site_overrides: dict[str, str]  # optional: entity -> official domain, set by the user
    plan: Plan
    tasks: Annotated[list[Task], merge_by_id]
    claims: Annotated[list[Claim], merge_by_id]
    snippets: Annotated[list[Snippet], merge_by_id]
    sources: Annotated[dict[str, str], merge_sources]  # url -> text actually shown to the LLM (for verification)
    opinions_ready: bool  # set by the opinion processor
    themes: list[Theme]
    opinion_summaries: list[OpinionSummary]
    sentiment_table: Table
    refs: dict[int, str]  # citation number -> claim id or theme id
    table: Table
    evidence_table: Table
    links: list
    fact_status: dict[str, str]
    unaddressed: list[str]
    insights: list[str]
    public_opinion: list[str]
    gaps: list[str]
    report: str
    final_citations: list[dict]
    trace: Annotated[list[str], add]


class ResearchInput(TypedDict):
    task: Task
    question: str
    entities: list[str]
    dimensions: list[str]
    official_sites: dict[str, list[str]]


def build_graph(deps: Deps):
    def sends(state, tasks, node):
        plan = state["plan"]
        return [
            Send(node, {
                "task": t, "question": state["question"], "entities": plan.entities,
                "dimensions": plan.dimensions, "official_sites": plan.official_sites,
            })
            for t in tasks
        ]

    def fan_out(state: GraphState):
        out = sends(state, state["tasks"], "research_web")
        if deps.settings.enable_community:
            first_per_dimension, seen = [], set()
            for t in state["tasks"]:
                if t.dimension not in seen:
                    seen.add(t.dimension)
                    first_per_dimension.append(t)
            out += sends(state, first_per_dimension, "research_community")
        plan = state["plan"]
        if app_enabled(deps.settings, plan.is_app):
            out += [
                Send("research_app", {"entity": e, "domains": plan.official_sites.get(e, []), "dimensions": plan.dimensions})
                for e in plan.entities
            ]
        return out

    def gate(state: GraphState):
        return {"trace": []}  # a join point: it only decides, in gate_route, whether both branches are done

    def gate_route(state: GraphState):
        opinions_done = state.get("opinions_ready", False) or not deps.settings.enable_community
        return "analyst" if opinions_done else END  # the facts tracks are always finished by the time this matters

    g = StateGraph(GraphState)
    g.add_node("planner", lambda state: run_planner(state, deps))
    g.add_node("sites", lambda state: run_sites(state, deps))
    g.add_node("research_web", lambda payload: run_researcher(payload, deps), input_schema=ResearchInput)
    g.add_node("research_community", lambda payload: run_community_researcher(payload, deps), input_schema=ResearchInput)
    g.add_node("research_app", lambda payload: run_appstore(payload, deps), input_schema=AppInput)
    g.add_node("opinions", lambda state: run_opinions(state, deps))
    g.add_node("gate", gate)
    g.add_node("analyst", lambda state: run_analyst(state, deps))
    g.add_node("writer", lambda state: run_writer(state, deps))

    g.add_edge(START, "planner")
    g.add_edge("planner", "sites")
    g.add_conditional_edges("sites", fan_out, ["research_web", "research_community", "research_app"])
    g.add_edge("research_web", "gate")
    g.add_edge("research_app", "gate")
    g.add_edge("research_community", "opinions")
    g.add_edge("opinions", "gate")
    g.add_conditional_edges("gate", gate_route, {"analyst": "analyst", END: END})
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)
    return g.compile()