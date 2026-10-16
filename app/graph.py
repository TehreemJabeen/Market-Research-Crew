"""The LangGraph workflow.

planner -+-> research_web ------> factcheck -+-> research_web (re-research loop, capped)
         |                                    +-> gate -+
         +-> research_community -> opinions ---------> gate -> analyst -> writer -> editor

Every task gets a web (facts) researcher; every dimension gets one community (opinions) researcher.
The gate is the join: the Analyst must run exactly once, after BOTH branches are really finished, even when
the facts branch loops back for another research round.
"""
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.analyst import run_analyst
from app.agents.community import run_community_researcher
from app.agents.editor import run_editor
from app.agents.factchecker import run_factchecker
from app.agents.opinions import run_opinions
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.writer import run_writer
from app.deps import Deps
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
    plan: Plan
    tasks: Annotated[list[Task], merge_by_id]
    claims: Annotated[list[Claim], merge_by_id]
    snippets: Annotated[list[Snippet], merge_by_id]
    sources: Annotated[dict[str, str], merge_sources]  # url -> text actually shown to the LLM (for verification)
    loop_count: int
    facts_ready: bool  # set by the fact-checker: no task still needs another round
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
    draft: str
    report: str
    final_citations: list[dict]
    metrics: dict
    trace: Annotated[list[str], add]


class ResearchInput(TypedDict):
    task: Task
    question: str
    entities: list[str]
    dimensions: list[str]


def build_graph(deps: Deps):
    def sends(state, tasks, node):
        plan = state["plan"]
        return [
            Send(node, {"task": t, "question": state["question"], "entities": plan.entities, "dimensions": plan.dimensions})
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
        return out

    def after_factcheck(state: GraphState):
        pending = [t for t in state["tasks"] if t.status == "needs_more"]
        return sends(state, pending, "research_web") if pending else "gate"

    def gate(state: GraphState):
        return {"trace": []}  # a join point: it only decides, in gate_route, whether both branches are done

    def gate_route(state: GraphState):
        facts_done = state.get("facts_ready", False)
        opinions_done = state.get("opinions_ready", False) or not deps.settings.enable_community
        return "analyst" if facts_done and opinions_done else END

    g = StateGraph(GraphState)
    g.add_node("planner", lambda state: run_planner(state, deps))
    g.add_node("research_web", lambda payload: run_researcher(payload, deps), input_schema=ResearchInput)
    g.add_node("research_community", lambda payload: run_community_researcher(payload, deps), input_schema=ResearchInput)
    g.add_node("factcheck", lambda state: run_factchecker(state, deps))
    g.add_node("opinions", lambda state: run_opinions(state, deps))
    g.add_node("gate", gate)
    g.add_node("analyst", lambda state: run_analyst(state, deps))
    g.add_node("writer", lambda state: run_writer(state, deps))
    g.add_node("editor", lambda state: run_editor(state, deps))

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["research_web", "research_community"])
    g.add_edge("research_web", "factcheck")
    g.add_edge("research_community", "opinions")
    g.add_conditional_edges("factcheck", after_factcheck, ["research_web", "gate"])
    g.add_edge("opinions", "gate")
    g.add_conditional_edges("gate", gate_route, {"analyst": "analyst", END: END})
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "editor")
    g.add_edge("editor", END)
    return g.compile()
