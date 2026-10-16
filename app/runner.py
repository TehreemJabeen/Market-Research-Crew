import time

from app.deps import Deps
from app.graph import build_graph

RECURSION_LIMIT = 40  # hard cap on graph steps: a runaway loop cannot go on forever


def initial_state(question: str, dimensions: list[str] | None = None) -> dict:
    state = {"question": question, "loop_count": 0, "trace": []}
    if dimensions:
        state["user_dimensions"] = [d.strip() for d in dimensions if d and d.strip()]
    return state


def run_crew(question: str, deps: Deps, dimensions: list[str] | None = None) -> dict:
    """Run the whole crew and return the final state plus usage numbers.
    `dimensions` optionally forces comparison dimensions; the planner fills the remaining slots itself."""
    graph = build_graph(deps)
    start = time.time()
    final = graph.invoke(initial_state(question, dimensions), config={"recursion_limit": RECURSION_LIMIT})
    final["usage"] = {**deps.llm.usage(), "seconds": round(time.time() - start, 1)}
    return final


def stream_crew(question: str, deps: Deps, dimensions: list[str] | None = None):
    """Yield (node_name, state_update) as each step finishes. Used by the UI for a live view."""
    graph = build_graph(deps)
    for chunk in graph.stream(
        initial_state(question, dimensions), config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
    ):
        for node, update in chunk.items():
            yield node, update
