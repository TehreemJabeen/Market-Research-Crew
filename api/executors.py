"""Executors turn a run request into events and a result. The real one drives the crew; the demo one replays a
recorded run (so the UI can be shown with no API keys and no waiting)."""
import json
import os
import time
from pathlib import Path
from typing import Callable

from fastapi.encoders import jsonable_encoder

from app.config import get_settings
from app.deps import build_deps
from app.runner import stream_crew
from app.tools.appstore import app_enabled
from app.tools.search import reset_search_calls, search_calls

Emit = Callable[[str, dict], None]
RESULT_KEYS = ("report", "final_citations", "gaps", "table", "evidence_table", "sentiment_table", "themes",
               "opinion_summaries", "unaddressed", "plan")
DEMO_FILE = Path(__file__).parent / "demo_run.json"


class Cancelled(Exception):
    pass


def describe(node: str, update: dict, settings) -> dict:
    """A small, UI-friendly summary of one agent step (full data goes into the final result only)."""
    data: dict = {"node": node, "trace": list(update.get("trace") or [])}
    if node == "planner":
        plan, tasks = update["plan"], update["tasks"]
        dims = list(dict.fromkeys(t.dimension for t in tasks))
        data["plan"] = {
            "rewritten_question": plan.rewritten_question, "audience": plan.audience, "entities": plan.entities,
            "dimensions": plan.dimensions, "is_app": plan.is_app, "n_web": len(tasks),
            "n_community": len(dims) if settings.enable_community else 0,
            "n_app": len(plan.entities) if app_enabled(settings, plan.is_app) else 0,
        }
    if node == "sites":
        data["official_sites"] = update["plan"].official_sites
    if "claims" in update:
        data["n_claims"] = len(update["claims"])
    if "snippets" in update:
        data["n_snippets"] = len(update["snippets"])
    return data


class RealExecutor:
    def __init__(self, deps_factory=build_deps, settings_factory=get_settings):
        self.deps_factory, self.settings_factory = deps_factory, settings_factory

    def run(self, options: dict, question: str, emit: Emit, is_cancelled: Callable[[], bool]) -> tuple[dict, dict]:
        settings = self.settings_factory(
            provider=options.get("provider"),
            enable_community=False if options.get("community") is False else None,
            app_mode=options.get("apps"),
            sentiment_backend=options.get("sentiment_backend"),
        )
        deps = self.deps_factory(settings)
        reset_search_calls()
        start = time.time()
        final: dict = {"claims": {}}
        for node, update in stream_crew(question, deps, options.get("dimensions") or None, options.get("sites") or None):
            if is_cancelled():
                raise Cancelled()
            if not update or node == "gate":
                continue
            emit("step", describe(node, update, settings))
            for claim in update.get("claims", []):
                final["claims"][claim.id] = claim  # later updates replace earlier ones
            for key in RESULT_KEYS:
                if key in update:
                    final[key] = update[key]
        final["claims"] = list(final["claims"].values())
        usage = {**deps.llm.usage(), "seconds": round(time.time() - start, 1), "tavily_calls": search_calls()}
        return jsonable_encoder(final), usage


class DemoExecutor:
    """Replays api/demo_run.json (recorded with `python -m api.record`)."""

    def __init__(self, path: Path = DEMO_FILE, speed: float | None = None):
        self.path = path
        self.speed = float(os.getenv("DEMO_SPEED", "1")) if speed is None else speed

    def run(self, options: dict, question: str, emit: Emit, is_cancelled: Callable[[], bool]) -> tuple[dict, dict]:
        demo = json.loads(self.path.read_text(encoding="utf-8"))
        for event in demo["events"]:
            time.sleep(max(0.0, event.get("delay", 0.5)) * self.speed)
            if is_cancelled():
                raise Cancelled()
            emit(event["type"], event["data"])
        result = {**demo["result"], "sample": True, "sample_question": demo.get("question", "")}
        return result, demo["usage"]
