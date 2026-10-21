"""FastAPI app. Development: uvicorn api.main:create_app --factory --reload  (Vite proxies /api to port 8000).
Production: the same command also serves the built React app from frontend/dist."""
import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.executors import DemoExecutor, RealExecutor
from api.manager import RunManager
from api.schemas import RunCreated, RunDetail, RunRequest, RunSummary
from api.store import TERMINAL, Store
from app.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
POLL_SECONDS = 0.4
HEARTBEAT_SECONDS = 15


def create_app(manager: RunManager | None = None, store: Store | None = None) -> FastAPI:
    demo = os.getenv("DEMO_MODE", "0").lower() in ("1", "true", "yes")
    if manager is None:
        store = store or Store(os.getenv("RUNS_DB", "runs.db"))
        manager = RunManager(store, DemoExecutor() if demo else RealExecutor())
    store = manager.store
    app = FastAPI(title="Market Research Crew API")
    if os.getenv("CORS_ORIGINS"):
        app.add_middleware(CORSMiddleware, allow_origins=os.environ["CORS_ORIGINS"].split(","), allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/config")
    def config():
        s = get_settings()
        return {
            "demo": demo,
            "providers": {"gemini": bool(s.gemini_api_key), "grok": bool(s.xai_api_key)},
            "default_provider": s.provider,
            "tavily": bool(s.tavily_api_key),
            "local_sentiment_model": importlib.util.find_spec("transformers") is not None,
            "app_country": s.app_country,
        }

    @app.post("/api/runs", response_model=RunCreated, status_code=202)
    def create_run(req: RunRequest):
        return RunCreated(run_id=manager.submit(req.model_dump()))

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs(limit: int = 100):
        return store.list_runs(min(max(limit, 1), 500))

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return run

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        if not store.get_run(run_id, with_result=False):
            raise HTTPException(404, "run not found")
        return {"cancelled": manager.cancel(run_id)}

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        run = store.get_run(run_id, with_result=False)
        if not run:
            raise HTTPException(404, "run not found")
        if run["status"] in ("queued", "running"):
            raise HTTPException(409, "cancel the run before deleting it")
        store.delete_run(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after: int = 0):
        if not store.get_run(run_id, with_result=False):
            raise HTTPException(404, "run not found")
        last = max(after, int(request.headers.get("last-event-id", "0") or 0))

        async def stream():
            nonlocal last
            quiet_since = time.time()
            while True:
                events = await asyncio.to_thread(store.events_after, run_id, last)
                for event in events:
                    last = event["seq"]
                    quiet_since = time.time()
                    yield f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                run = await asyncio.to_thread(store.get_run, run_id, False)
                if run["status"] in TERMINAL and not events:
                    return
                if await request.is_disconnected():
                    return
                if time.time() - quiet_since > HEARTBEAT_SECONDS:
                    quiet_since = time.time()
                    yield ": ping\n\n"
                await asyncio.sleep(POLL_SECONDS)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    # ---- the built React app (production) ------------------------------------------------------
    dist = Path(os.getenv("FRONTEND_DIST", ROOT / "frontend" / "dist"))
    if (dist / "index.html").exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            target = (dist / path).resolve()
            if path and target.is_file() and dist.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app
