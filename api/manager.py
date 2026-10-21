"""Runs the crew one request at a time. Every run has its own rate limiter, so two at once would double the request
rate and trip the free-tier limits: extra runs wait in a queue."""
import os
import queue
import threading
import time
import uuid

from api.executors import Cancelled
from api.store import Store
from app.llm import BudgetExceeded


class RunManager:
    def __init__(self, store: Store, executor):
        self.store, self.executor = store, executor
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self.store.mark_orphans_failed()
        threading.Thread(target=self._loop, name="run-worker", daemon=True).start()

    # ---- public --------------------------------------------------------------------------------

    def submit(self, request: dict) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.store.create_run(run_id, request["question"], {k: v for k, v in request.items() if k != "question"})
        self.store.add_event(run_id, "status", {"status": "queued", "position": self._queue.qsize() + 1})
        self._queue.put(run_id)
        return run_id

    def cancel(self, run_id: str) -> bool:
        run = self.store.get_run(run_id, with_result=False)
        if not run or run["status"] not in ("queued", "running"):
            return False
        with self._lock:
            self._cancel.add(run_id)
        if run["status"] == "queued":
            self._finish(run_id, "cancelled")
        else:
            self.store.add_event(run_id, "status", {"status": "cancelling"})
        return True

    # ---- worker --------------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            self._run(self._queue.get())

    def _is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancel

    def _finish(self, run_id: str, status: str, **fields) -> None:
        self.store.update_run(run_id, status=status, finished_at=time.time(), **fields)
        self.store.add_event(run_id, status, {k: v for k, v in fields.items() if k in ("error",)})

    def _run(self, run_id: str) -> None:
        run = self.store.get_run(run_id, with_result=False)
        if not run or run["status"] != "queued":
            return  # cancelled while waiting
        self.store.update_run(run_id, status="running", started_at=time.time())
        self.store.add_event(run_id, "status", {"status": "running"})
        try:
            result, usage = self.executor.run(
                run["options"], run["question"],
                lambda type_, data: self.store.add_event(run_id, type_, data),
                lambda: self._is_cancelled(run_id),
            )
            self.store.update_run(run_id, result=result, usage=usage)
            self._finish(run_id, "done")
        except Cancelled:
            self._finish(run_id, "cancelled")
        except BudgetExceeded as exc:
            self._finish(run_id, "stopped", error=f"{_mask(str(exc))}. Raise MAX_LLM_CALLS in .env, or ask for fewer dimensions.")
        except Exception as exc:  # the run failed, the server must not
            self._finish(run_id, "failed", error=_mask(f"{type(exc).__name__}: {exc}"))


def _mask(text: str) -> str:
    """Never show an API key in an error message."""
    for name in ("GEMINI_API_KEY", "XAI_API_KEY", "TAVILY_API_KEY"):
        key = os.getenv(name, "")
        if len(key) >= 8:
            text = text.replace(key, "***")
    return text[:500]
