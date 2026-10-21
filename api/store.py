"""SQLite storage for runs and their event log (standard library only)."""
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, question TEXT NOT NULL, options TEXT NOT NULL, status TEXT NOT NULL,
  created_at REAL NOT NULL, started_at REAL, finished_at REAL, error TEXT, usage TEXT, result TEXT
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);
"""
TERMINAL = ("done", "failed", "stopped", "cancelled")


class Store:
    def __init__(self, path: str = "runs.db"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # ---- runs ----------------------------------------------------------------------------------

    def create_run(self, run_id: str, question: str, options: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO runs (id, question, options, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
                (run_id, question, json.dumps(options), time.time()),
            )
            self._db.commit()

    def update_run(self, run_id: str, **fields) -> None:
        if not fields:
            return
        for key in ("usage", "result"):
            if key in fields and fields[key] is not None:
                fields[key] = json.dumps(fields[key], default=str)
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))
            self._db.commit()

    def get_run(self, run_id: str, with_result: bool = True) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row(row, with_result) if row else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, with_result=False) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            self._db.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            cur = self._db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self._db.commit()
        return cur.rowcount > 0

    def mark_orphans_failed(self) -> int:
        """Runs still 'queued' or 'running' when the server starts were interrupted by a restart."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE runs SET status = 'failed', finished_at = ?, error = 'The server restarted while this run was active.' "
                "WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
            self._db.commit()
        return cur.rowcount

    # ---- events --------------------------------------------------------------------------------

    def add_event(self, run_id: str, type_: str, data: dict) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?", (run_id,)).fetchone()
            seq = row[0]
            self._db.execute(
                "INSERT INTO events (run_id, seq, ts, type, data) VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, time.time(), type_, json.dumps(data, default=str)),
            )
            self._db.commit()
        return seq

    def events_after(self, run_id: str, seq: int = 0) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, ts, type, data FROM events WHERE run_id = ? AND seq > ? ORDER BY seq", (run_id, seq)
            ).fetchall()
        return [{"seq": r["seq"], "ts": r["ts"], "type": r["type"], "data": json.loads(r["data"])} for r in rows]

    # ---- helpers -------------------------------------------------------------------------------

    @staticmethod
    def _row(row: sqlite3.Row, with_result: bool) -> dict:
        usage = json.loads(row["usage"]) if row["usage"] else None
        out = {
            "id": row["id"], "question": row["question"], "options": json.loads(row["options"]), "status": row["status"],
            "created_at": row["created_at"], "finished_at": row["finished_at"], "error": row["error"], "usage": usage,
            "seconds": (usage or {}).get("seconds"), "llm_calls": (usage or {}).get("llm_calls"),
            "tavily_calls": (usage or {}).get("tavily_calls"),
        }
        if with_result:
            out["result"] = json.loads(row["result"]) if row["result"] else None
        return out
