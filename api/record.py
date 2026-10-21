"""Record one run so the UI can replay it (DEMO_MODE=1).

    python -m api.record "Compare Notion, Coda and ClickUp for small teams"      # a real run (needs API keys)
    python -m api.record --fake                                                   # synthetic sample data, no keys
"""
import argparse
import json
import sys
import time
from pathlib import Path

from api.executors import DEMO_FILE, RealExecutor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Compare Alpha and Beta for small teams")
    parser.add_argument("--fake", action="store_true", help="use the scripted test doubles (sample data, no keys)")
    parser.add_argument("--out", default=str(DEMO_FILE))
    args = parser.parse_args()

    executor = RealExecutor()
    if args.fake:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tests.fakes import make_fake_deps

        executor = RealExecutor(deps_factory=lambda settings: make_fake_deps(app_mode=settings.app_mode))
    events, last = [], time.time()

    def emit(type_: str, data: dict) -> None:
        nonlocal last
        now = time.time()
        events.append({"type": type_, "data": data, "delay": 0.7 if args.fake else round(min(now - last, 3.0), 2)})
        last = now

    result, usage = executor.run({"apps": "auto"}, args.question, emit, lambda: False)
    Path(args.out).write_text(
        json.dumps({"question": args.question, "sample": args.fake, "events": events, "result": result, "usage": usage}, indent=1),
        encoding="utf-8",
    )
    print(f"Recorded {len(events)} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
