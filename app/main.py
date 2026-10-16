"""CLI: python -m app.main "Compare Notion, Coda and ClickUp for small teams" """
import argparse
import re
import sys
from pathlib import Path

from app.config import get_settings
from app.deps import build_deps
from app.llm import BudgetExceeded, LLMError
from app.runner import run_crew


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent market research crew")
    parser.add_argument("question", help="the research question")
    parser.add_argument("--provider", choices=["gemini", "grok"], help="override LLM_PROVIDER")
    parser.add_argument("--dimensions", help="comma-separated dimensions you insist on, e.g. 'pricing,security'")
    parser.add_argument("--no-factcheck", action="store_true", help="disable the fact-checker (ablation)")
    parser.add_argument("--no-community", action="store_true", help="facts only: skip the Hacker News / Reddit opinions track")
    parser.add_argument("--sentiment-backend", choices=["hf", "llm"], help="override SENTIMENT_BACKEND")
    parser.add_argument("--out", default="outputs", help="folder for the report")
    args = parser.parse_args()

    settings = get_settings(
        provider=args.provider,
        enable_factchecker=False if args.no_factcheck else None,
        enable_community=False if args.no_community else None,
        sentiment_backend=args.sentiment_backend,
    )
    try:
        deps = build_deps(settings)
    except LLMError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2

    dimensions = [d.strip() for d in args.dimensions.split(",")] if args.dimensions else None
    try:
        final = run_crew(args.question, deps, dimensions)
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}. Raise MAX_LLM_CALLS in .env, or ask for fewer dimensions.", file=sys.stderr)
        return 3

    print("\n".join(final.get("trace", [])))
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.question.lower()).strip("-")[:60] or "report"
    path = out_dir / f"{slug}.md"
    path.write_text(final["report"], encoding="utf-8")
    print(f"\nReport written to {path}")
    print(f"Metrics: {final.get('metrics', {})}")
    print(f"Usage: {final['usage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())