"""Run ONLY the planner on one question and print what it produced. Uses the LLM (3 calls), no Tavily.

    python -m evals.check_planner "Compare Notion, Coda and ClickUp for a 10-person team under $100/month"
    python -m evals.check_planner "Duolingo vs Babbel" --dims "pricing,offline use"
    python -m evals.check_planner "..." --provider groq
"""
import argparse
import sys

from app.agents.planner import run_planner
from app.config import get_settings
from app.llm import make_llm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--dims", default="", help="comma-separated dimensions you want forced in")
    ap.add_argument("--provider", choices=["gemini", "groq", "grok"], default=None)
    args = ap.parse_args()

    settings = get_settings(provider=args.provider)
    llm = make_llm(settings)

    class Deps:  # the planner only needs the llm and settings
        pass

    deps = Deps()
    deps.llm, deps.settings = llm, settings
    state = {"question": args.question, "user_dimensions": [d.strip() for d in args.dims.split(",") if d.strip()]}
    print(f"Running planner with {settings.provider} (3 model calls, can take 20-40 s)...", flush=True)
    out = run_planner(state, deps)
    plan, tasks = out["plan"], out["tasks"]

    print(f"\nQuestion        : {args.question}")
    print(f"Rewritten       : {plan.rewritten_question}")
    print(f"Audience        : {plan.audience or '-'}")
    print(f"Decision        : {plan.decision or '-'}")
    print(f"Products        : {', '.join(plan.entities) or '-'}")
    print(f"Is app          : {plan.is_app}")
    print("\nDimensions (most important first):")
    for i, d in enumerate(plan.dimensions, 1):
        print(f"  {i}. {d}  -  {plan.dimension_notes.get(d, '')}")
    print("\nSub-questions:")
    for t in tasks:
        print(f"  [{t.dimension}] {t.sub_question}")
    print("\nOutline (report columns):", ", ".join(plan.outline))
    print("\nTrace:")
    for line in out["trace"]:
        print(f"  {line}")
    print(f"\nLLM calls: {llm.usage()['llm_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())