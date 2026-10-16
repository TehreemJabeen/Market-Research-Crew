"""Compare four configurations on the same questions.

    python -m evals.run_evals --limit 2

crew               full workflow: facts track with the Fact-Checker + community opinions track
crew_no_factcheck  Fact-Checker switched off (ablation)
crew_no_community  facts only (ablation: what the opinions track adds)
single_agent       one search + one LLM call (baseline)

Metrics (computed in code, no LLM judge):
  citation_accuracy  share of citations whose quote really appears in the cited page or post (facts AND themes)
  coverage           share of `must_mention` terms present in the report (crude; extend with reference facts)
  themes / domains   recurring user themes found; distinct sites cited
  llm_calls / tokens / seconds   cost and latency
"""
import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from app.config import get_settings
from app.deps import build_deps
from app.runner import run_crew
from app.tools.verify import citation_accuracy
from evals.baseline import run_single_agent

MODES = ["crew", "crew_no_factcheck", "crew_no_community", "single_agent"]
HERE = Path(__file__).parent


def coverage(report: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    low = report.lower()
    return sum(t.lower() in low for t in terms) / len(terms)


def domains(citations: list[dict]) -> int:
    return len({urlparse(c["url"]).netloc for c in citations if c.get("url")})


def run_one(mode: str, item: dict, base_settings) -> dict:
    settings = replace(
        base_settings,
        enable_factchecker=(mode != "crew_no_factcheck"),
        enable_community=(mode != "crew_no_community"),
    )
    deps = build_deps(settings)
    start = time.time()
    try:
        res = run_single_agent(item["question"], deps) if mode == "single_agent" else run_crew(item["question"], deps)
    except Exception as exc:  # record the failure instead of aborting the whole eval
        return {"id": item["id"], "mode": mode, "error": str(exc)}
    usage = deps.llm.usage()
    return {
        "id": item["id"],
        "mode": mode,
        "citation_accuracy": round(citation_accuracy(res["final_citations"], res["sources"], settings.fuzzy_threshold), 3),
        "citations": len(res["final_citations"]),
        "coverage": round(coverage(res["report"], item.get("must_mention", [])), 3),
        "themes": len(res.get("themes", [])),
        "domains": domains(res["final_citations"]),
        "llm_calls": usage["llm_calls"],
        "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "seconds": round(time.time() - start, 1),
    }


def summarize(rows: list[dict]) -> str:
    lines = [
        "| mode | runs | citation accuracy | coverage | avg citations | avg themes | avg source sites | avg LLM calls | avg tokens | avg seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        ok = [r for r in rows if r["mode"] == mode and "error" not in r]
        failed = sum(1 for r in rows if r["mode"] == mode and "error" in r)
        if not ok:
            lines.append(f"| {mode} | 0 ok / {failed} failed | - | - | - | - | - | - | - | - |")
            continue
        avg = lambda key: round(statistics.mean(r[key] for r in ok), 2)
        lines.append(
            f"| {mode} | {len(ok)}" + (f" ({failed} failed)" if failed else "")
            + f" | {avg('citation_accuracy')} | {avg('coverage')} | {avg('citations')} | {avg('themes')} | {avg('domains')}"
            + f" | {avg('llm_calls')} | {avg('tokens')} | {avg('seconds')} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    args = parser.parse_args()

    items = json.loads((HERE / "questions.json").read_text())[: args.limit]
    settings = get_settings()
    rows = []
    for item in items:
        for mode in args.modes:
            print(f"running {item['id']} / {mode} ...", flush=True)
            rows.append(run_one(mode, item, settings))
            print("  ->", rows[-1], flush=True)

    (HERE / "results.json").write_text(json.dumps(rows, indent=2))
    table = summarize(rows)
    (HERE / "results.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
