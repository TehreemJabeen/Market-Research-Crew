"""Planner quality: of the dimensions an expert would use, how many does the planner find, and how many of
its own are on the expert list?   python -m evals.planner_eval --limit 3

Two modes show what scouting adds:
  scouted   planner with scouting searches (evidence-based)
  llm_only  planner from the model's own knowledge

Matching is done in code (word overlap plus a small synonym map), so it is cheap and reproducible but
imperfect: read the printed pairs before trusting a number.
"""
import argparse
import json
import re
import statistics
import time
from dataclasses import replace
from pathlib import Path

from app.agents.planner import run_planner
from app.config import get_settings
from app.deps import build_deps

HERE = Path(__file__).parent
MODES = {"scouted": True, "llm_only": False}
STOPWORDS = {"of", "and", "the", "for", "to", "a", "in", "on", "&", "vs"}
SYNONYMS = {
    "price": "pricing", "prices": "pricing", "cost": "pricing", "costs": "pricing", "plan": "pricing", "plans": "pricing",
    "feature": "features", "capabilities": "features", "functionality": "features",
    "integration": "integrations", "usability": "ease", "simplicity": "ease",
    "speed": "performance", "latency": "performance", "scaling": "scalability", "scalable": "scalability",
    "compliance": "security", "privacy": "security",
}


def _canon_tokens(name: str) -> set[str]:
    tokens = [t for t in re.sub(r"[^a-z0-9]+", " ", name.casefold()).split() if t not in STOPWORDS]
    return {SYNONYMS.get(t, t) for t in tokens}


def matches(a: str, b: str) -> bool:
    ta, tb = _canon_tokens(a), _canon_tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5


def dimension_scores(predicted: list[str], expert: list[str]) -> dict:
    """recall = share of expert dimensions the planner found; precision = share of planner dimensions on the list."""
    found = [e for e in expert if any(matches(p, e) for p in predicted)]
    useful = [p for p in predicted if any(matches(p, e) for e in expert)]
    return {
        "recall": round(len(found) / len(expert), 3) if expert else 0.0,
        "precision": round(len(useful) / len(predicted), 3) if predicted else 0.0,
        "missed": [e for e in expert if e not in found],
    }


def run_one(item: dict, expert: list[str], mode: str, base_settings) -> dict:
    deps = build_deps(replace(base_settings, enable_scout=MODES[mode]))
    start = time.time()
    try:
        out = run_planner({"question": item["question"]}, deps)
    except Exception as exc:
        return {"id": item["id"], "mode": mode, "error": str(exc)}
    predicted = out["plan"].dimensions
    return {
        "id": item["id"], "mode": mode, "dimensions": predicted, "n_dims": len(predicted),
        **dimension_scores(predicted, expert),
        "llm_calls": deps.llm.usage()["llm_calls"], "seconds": round(time.time() - start, 1),
    }


def summarize(rows: list[dict]) -> str:
    lines = ["| mode | runs | dimension recall | dimension precision | avg dimensions | avg LLM calls | avg seconds |",
             "|---|---|---|---|---|---|---|"]
    for mode in MODES:
        ok = [r for r in rows if r["mode"] == mode and "error" not in r]
        failed = sum(1 for r in rows if r["mode"] == mode and "error" in r)
        if not ok:
            lines.append(f"| {mode} | 0 ok / {failed} failed | - | - | - | - | - |")
            continue
        avg = lambda k: round(statistics.mean(r[k] for r in ok), 2)
        lines.append(f"| {mode} | {len(ok)}" + (f" ({failed} failed)" if failed else "")
                     + f" | {avg('recall')} | {avg('precision')} | {avg('n_dims')} | {avg('llm_calls')} | {avg('seconds')} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = json.loads((HERE / "questions.json").read_text())[: args.limit]
    expert = json.loads((HERE / "expert_dimensions.json").read_text())
    settings = get_settings()
    rows = []
    for item in items:
        if item["id"] not in expert:
            print(f"skipping {item['id']}: no expert dimensions")
            continue
        for mode in MODES:
            print(f"planning {item['id']} / {mode} ...", flush=True)
            rows.append(run_one(item, expert[item["id"]], mode, settings))
            print("  ->", rows[-1], flush=True)
    (HERE / "planner_results.json").write_text(json.dumps(rows, indent=2))
    table = summarize(rows)
    (HERE / "planner_results.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
