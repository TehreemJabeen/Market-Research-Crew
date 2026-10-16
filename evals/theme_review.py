"""Human review of the opinions track: are the themes faithful, and do you agree with the fact-to-evidence links?

    python -m evals.theme_review export "Compare Notion, Coda and ClickUp for small teams"
    # open evals/theme_review.csv and put y or n in the last column of each row
    python -m evals.theme_review score evals/theme_review.csv

Theme rows ask: do the example quotes really support this theme?
Link rows ask: do you agree with the relation the model assigned between the fact and the user theme?
Aim for about 20 reviewed rows of each kind; the two printed rates are the numbers to quote.
"""
import argparse
import csv
import sys
from pathlib import Path

from app.config import get_settings
from app.deps import build_deps
from app.runner import run_crew

HERE = Path(__file__).parent
FIELDS = ["kind", "item", "evidence", "model_says", "human_y_or_n"]


def review_rows(final: dict) -> list[dict]:
    """Build the review sheet from a finished run (pure: no LLM, no network)."""
    claims = {c.id: c for c in final.get("claims", [])}
    themes = {t.id: t for t in final.get("themes", [])}
    rows = []
    for t in themes.values():
        rows.append({
            "kind": "theme",
            "item": f"{t.entity} / {t.dimension}: {t.label}",
            "evidence": " | ".join(f'"{e.quote}" ({e.url})' for e in t.examples),
            "model_says": f"{t.stance}, {t.n_threads} threads",
            "human_y_or_n": "",
        })
    for link in final.get("links", []):
        if link.relation == "unrelated" or link.claim_id not in claims or link.theme_id not in themes:
            continue
        rows.append({
            "kind": "link",
            "item": f"FACT: {claims[link.claim_id].text}",
            "evidence": f"THEME: {themes[link.theme_id].label}",
            "model_says": f"{link.relation}" + (f" ({link.note})" if link.note else ""),
            "human_y_or_n": "",
        })
    return rows


def score_review(rows: list[dict]) -> dict[str, dict]:
    """Agreement rate per kind, ignoring rows you have not answered yet."""
    out: dict[str, dict] = {}
    for kind in ("theme", "link"):
        answers = [r["human_y_or_n"].strip().lower() for r in rows if r["kind"] == kind and r["human_y_or_n"].strip()]
        agree = sum(a.startswith("y") for a in answers)
        out[kind] = {"reviewed": len(answers), "agree": agree, "rate": round(agree / len(answers), 3) if answers else None}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export")
    export.add_argument("question")
    score = sub.add_parser("score")
    score.add_argument("csv_path")
    args = parser.parse_args()

    if args.cmd == "export":
        final = run_crew(args.question, build_deps(get_settings()))
        rows = review_rows(final)
        path = HERE / "theme_review.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}: fill the last column with y or n, then run the score command")
        return 0

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        result = score_review(list(csv.DictReader(f)))
    for kind, r in result.items():
        label = "theme faithfulness" if kind == "theme" else "link agreement"
        rate = "n/a (no rows answered)" if r["rate"] is None else r["rate"]
        print(f"{label}: {rate}  ({r['agree']}/{r['reviewed']} reviewed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
