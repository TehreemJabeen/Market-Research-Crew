"""Compare sentiment scorers on hand-labelled snippets.

    python -m evals.sentiment_eval --backends llm hf --hf-models cardiffnlp/twitter-roberta-base-sentiment-latest

Each label is the stance toward the NAMED entity on the named dimension. The report shows accuracy, macro-F1,
accuracy on plain versus comparative snippets (where a text-only model cannot know who is favoured), seconds
and LLM calls, so you can see where the free local model is good enough and where the LLM is worth its cost.

`evals/sentiment_labels.json` holds SYNTHETIC starter examples. Replace them with real snippets from your own
runs, hand-labelled (about 50), before quoting any number.
"""
import argparse
import json
import time
from pathlib import Path

from app.config import get_settings
from app.deps import build_deps
from app.opinion import is_comparative
from app.tools.sentiment import HFScorer, LLMScorer, ScoreItem

HERE = Path(__file__).parent
LABELS = ["positive", "negative", "neutral"]


def accuracy(gold: list[str], pred: list[str]) -> float:
    return round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 3) if gold else 0.0


def macro_f1(gold: list[str], pred: list[str], labels: list[str] = LABELS) -> float:
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(f1s) / len(f1s), 3)


def evaluate(scorer, items: list[dict], entities: list[str]) -> dict:
    start = time.time()
    predicted = [p.label for p in scorer.score([ScoreItem(i["entity"], i["dimension"], i["text"]) for i in items])]
    seconds = round(time.time() - start, 1)
    gold = [i["label"] for i in items]
    comparative = [is_comparative(i["text"], entities) for i in items]

    def subset(flag: bool):
        pairs = [(g, p) for g, p, c in zip(gold, predicted, comparative) if c == flag]
        return accuracy([g for g, _ in pairs], [p for _, p in pairs]) if pairs else None

    return {
        "n": len(items), "accuracy": accuracy(gold, predicted), "macro_f1": macro_f1(gold, predicted),
        "accuracy_plain": subset(False), "accuracy_comparative": subset(True), "n_comparative": sum(comparative),
        "seconds": seconds,
    }


def summarize(results: dict[str, dict]) -> str:
    lines = ["| scorer | n | accuracy | macro-F1 | acc. plain | acc. comparative (n) | seconds | LLM calls |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n']} | {r['accuracy']} | {r['macro_f1']} | {r['accuracy_plain']} | "
            f"{r['accuracy_comparative']} ({r['n_comparative']}) | {r['seconds']} | {r.get('llm_calls', 0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["llm", "hf"], choices=["llm", "hf"])
    parser.add_argument("--hf-models", nargs="+", default=None, help="Hugging Face models to try (default: SENTIMENT_MODEL)")
    args = parser.parse_args()

    items = json.loads((HERE / "sentiment_labels.json").read_text())["items"]
    entities = sorted({i["entity"] for i in items})
    settings = get_settings()
    results: dict[str, dict] = {}

    if "llm" in args.backends:
        deps = build_deps(settings)
        before = deps.llm.usage()["llm_calls"]
        results["llm"] = evaluate(LLMScorer(deps.llm), items, entities)
        results["llm"]["llm_calls"] = deps.llm.usage()["llm_calls"] - before
    if "hf" in args.backends:
        for model in args.hf_models or [settings.sentiment_model]:
            try:
                results[f"hf:{model}"] = evaluate(HFScorer(model), items, entities)
            except ImportError as exc:
                print(f"skipping hf:{model}: {exc}")
    table = summarize(results)
    (HERE / "sentiment_results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
