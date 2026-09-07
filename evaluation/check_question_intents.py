"""Diagnostic: what Intent does QueryEngine's IntentClassifier actually
assign to each question in our eval sets?

QueryEngine routes a query to a different combination of rewrite / HyDE /
multi-query / decomposition depending on its classified Intent (FACTUAL,
PROCEDURAL, COMPARISON, CHITCHAT -- see ragforge.query.Intent). That
routing happens automatically, per question, inside AdvancedRAGStrategy
whenever a "query_engine"-enabled variant runs collect_predictions.py --
there is nothing the evaluation harness itself needs to do to "invoke"
the right query method per question, since the system already decides
that for itself at the intent-classification step.

What the harness does NOT tell you on its own is whether our eval
questions actually exercise all the intent-routing branches worth
measuring. This script answers that empirically: it classifies every
question in every eval_questions*.json file with the exact same
IntentClassifier the real pipeline uses, and reports the Intent
distribution -- both overall and cross-tabulated against each question's
own "type" field (summary/detail/cross_paragraph), since those are two
different, independently-assigned taxonomies that do not necessarily
line up one-to-one.

Only needs the Anthropic API (one classify() call per question) -- no
Milvus, no Ollama, no indexing, so this runs even when the vector store
isn't up.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from ragforge.config import RagForgeConfig  # noqa: E402
from ragforge.llm import LlmService  # noqa: E402
from ragforge.query.intent_classifier import IntentClassifier  # noqa: E402

QUESTION_FILES = {
    "clinical (54Q)": "eval_questions.json",
    "ccby (14Q)": "eval_questions_ccby.json",
    "cdc (15Q)": "eval_questions_cdc_program_eval.json",
    "ragbench (15Q)": "eval_questions_ragbench.json",
}


def main() -> None:
    config = RagForgeConfig.load(str(PROJECT_ROOT / "config" / "stage2.yml"))
    llm = LlmService(config.llm)
    classifier = IntentClassifier(llm)

    grand_total = Counter()
    by_type_grand = Counter()

    for label, fname in QUESTION_FILES.items():
        path = PROJECT_ROOT / "evaluation" / fname
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        intent_counts = Counter()
        by_type = Counter()  # (question_type, intent) -> count

        print(f"\n=== {label} -- {fname} ===")
        for item in items:
            question = item["question"]
            qtype = item.get("type", "?")
            intent = classifier.classify(question)
            intent_counts[intent.value] += 1
            by_type[(qtype, intent.value)] += 1
            grand_total[intent.value] += 1
            by_type_grand[(qtype, intent.value)] += 1
            print(f"  [{intent.value:<10}] ({qtype:<14}) {question[:80]}")

        print(f"  --- {label} intent distribution: {dict(intent_counts)} ---")

    print("\n=== GRAND TOTAL across all 4 eval sets ===")
    print(dict(grand_total))

    print("\n=== question 'type' (our taxonomy) vs classified Intent (system's taxonomy) ===")
    types = sorted({t for (t, _i) in by_type_grand})
    intents = sorted({i for (_t, i) in by_type_grand})
    header = f"{'type':<16}" + "".join(f"{i:<12}" for i in intents)
    print(header)
    for t in types:
        row = f"{t:<16}" + "".join(f"{by_type_grand.get((t, i), 0):<12}" for i in intents)
        print(row)


if __name__ == "__main__":
    main()
