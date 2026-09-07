"""Phase 2 of the RAGAS evaluation add-on: score collected predictions.

This script runs in its OWN throwaway virtual environment
(evaluation/.ragas_venv, see requirements-ragas.txt) -- completely separate
from the project's main .venv. It reads a pure-data predictions JSON
produced by collect_predictions.py and never imports the `ragforge`
package, so its pinned (slightly older) langchain/ragas dependencies can
never conflict with or downgrade anything the main pipeline relies on.

Setup (one time):
    python3 -m venv evaluation/.ragas_venv
    evaluation/.ragas_venv/bin/pip install -r evaluation/requirements-ragas.txt

Usage:
    evaluation/.ragas_venv/bin/python evaluation/run_ragas_eval.py

Requires:
  - ANTHROPIC_API_KEY in the project's .env (same key the main pipeline
    already uses -- RAGAS uses it as the "judge" LLM for faithfulness /
    answer_relevancy / context_precision / context_recall).
  - Ollama running at localhost:11434 with the same embedding model the
    main pipeline uses (used only by answer_relevancy, to embed text).

To remove this evaluation layer entirely: delete the whole `evaluation/`
folder, or just this file + requirements-ragas.txt + evaluation/.ragas_venv.
Nothing outside evaluation/ is ever touched, so removal is always safe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OllamaEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Score RagForge predictions with RAGAS.")
    parser.add_argument(
        "--predictions",
        default=str(PROJECT_ROOT / "evaluation" / "reports" / "predictions.json"),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "evaluation" / "reports" / "ragas_report.json"),
    )
    parser.add_argument(
        "--judge-model",
        default="claude-haiku-4-5",
        help="Anthropic model used as the RAGAS judge (same one stage1.yml uses by default).",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model, must already be pulled (`ollama list`).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY not found. Check that evaluation/run_ragas_eval.py "
            "can see the project's .env (it looks in the project root)."
        )

    with open(args.predictions, "r", encoding="utf-8") as f:
        raw_predictions = json.load(f)

    # max_tokens matters here: RAGAS's own faithfulness/answer_relevancy
    # metrics make multi-step structured-JSON calls (statement extraction,
    # NLI verdicts) that can run longer than ChatAnthropic's low default
    # max_tokens, especially when the retrieved context is long. Too low a
    # cap truncates the judge's output mid-JSON and shows up as
    # LLMDidNotFinishException.
    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=args.judge_model, max_tokens=4096))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embedding_model))

    # faithfulness / answer_relevancy work without a human-written reference
    # answer; context_precision / context_recall need one. Only turn those
    # two on once every item in eval_questions.json has a "reference" filled
    # in -- otherwise ragas would error out on the missing column.
    has_reference = bool(raw_predictions) and all(item.get("reference") for item in raw_predictions)

    metrics = [faithfulness, answer_relevancy]
    if has_reference:
        metrics += [context_precision, context_recall]
    else:
        print(
            'Note: skipping context_precision/context_recall -- add "reference" '
            "answers to evaluation/eval_questions.json to enable them."
        )

    # Only pass RAGAS's own columns through -- collect_predictions.py now
    # also writes retrieved_chunk_ids/reference_chunk_ids/hit (for the
    # hitRate metric below), which RAGAS doesn't know about and doesn't
    # need, so keep them out of the EvaluationDataset it builds.
    RAGAS_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference"}
    dataset = EvaluationDataset.from_list(
        [
            {k: v for k, v in item.items() if k in RAGAS_COLUMNS and v is not None}
            for item in raw_predictions
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    per_sample_df = result.to_pandas()
    numeric_cols = per_sample_df.select_dtypes(include="number").columns
    averaged_scores = per_sample_df[numeric_cols].mean(numeric_only=True).to_dict()

    # hitRate is a deterministic IR metric computed by collect_predictions.py
    # (did retrieval surface a question's reference_chunk_id at all) -- it
    # doesn't go through RAGAS or the judge LLM, so fold it in separately
    # from the RAGAS metrics above instead of asking `evaluate()` for it.
    hits = [item["hit"] for item in raw_predictions if item.get("hit") is not None]
    if hits:
        averaged_scores["hit_rate"] = sum(hits) / len(hits)
    else:
        print(
            'Note: skipping hit_rate -- no question in eval_questions.json has '
            '"reference_chunk_ids" filled in yet. Run evaluation/list_chunks.py '
            "to pick them."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "averaged_scores": averaged_scores,
        "per_sample": per_sample_df.to_dict(orient="records"),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== RAGAS scores (averaged across all questions) ===")
    for name, score in averaged_scores.items():
        print(f"  {name}: {score:.4f}")
    print(f"\nFull per-question report written to {out_path}")


if __name__ == "__main__":
    main()
