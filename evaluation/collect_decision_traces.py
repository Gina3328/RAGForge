"""Phase 1b of the evaluation add-on: collect per-question DECISION traces
for the Stage 3/4 strategies that make a score-based routing decision --
on top of the final answer collect_predictions.py already captures.

Why this exists (and why collect_predictions.py + RAGAS aren't enough):
RAGAS scores answer quality (faithfulness, relevancy, hit-rate). It has no
way to see the *internal* decision a strategy made on the way to that
answer -- CRAG's computed confidence score, which complexity tier Adaptive
routed a query to, how many rounds Agentic's loop actually ran, whether
Self-RAG decided a query needed retrieval at all. Those values only exist
as local state inside each strategy's execute() call; they never reach the
final GenerationResult. This script captures them by wrapping each
strategy's decision method with a "spy" (records what it returns, then
calls straight through to the original -- never changes behavior), so the
trace reflects exactly what execute() itself saw and acted on.

Design mirrors collect_predictions.py on purpose: same environment (this
script runs inside the normal RagForge .venv, imports `ragforge` as-is,
adds zero new dependencies), same self-contained/removable placement under
evaluation/, same eval_questions.json-shaped input. It only differs in
what it records per question. Reuses the exact builders cli_stage3.py /
cli_stage4.py use (build_self_rag_strategy, build_crag_strategy,
build_adaptive_strategy, build_agentic_strategy) so this can never
silently drift from what the real CLIs actually construct.

Usage (from the project root, e.g. via `uv run`):

    uv run python evaluation/collect_decision_traces.py \\
        --strategy crag --document evaluation/documents/cdc_program_evaluation_framework_2024.pdf \\
        --questions evaluation/eval_questions_cdc_program_eval.json

    uv run python evaluation/collect_decision_traces.py \\
        --strategy agentic --document evaluation/documents/cdc_program_evaluation_framework_2024.pdf \\
        --questions evaluation/eval_questions_cdc_program_eval.json \\
        --label post_fix

Each run writes evaluation/reports/decision_traces_<strategy>[_<label>].json.
Run once per strategy (crag / adaptive / self_rag / agentic) per document.
For a before/after comparison (e.g. pre-fix vs post-fix agentic_rag_strategy.py),
run twice against the two git revisions and give each run a different
--label so the output files don't clobber each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Make the existing `ragforge` package importable without installing
# anything new or touching pyproject.toml / the project's .venv -- same
# trick collect_predictions.py uses.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from ragforge.chunker.chunker_factory import create_chunker  # noqa: E402
from ragforge.cli_stage2 import build_advanced_strategy, build_generator  # noqa: E402 (advanced_strategy unused directly but kept for parity with the CLIs' wiring order)
from ragforge.cli_stage3 import (  # noqa: E402
    build_adaptive_strategy,
    build_crag_strategy,
    build_self_rag_strategy,
)
from ragforge.cli_stage4 import build_agentic_strategy  # noqa: E402
from ragforge.config import RagForgeConfig  # noqa: E402
from ragforge.embedding.embedding_service import EmbeddingService  # noqa: E402
from ragforge.generator.simple_generator import SimpleGenerator  # noqa: E402
from ragforge.llm.llm_service import LlmService  # noqa: E402
from ragforge.parser.parser_router import ParserRouter  # noqa: E402
from ragforge.pipeline.rag_pipeline import RAGPipeline  # noqa: E402
from ragforge.query.query_engine import QueryEngine  # noqa: E402
from ragforge.retriever.dense_retriever import DenseRetriever  # noqa: E402
from ragforge.retriever.hybrid_retriever import HybridRetriever  # noqa: E402
from ragforge.store.milvus_vector_store import MilvusVectorStore  # noqa: E402

STRATEGY_CHOICES = ["self_rag", "crag", "adaptive", "agentic"]


def build_everything(
    config: RagForgeConfig,
    crag_confidence_source: str = "dense",
    agentic_merge_source: str = "rrf",
    agentic_dense_min_score: float = 0.0,
    agentic_window_order: str = "score",
    agentic_window_chunks: int = 10,
    agentic_window_chars: int = 200,
    agentic_window_show_score: str = "none",
):
    """Build every collaborator + every Stage 3/4 strategy, exactly the
    way cli_stage3.py / cli_stage4.py's main() does. Returns a dict with
    everything a run_<strategy>() function below might need.

    crag_confidence_source: research-only override threaded through to
    build_crag_strategy() -- "dense" (default, matches production) or
    "hybrid" (RRF-fused, for the RRF-vs-dense confidence comparison).

    agentic_merge_source / agentic_dense_min_score: research-only overrides
    threaded through to build_agentic_strategy() -- "rrf" (default, the
    original cross-collection merge) or "dense" (raw cosine score + a
    minimum-score filter), for the RRF-vs-dense cross-collection comparison.
    """
    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)
    parser_router = ParserRouter()
    chunker = create_chunker(config.chunker, embedding_service)
    llm_service = LlmService(config.llm)

    naive_pipeline = RAGPipeline(
        parser_router,
        chunker,
        embedding_service,
        vector_store,
        DenseRetriever(embedding_service, vector_store, config.retriever),
        SimpleGenerator(config.llm),
    )

    dense_retriever = DenseRetriever(embedding_service, vector_store, config.retriever)
    hybrid_retriever = HybridRetriever(embedding_service, vector_store, config.retriever)
    query_engine = QueryEngine.from_llm(llm_service) if config.query_engine.enabled else None
    shared_generator = build_generator(config, llm_service)

    self_rag_strategy = build_self_rag_strategy(
        config, hybrid_retriever, shared_generator, llm_service, query_engine
    )
    crag_strategy = build_crag_strategy(
        config, dense_retriever, hybrid_retriever, shared_generator, query_engine,
        confidence_source=crag_confidence_source,
    )
    adaptive_strategy = build_adaptive_strategy(
        config, llm_service, dense_retriever, hybrid_retriever, shared_generator, query_engine, self_rag_strategy
    )
    agentic_strategy = build_agentic_strategy(
        config, dense_retriever, hybrid_retriever, shared_generator, llm_service, query_engine, vector_store,
        merge_score_source=agentic_merge_source,
        dense_min_score=agentic_dense_min_score,
        window_order=agentic_window_order,
        window_chunks=agentic_window_chunks,
        window_chars=agentic_window_chars,
        window_show_score=agentic_window_show_score,
    )

    return {
        "naive_pipeline": naive_pipeline,
        "self_rag": self_rag_strategy,
        "crag": crag_strategy,
        "adaptive": adaptive_strategy,
        "agentic": agentic_strategy,
        "vector_store": vector_store,
    }


# --- Per-strategy tracing -----------------------------------------------
#
# Each run_<strategy>() wraps the exact internal method(s) that make the
# strategy's routing decision with a "spy": a function that records
# whatever the original call returns, then returns that same value
# unchanged. strategy.execute() is called normally underneath, so this
# never alters what actually happens -- it only observes it.


def run_crag(strategy, query: str, collection: str) -> tuple[object, dict]:
    trace: dict = {}
    original = strategy._evaluate_confidence

    def spy(result):
        confidence = original(result)
        trace["confidence"] = confidence
        return confidence

    with patch.object(strategy, "_evaluate_confidence", side_effect=spy):
        gen_result = strategy.execute(query, collection)

    confidence = trace.get("confidence")
    if confidence is None:
        trace["route"] = "unknown"
    elif confidence >= strategy.high_threshold:
        trace["route"] = "direct_generate"  # CRAG's "Correct" path
    elif confidence >= strategy.low_threshold:
        trace["route"] = "supplemental_retrieval"  # CRAG's "Ambiguous" path
    else:
        trace["route"] = "decline"  # CRAG's "Incorrect" path
    trace["high_threshold"] = strategy.high_threshold
    trace["low_threshold"] = strategy.low_threshold
    return gen_result, trace


def run_adaptive(strategy, query: str, collection: str) -> tuple[object, dict]:
    trace: dict = {}
    original = strategy._assess_complexity

    def spy(q):
        complexity = original(q)
        trace["complexity"] = complexity.value
        return complexity

    with patch.object(strategy, "_assess_complexity", side_effect=spy):
        gen_result = strategy.execute(query, collection)

    return gen_result, trace


def run_self_rag(strategy, query: str, collection: str) -> tuple[object, dict]:
    trace: dict = {"relevance_checks": []}
    original_should = strategy._should_retrieve
    original_relevant = strategy._is_relevant

    def spy_should(q):
        result = original_should(q)
        trace["should_retrieve"] = result
        return result

    def spy_relevant(q, result):
        verdict = original_relevant(q, result)
        trace["relevance_checks"].append(verdict)
        return verdict

    with patch.object(strategy, "_should_retrieve", side_effect=spy_should), patch.object(
        strategy, "_is_relevant", side_effect=spy_relevant
    ):
        gen_result = strategy.execute(query, collection)

    trace["retries_used"] = max(0, len(trace["relevance_checks"]) - 1)
    return gen_result, trace


def run_agentic(strategy, query: str, collection: str) -> tuple[object, dict]:
    trace: dict = {"rounds": []}
    original = strategy._decide_action

    def spy(original_query, chunks, iteration, recent_chunks=None, used_queries=None):
        decision = original(original_query, chunks, iteration, recent_chunks, used_queries)
        trace["rounds"].append(
            {
                "iteration": iteration,
                "action": decision.action.value,
                "new_query": decision.new_query,
                "chunks_seen": len(chunks),
                # What the decision prompt's chunk window contained
                # (size, per-collection counts, ordering, chunk/char limits).
                "window": dict(getattr(strategy, "last_window", {}) or {}),
            }
        )
        return decision

    with patch.object(strategy, "_decide_action", side_effect=spy):
        gen_result = strategy.execute(query, collection)

    trace["num_rounds"] = len(trace["rounds"])
    # Per-retrieval-call, per-collection candidate/kept counts (see
    # AgenticRAGStrategy.retrieval_log) -- shows how much of each round's
    # pool came from which collection.
    trace["retrieval_log"] = list(getattr(strategy, "retrieval_log", []))
    return gen_result, trace


RUNNERS = {
    "self_rag": run_self_rag,
    "crag": run_crag,
    "adaptive": run_adaptive,
    "agentic": run_agentic,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-question decision traces for one Stage 3/4 strategy."
    )
    parser.add_argument("--strategy", required=True, choices=STRATEGY_CHOICES)
    parser.add_argument("--document", required=True, help="Path to the document to index before querying.")
    parser.add_argument("--questions", required=True, help="Path to an eval_questions_*.json file.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "stage4.yml"),
        help="Defaults to config/stage4.yml (has every Stage 3/4 setting; stage3.yml would also work "
        "for every strategy except agentic).",
    )
    parser.add_argument("--label", default=None, help="Optional suffix for the output filename, e.g. 'post_fix'.")
    parser.add_argument(
        "--crag-confidence-source",
        choices=["dense", "hybrid"],
        default="dense",
        help="Research-only override for CRAG's confidence signal: 'dense' (default, current production "
        "design) or 'hybrid' (RRF-fused score, matching an earlier/Java design) -- for the RRF-vs-dense "
        "confidence comparison. Only affects --strategy crag; ignored for every other strategy.",
    )
    parser.add_argument(
        "--agentic-merge-source",
        choices=["rrf", "dense"],
        default="rrf",
        help="Research-only override for Agentic RAG's cross-collection merge: 'rrf' (default, original "
        "behavior: per-collection top-5 by RRF score) or 'dense' (raw cosine score, with "
        "--agentic-dense-min-score as a minimum). Only affects --strategy agentic.",
    )
    parser.add_argument(
        "--agentic-dense-min-score",
        type=float,
        default=0.0,
        help="Minimum raw dense cosine score a chunk needs to survive the cross-collection merge when "
        "--agentic-merge-source dense. Ignored in rrf mode. Pick it with evaluation/calibrate_dense_threshold.py.",
    )
    parser.add_argument(
        "--agentic-window-order",
        choices=["score", "dense"],
        default="score",
        help="Research-only: how the chunk window shown to the Agent's GENERATE/RETRIEVE decision is "
        "ordered. 'score' (default, original): by .score (RRF in rrf merge mode); 'dense': by raw dense cosine.",
    )
    parser.add_argument("--agentic-window-chunks", type=int, default=10,
                        help="Research-only: how many chunks the decision prompt shows (default 10, original).")
    parser.add_argument("--agentic-window-chars", type=int, default=200,
                        help="Research-only: leading characters shown per chunk in the decision prompt (default 200, original).")
    parser.add_argument("--agentic-window-show-score", choices=["none", "rrf", "dense"], default="none",
                        help="Research-only: prefix each chunk in the decision prompt with its score "
                        "(rrf = .score at 4 dp, dense = raw cosine at 2 dp; default none, original).")
    parser.add_argument("--out", default=None, help="Defaults to evaluation/reports/decision_traces_<strategy>[_<label>].json.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = f"_{args.label}" if args.label else ""
        out_path = PROJECT_ROOT / "evaluation" / "reports" / f"decision_traces_{args.strategy}{suffix}.json"

    config = RagForgeConfig.load(args.config)
    built = build_everything(
        config,
        crag_confidence_source=args.crag_confidence_source,
        agentic_merge_source=args.agentic_merge_source,
        agentic_dense_min_score=args.agentic_dense_min_score,
        agentic_window_order=args.agentic_window_order,
        agentic_window_chunks=args.agentic_window_chunks,
        agentic_window_chars=args.agentic_window_chars,
        agentic_window_show_score=args.agentic_window_show_score,
    )
    strategy = built[args.strategy]
    run_fn = RUNNERS[args.strategy]

    print(f"Strategy: {args.strategy}")
    print(f"Indexing {args.document} ...")
    collection = built["naive_pipeline"].index_document(args.document)
    print(f"Indexed. Collection: {collection}")

    try:
        store_collections = built["vector_store"].list_collections()
    except Exception:
        store_collections = None
    print(f"Collections in the vector store: {store_collections}")

    with open(args.questions, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    records = []
    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        print(f"[{i}/{len(eval_items)}] {question}")
        gen_result, trace = run_fn(strategy, question, collection)
        records.append(
            {
                "question": question,
                "type": item.get("type"),
                "reference": item.get("reference"),
                # None marks a deliberately unanswerable question (no
                # "reference" key at all) -- correct behavior there is
                # declining/abstaining, not producing any specific text.
                "answerable": "reference" in item and item["reference"] is not None,
                "answer": gen_result.answer,
                # Captured the same way collect_predictions.py does it, so
                # convert_traces_to_predictions.py can turn this record into
                # a RAGAS-scorable prediction (RAGAS's faithfulness /
                # answer_relevancy / context_precision / context_recall all
                # need the actual retrieved text, not just the final answer).
                "retrieved_contexts": [c.content for c in gen_result.retrieved_chunks],
                "trace": trace,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": args.strategy,
                "document": args.document,
                "collection": collection,
                # Provenance: what was in the vector store and which
                # research-only settings were active for this run.
                "vector_store_collections": store_collections,
                "settings": {
                    "agentic_merge_source": args.agentic_merge_source,
                    "agentic_dense_min_score": args.agentic_dense_min_score,
                    "agentic_window_order": args.agentic_window_order,
                    "agentic_window_chunks": args.agentic_window_chunks,
                    "agentic_window_chars": args.agentic_window_chars,
                    "agentic_window_show_score": args.agentic_window_show_score,
                    "crag_confidence_source": args.crag_confidence_source,
                    "score_threshold": config.retriever.score_threshold,
                },
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {len(records)} decision traces to {out_path}")


if __name__ == "__main__":
    main()
