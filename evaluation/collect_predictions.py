"""Phase 1 of the RAGAS evaluation add-on: run the RagForge pipeline over a
set of eval questions and dump (question, answer, retrieved contexts) to a
plain JSON file.

This script runs INSIDE the normal RagForge environment (the same `.venv`
you already use to run `cli.py` / `cli_stage2.py`) and only imports the
`ragforge` package as-is -- it adds zero new dependencies and never touches
any file outside `evaluation/`. Safe to delete the whole `evaluation/`
folder at any time without affecting the rest of the project.

--- Ablation mode (--variant) ---

Originally this script only knew how to run the Stage 1 baseline pipeline
(`ragforge.cli.build_pipeline`). It now also supports running an
incremental ablation study: each --variant after "baseline" cumulatively
turns ON one more Stage 2 feature on top of the previous row, using the
exact same component-selection logic `cli_stage2.py` uses (imported from
there, not reimplemented -- see build_runner() below), so this script can
never silently drift from what cli_stage2.py actually builds:

    baseline      Stage 1: fixed_size chunking, DenseRetriever, SimpleGenerator
    chunking      + ParentChild chunking
    hybrid        + Hybrid (dense+sparse) retrieval
    reranking     + Cross-Encoder reranking
    query_engine  + query engine (rewrite + HyDE + multi-query/decomposition)
    generation    + citation generation + hallucination detection

Two factors that might otherwise be tracked separately are NOT separable
with this codebase as it stands, so this script is honest about collapsing
them rather than faking a split:

  1. HyDE and multi-query generation can't be isolated from each other.
     QueryEngine classifies intent, then always runs rewrite+HyDE together,
     then always runs either multi-query or decomposition depending on that
     intent -- it never actually branches on its own enable_rewrite/
     enable_hyde/enable_multi_query/enable_decomposition config flags
     (those flags exist on QueryEngineConfig but nothing inside
     QueryEngine.process_query reads them). So there's no way to turn on
     HyDE without also turning on rewrite and multi-query/decomposition --
     they're bundled into a single "query_engine" variant here. Splitting
     them apart for real would mean changing QueryEngine.process_query
     itself to honor those flags, which is out of scope for an evaluation
     script.

  2. Context compression has no on/off switch in GeneratorConfig at all
     (cli_stage2.py's build_advanced_strategy always builds a
     ContextCompressor) -- so it's a constant background factor present in
     every non-baseline variant here, not something this ablation isolates.

Usage (from the project root, e.g. via PyCharm's terminal or `uv run`):

    uv run python evaluation/collect_predictions.py \\
        --document /path/to/your/document.md --variant baseline

    uv run python evaluation/collect_predictions.py \\
        --document /path/to/your/document.md --variant hybrid

Each variant writes to its own predictions file by default
(evaluation/reports/predictions_<variant>.json, except "baseline" which
keeps the original plain predictions.json name for backwards compatibility
with the report already sitting in evaluation/reports/), so running every
variant in turn doesn't clobber earlier results -- feed each one into
run_ragas_eval.py's --predictions/--out flags to build up the full table.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the existing `ragforge` package importable without installing
# anything new or touching pyproject.toml / the project's .venv.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from ragforge.chunker.chunker_factory import create_chunker  # noqa: E402
from ragforge.cli import build_pipeline  # noqa: E402 (Stage 1 baseline, reused as-is)
from ragforge.cli_stage2 import (  # noqa: E402 (Stage 2 wiring, reused as-is)
    build_advanced_strategy,
    build_generator,
    build_retriever,
)
from ragforge.config import RagForgeConfig  # noqa: E402
from ragforge.embedding.embedding_service import EmbeddingService  # noqa: E402
from ragforge.generator.generation_result import GenerationResult  # noqa: E402
from ragforge.llm.llm_service import LlmService  # noqa: E402
from ragforge.parser.parser_router import ParserRouter  # noqa: E402
from ragforge.pipeline.rag_pipeline import RAGPipeline  # noqa: E402
from ragforge.store.milvus_vector_store import MilvusVectorStore  # noqa: E402


@dataclass
class Variant:
    """One row of the incremental ablation study.

    Each field mirrors one of AdvancedRAGStrategy's config-driven choices.
    Every variant after "baseline" carries forward the previous row's
    settings plus exactly one change, so the difference between two
    adjacent variants' scores isolates that one feature's effect.
    """

    label: str
    chunker_strategy: str
    retriever_strategy: str
    reranker_enabled: bool
    query_engine_enabled: bool
    enable_citation: bool
    enable_hallucination_detection: bool


# Insertion order here IS the ablation's row order (baseline through the
# fully-featured pipeline, with HyDE and multi-query merged -- see the
# module docstring for why) -- argparse's --variant choices and help text
# below both rely on that order.
VARIANTS: dict[str, Variant] = {
    "baseline": Variant(
        label="Baseline (Stage 1)",
        chunker_strategy="fixed_size",
        retriever_strategy="dense",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "chunking": Variant(
        label="+ Chunking (ParentChild)",
        chunker_strategy="parent_child",
        retriever_strategy="dense",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "hybrid": Variant(
        label="+ Hybrid retrieval (Dense+Sparse)",
        chunker_strategy="parent_child",
        retriever_strategy="hybrid",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "reranking": Variant(
        label="+ Reranking",
        chunker_strategy="parent_child",
        retriever_strategy="hybrid",
        reranker_enabled=True,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "query_engine": Variant(
        label="+ Query engine (rewrite + HyDE + multi-query/decomposition, bundled)",
        chunker_strategy="parent_child",
        retriever_strategy="hybrid",
        reranker_enabled=True,
        query_engine_enabled=True,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "generation": Variant(
        label="+ Citation + Hallucination detection",
        chunker_strategy="parent_child",
        retriever_strategy="hybrid",
        reranker_enabled=True,
        query_engine_enabled=True,
        enable_citation=True,
        enable_hallucination_detection=True,
    ),
    # --- Chunker-isolation variants (not part of the cumulative chain above) ---
    # These three hold every setting at baseline's values (dense retriever,
    # no reranker, no query engine, no citation/hallucination detection) and
    # only swap the chunker strategy -- so, together with "baseline"
    # (fixed_size) and "chunking" (parent_child), they give a clean 5-way
    # head-to-head comparison of every chunker this project implements,
    # isolated from every other pipeline change.
    "chunking_recursive": Variant(
        label="[Chunker isolation] Recursive",
        chunker_strategy="recursive",
        retriever_strategy="dense",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "chunking_semantic": Variant(
        label="[Chunker isolation] Semantic",
        chunker_strategy="semantic",
        retriever_strategy="dense",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
    "chunking_structure_aware": Variant(
        label="[Chunker isolation] StructureAware",
        chunker_strategy="structure_aware",
        retriever_strategy="dense",
        reranker_enabled=False,
        query_engine_enabled=False,
        enable_citation=False,
        enable_hallucination_detection=False,
    ),
}


def build_runner(variant_name: str, config: RagForgeConfig):
    """Build everything needed to index + query for one variant.

    Returns (index_fn, query_fn):
      index_fn(document_path: str) -> collection name
      query_fn(collection: str, question: str) -> GenerationResult

    Both variants end up presenting this exact same two-callable interface
    so main() below never needs to know which strategy is underneath.
    """
    variant = VARIANTS[variant_name]

    if variant_name == "baseline":
        # Reuse Stage 1's build_pipeline exactly as before -- keeps this
        # variant's numbers comparable to the baseline run already sitting
        # in evaluation/reports/ragas_report.json.
        pipeline = build_pipeline(config)
        return pipeline.index_document, pipeline.query

    # Every other variant applies this row's cumulative feature set onto a
    # RagForgeConfig loaded from stage2.yml, then wires it up exactly the
    # way cli_stage2.py's build_* functions do -- reused here, not
    # reimplemented, so this script can never silently drift from what
    # cli_stage2.py actually builds.
    config.chunker.strategy = variant.chunker_strategy
    config.retriever.strategy = variant.retriever_strategy
    config.reranker.enabled = variant.reranker_enabled
    config.query_engine.enabled = variant.query_engine_enabled
    config.generator.enable_citation = variant.enable_citation
    config.generator.enable_hallucination_detection = variant.enable_hallucination_detection

    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)
    parser_router = ParserRouter()
    chunker = create_chunker(config.chunker, embedding_service)
    llm_service = LlmService(config.llm)

    retriever = build_retriever(config, embedding_service, vector_store)
    generator = build_generator(config, llm_service)
    strategy = build_advanced_strategy(config, retriever, generator, llm_service)

    # Only used for indexing -- indexing never touches retriever/generator,
    # so reusing this variant's "advanced" retriever/generator here (rather
    # than a fresh Stage 1 pair) is harmless.
    indexing_pipeline = RAGPipeline(
        parser_router, chunker, embedding_service, vector_store, retriever, generator
    )

    def query_fn(collection: str, question: str) -> GenerationResult:
        # AdvancedRAGStrategy.execute takes (query, collection) -- the
        # opposite argument order from RAGPipeline.query(collection, query)
        # -- flipped here so both variants present the same query_fn shape.
        return strategy.execute(question, collection)

    return indexing_pipeline.index_document, query_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect RagForge predictions for later RAGAS scoring -- one row "
            "of the Stage 2 incremental ablation study per run."
        )
    )
    parser.add_argument(
        "--document", required=True, help="Path to the document to index before querying."
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="baseline",
        help="Which ablation row to run: " + "; ".join(f"{k}={v.label}" for k, v in VARIANTS.items()),
    )
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "evaluation" / "eval_questions.json"),
        help="Path to the eval question set (see evaluation/eval_questions.json).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Defaults to config/stage1.yml for --variant baseline, config/stage2.yml otherwise.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Defaults to evaluation/reports/predictions.json for --variant baseline "
            "(unchanged from before, for backwards compatibility), "
            "evaluation/reports/predictions_<variant>.json otherwise."
        ),
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    config_path = args.config or str(
        PROJECT_ROOT / "config" / ("stage1.yml" if args.variant == "baseline" else "stage2.yml")
    )
    if args.out:
        out_path = Path(args.out)
    elif args.variant == "baseline":
        out_path = PROJECT_ROOT / "evaluation" / "reports" / "predictions.json"
    else:
        out_path = PROJECT_ROOT / "evaluation" / "reports" / f"predictions_{args.variant}.json"

    config = RagForgeConfig.load(config_path)
    index_fn, query_fn = build_runner(args.variant, config)

    variant = VARIANTS[args.variant]
    print(f"Variant: {args.variant} ({variant.label})")

    print(f"Indexing {args.document} ...")
    collection = index_fn(args.document)
    print(f"Indexed. Collection: {collection}")

    with open(args.questions, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    predictions = []
    hits: list[float] = []
    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        print(f"[{i}/{len(eval_items)}] {question}")
        result = query_fn(collection, question)

        # Chunk id isn't on a fixed attribute name in every version of the
        # pipeline's result objects (seen both `chunk_id` and `id` used
        # across the codebase) -- try both instead of assuming, so this
        # never hard-crashes a whole eval run over one naming mismatch.
        retrieved_chunk_ids = [
            getattr(c, "chunk_id", None) or getattr(c, "id", None) for c in result.retrieved_chunks
        ]

        reference_chunk_ids = item.get("reference_chunk_ids") or []
        hit = _hit_rate(reference_chunk_ids, retrieved_chunk_ids)
        if hit is not None:
            hits.append(hit)

        predictions.append(
            {
                "user_input": question,
                "response": result.answer,
                "retrieved_contexts": [c.content for c in result.retrieved_chunks],
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "reference_chunk_ids": reference_chunk_ids,
                # 1.0 if any reference_chunk_id was retrieved, 0.0 if none
                # were, None if this question has no reference_chunk_ids yet
                # (run evaluation/list_chunks.py to pick them).
                "hit": hit,
                # None if the question in eval_questions.json has no
                # human-written reference answer yet -- run_ragas_eval.py
                # will only enable context_precision/context_recall once
                # every item has one.
                "reference": item.get("reference"),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(predictions)} predictions to {out_path}")
    if hits:
        print(
            f"hitRate: {sum(hits) / len(hits):.4f} "
            f"({sum(hits):.0f}/{len(hits)} questions with reference_chunk_ids hit; "
            f"{len(predictions) - len(hits)} question(s) skipped -- no reference_chunk_ids set)"
        )
    else:
        print(
            "hitRate: n/a -- no question has \"reference_chunk_ids\" filled in yet. "
            "Run evaluation/list_chunks.py against your document, pick the chunk_id(s) "
            "that answer each question, and add them to eval_questions.json."
        )


def _hit_rate(reference_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float | None:
    """1.0 if at least one reference chunk was retrieved, 0.0 if none were,
    None if there's nothing to check (question has no reference_chunk_ids
    yet). Deterministic IR metric -- no LLM judge involved, unlike the
    RAGAS metrics in run_ragas_eval.py."""
    if not reference_chunk_ids:
        return None
    return 1.0 if any(rid in retrieved_chunk_ids for rid in reference_chunk_ids) else 0.0


if __name__ == "__main__":
    main()
