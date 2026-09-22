"""Interactive command-line entry point for the RagForge Stage 4 pipeline.

This is the Stage 4 equivalent of cli_stage3.py: it reads
config/stage4.yml and wires up a *registry* of RAG strategies --
advanced (Stage 2), self_rag, crag, adaptive (Stage 3), and agentic
(Stage 4) -- keyed by name. "naive" stays a special case handled
directly through the Stage 1 RAGPipeline, the same way the earlier CLIs
already treat it.

Component builders shared with earlier stages (build_generator,
build_advanced_strategy from cli_stage2; build_self_rag_strategy,
build_crag_strategy, build_adaptive_strategy from cli_stage3) are
imported rather than duplicated, so every entry point stays driven by
the same config-based construction logic for the parts they share. Only
build_agentic_strategy is new here.

The one deliberate interaction-model change from Stage 3: there is no
"use <collection>" command. Agentic RAG is meant to search across every
collection on its own (see AgenticRAGStrategy._retrieve_all_collections),
so instead of asking the user to pick a collection up front, every
"query" command runs a quick single-shot dense retrieval against each
known collection, compares the average result score, and automatically
picks whichever collection looks most relevant to that specific
question before handing off to the selected strategy.

Known limitation: this auto-selection is only as reliable as the
embedding model's ability to score cross-lingual similarity. Testing
showed a Chinese question can score the one genuinely relevant
(English-language) collection *lowest* among several unrelated
collections, while the same question in English scores it clearly
highest -- the underlying embedding model's cross-lingual alignment is
weak enough that topical relevance gets swamped by noise when the
query and document languages differ. This affects every
single-collection strategy (advanced/self_rag/crag/adaptive/naive);
Agentic RAG is naturally immune to it, since it searches every
collection every round regardless of which one this quick scan picked.
Fixing this properly would mean either query-language-aware collection
selection or a stronger multilingual embedding model -- out of scope
for this pass, so for now cross-lingual queries against a
multi-collection store should be treated with caution, or verified
with "collections" plus the quick-scan scores this function prints.

Available commands:
    index <path>                    Index a document into the vector store
    query [strategy] <text>         Ask a question (default: current strategy)
    strategy [name]                 Show or switch the default strategy
    chunk <strategy>                Switch the chunking strategy used for future indexing
    status                          Show the current strategy and configuration
    collections                     List all collections in the vector store
    eval                            Run the evaluation set (not yet implemented)
    compare                         Compare all strategies (not yet implemented)
    help                            Show this help text
    quit / exit                     Exit the program
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from ragforge.chunker.chunker_factory import create_chunker
from ragforge.cli_stage2 import build_advanced_strategy, build_generator
from ragforge.cli_stage3 import build_adaptive_strategy, build_crag_strategy, build_self_rag_strategy
from ragforge.config import RagForgeConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.generator.generator import Generator
from ragforge.generator.simple_generator import SimpleGenerator
from ragforge.llm.llm_service import LlmService
from ragforge.parser.parser_router import ParserRouter
from ragforge.pipeline.rag_pipeline import RAGPipeline
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.dense_retriever import DenseRetriever
from ragforge.retriever.hybrid_retriever import HybridRetriever
from ragforge.retriever.retriever import Retriever
from ragforge.store.milvus_vector_store import MilvusVectorStore
from ragforge.store.vector_store import VectorStore
from ragforge.strategy.agentic_rag_strategy import AgenticRAGStrategy
from ragforge.strategy.rag_strategy import RAGStrategy

CONFIG_PATH = "config/stage4.yml"

BANNER = """
======================================
        RagForge - Stage 4
======================================
"""

HELP_TEXT = """
Available commands:
  index <path>                Index a document into the vector store
  query [strategy] <text>      Ask a question (default: current strategy)
  strategy [name]               Show available strategies, or switch the default
  chunk <strategy>             Switch chunking strategy for future indexing
  status                        Show current strategy and configuration
  collections                   List all collections
  eval                           Run the evaluation set (not yet implemented)
  compare                        Compare all strategies (not yet implemented)
  help                           Show this help text
  quit / exit                    Exit the program

Strategy names for "query [strategy] <text>": naive, advanced, self_rag, crag, adaptive, agentic

Note: there is no "use <collection>" command in Stage 4. Every query
automatically picks whichever collection looks most relevant to that
question (see "status" for details), and Agentic RAG additionally
searches across every collection on its own during its retrieval rounds.
"""


def build_agentic_strategy(
    config: RagForgeConfig,
    dense_retriever: Retriever,
    hybrid_retriever: Retriever,
    generator: Generator,
    llm_service: LlmService,
    query_engine: QueryEngine | None,
    vector_store: VectorStore | None,
) -> AgenticRAGStrategy:
    """Assemble the Agentic RAG strategy from already-built collaborators.

    vector_store and collection_prefix are what power
    AgenticRAGStrategy's own internal cross-collection retrieval (see
    its module docstring) -- no other Stage 3 strategy needs either.
    """
    return AgenticRAGStrategy(
        dense_retriever,
        hybrid_retriever,
        generator,
        llm_service,
        query_engine,
        vector_store,
        config.vector_store.collection_prefix,
        config.strategy.agentic.max_iterations,
    )


def _select_best_collection(
    query_text: str,
    dense_retriever: Retriever,
    vector_store: VectorStore,
    collection_prefix: str,
) -> str | None:
    """Pick the collection that looks most relevant to query_text.

    Runs one quick dense retrieval per known collection and compares
    the average result score, returning the short name (prefix
    stripped) of whichever collection scored highest. This is what lets
    Stage 4's CLI skip a manual "use <collection>" step -- each query
    picks its own best-matching collection automatically. Returns None
    if no collection exists yet, or if listing collections fails.
    """
    try:
        collections = vector_store.list_collections()
    except Exception:
        return None
    if not collections:
        return None

    prefix = collection_prefix
    best_collection: str | None = None
    best_score = -1.0
    for collection in collections:
        short_name = collection[len(prefix):] if prefix and collection.startswith(prefix) else collection
        try:
            quick_result = dense_retriever.retrieve(query_text, short_name)
            scores = [r.score for r in quick_result.results] if quick_result.results else []
            avg_score = sum(scores) / len(scores) if scores else 0.0
        except Exception:
            print(f"  (quick scan of collection '{collection}' failed, skipping)")
            continue
        # Surfaced as a plain print (not logger.debug) so it's visible by
        # default -- picking the wrong collection silently was hard to
        # diagnose during testing, since the only visible symptom was a
        # wrong or "not enough information" answer several steps later.
        print(f"  scan: {short_name} -> avg_score={avg_score:.3f} ({len(scores)} result(s))")
        if avg_score > best_score:
            best_score = avg_score
            best_collection = short_name

    return best_collection


def main() -> None:
    # Configure logging before anything else gets constructed, so that
    # the logger.info(...) diagnostics scattered across the strategy
    # classes are actually visible on the console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load ANTHROPIC_API_KEY (and any other secrets) from .env before
    # anything that needs it gets constructed.
    load_dotenv()

    print(BANNER)

    print(f"Loading config: {CONFIG_PATH}")
    config = RagForgeConfig.load(CONFIG_PATH)
    print("Config loaded.")

    # Shared components -- indexing and every strategy below all read
    # from the same embedding service, vector store, and chunker, so a
    # document indexed once is queryable by any of them.
    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)
    parser_router = ParserRouter()
    chunker = create_chunker(config.chunker, embedding_service)
    llm_service = LlmService(config.llm)

    # Stage 1 baseline: kept as a special case outside the strategy
    # registry below, the same way earlier CLIs treat it.
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

    # Every strategy shares one citation-capable generator.
    shared_generator = build_generator(config, llm_service)

    advanced_strategy = build_advanced_strategy(config, hybrid_retriever, shared_generator, llm_service)
    self_rag_strategy = build_self_rag_strategy(config, hybrid_retriever, shared_generator, llm_service, query_engine)
    crag_strategy = build_crag_strategy(config, dense_retriever, hybrid_retriever, shared_generator, query_engine)
    adaptive_strategy = build_adaptive_strategy(
        config, llm_service, dense_retriever, hybrid_retriever, shared_generator, query_engine, self_rag_strategy
    )
    agentic_strategy = build_agentic_strategy(
        config, dense_retriever, hybrid_retriever, shared_generator, llm_service, query_engine, vector_store
    )

    strategies: dict[str, RAGStrategy] = {
        "advanced": advanced_strategy,
        "self_rag": self_rag_strategy,
        "crag": crag_strategy,
        "adaptive": adaptive_strategy,
        "agentic": agentic_strategy,
    }

    print("Pipelines ready.\n")

    # Informational only: just lets the user see what's already indexed
    # at startup. Nothing here is "selected" -- every query below picks
    # its own best-matching collection.
    try:
        existing_collections = vector_store.list_collections()
        if existing_collections:
            print(f"Existing collections: {', '.join(existing_collections)}")
    except Exception:
        pass

    last_indexed_collection = ""
    # config.strategy.name picks the initial default; fall back to
    # "agentic" if it names a strategy that isn't in the registry
    # (e.g. a typo in stage4.yml).
    current_strategy_name = config.strategy.name if config.strategy.name in strategies else "agentic"

    print(f"System ready (strategy: {current_strategy_name}). Type 'help' to see available commands.\n")

    while True:
        try:
            raw_input_line = input("ragforge-s4> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw_input_line:
            continue

        parts = raw_input_line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        try:
            if command == "index":
                if not argument:
                    print("Usage: index <file path>")
                    continue
                if not Path(argument).exists():
                    print(f"File not found: {argument}")
                    continue
                last_indexed_collection = naive_pipeline.index_document(argument)
                print(f"Indexed. Collection: {last_indexed_collection}")

            elif command == "query":
                if not argument:
                    print(
                        "Usage: query [strategy] <question>  "
                        "(strategy: naive, advanced, self_rag, crag, adaptive, agentic)"
                    )
                    continue

                # An optional leading strategy name picks a strategy just
                # for this one query; otherwise fall back to whichever
                # strategy is currently the default.
                sub_parts = argument.split(maxsplit=1)
                if len(sub_parts) > 1 and (sub_parts[0] == "naive" or sub_parts[0] in strategies):
                    strategy_name, question = sub_parts[0], sub_parts[1]
                else:
                    strategy_name, question = current_strategy_name, argument

                best_collection = _select_best_collection(
                    question, dense_retriever, vector_store, config.vector_store.collection_prefix
                )
                if not best_collection:
                    print("No collection selected yet. Index a document first.")
                    continue
                print(f"(auto-selected collection: {best_collection})")

                if strategy_name == "naive":
                    result = naive_pipeline.query(best_collection, question)
                else:
                    result = strategies[strategy_name].execute(question, best_collection)

                print(f"\n--- Answer (strategy: {strategy_name}) ---")
                print(result.answer)
                if result.cited_chunks:
                    print(f"\nCited chunks: {', '.join(result.cited_chunks)}")
                if result.hallucination_score >= 0:
                    print(f"Hallucination rate: {result.hallucination_score:.2f}")
                print(f"\n({len(result.retrieved_chunks)} chunk(s) retrieved)")

            elif command == "strategy":
                if not argument:
                    print(f"Available strategies: naive, {', '.join(strategies.keys())}")
                    print(f"Current strategy: {current_strategy_name}")
                    continue
                if argument == "naive" or argument in strategies:
                    current_strategy_name = argument
                    print(f"Default strategy switched to: {current_strategy_name}")
                else:
                    print(f"Unknown strategy: {argument}. Available: naive, {', '.join(strategies.keys())}")

            elif command == "chunk":
                if not argument:
                    print("Usage: chunk <strategy>  (fixed_size | recursive | semantic | parent_child | structure_aware)")
                    continue
                config.chunker.strategy = argument
                chunker = create_chunker(config.chunker, embedding_service)
                naive_pipeline.chunker = chunker
                print(f"Chunking strategy switched to: {argument}")

            elif command == "status":
                print(f"Last indexed collection: {last_indexed_collection or '(none)'}")
                print("Collection selection: automatic per query (no manual 'use')")
                print(f"Current strategy: {current_strategy_name}")
                print(f"Available strategies: naive, {', '.join(strategies.keys())}")
                print(f"Chunker strategy: {config.chunker.strategy}")
                print(f"Retriever strategy: {config.retriever.strategy}")
                print(f"Query engine: {'enabled' if config.query_engine.enabled else 'disabled'}")
                print(
                    f"Self-RAG max retries: {config.strategy.self_rag.max_retries}\n"
                    f"CRAG thresholds: high={config.strategy.crag.high_threshold}, "
                    f"low={config.strategy.crag.low_threshold}\n"
                    f"Adaptive: parallel_sub_queries={config.strategy.adaptive.parallel_sub_queries}, "
                    f"max_sub_queries={config.strategy.adaptive.max_sub_queries}\n"
                    f"Agentic: max_iterations={config.strategy.agentic.max_iterations}"
                )

            elif command == "collections":
                names = vector_store.list_collections()
                print(f"Collections ({len(names)}):")
                for name in names:
                    print(f"  - {name}")

            elif command in ("eval", "compare"):
                # Evaluator.py (Stage 2 step 5) hasn't been implemented yet.
                print(f"'{command}' isn't wired up yet -- Evaluator.py is still empty.")

            elif command == "help":
                print(HELP_TEXT)

            elif command in ("quit", "exit"):
                print("Goodbye!")
                break

            else:
                print(f"Unknown command: {command}. Type 'help' for a list of commands.")

        except Exception as e:
            # Keep the REPL alive even if a single command fails.
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
