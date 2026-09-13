"""Interactive command-line entry point for the RagForge Stage 3 pipeline.

This is the Stage 3 equivalent of cli_stage2.py: it reads
config/stage3.yml and wires up a *registry* of RAG strategies --
advanced (Stage 2), self_rag, crag, and adaptive (Stage 3) -- keyed by
name, mirroring the Java reference's Stage3App (a
`dict[str, RAGStrategy]` registry plus CLI dispatch; there is no
separate orchestrator class). "naive" stays a special case handled
directly through the Stage 1 RAGPipeline, the same way cli_stage2.py
already treats it -- RAGPipeline doesn't implement the RAGStrategy
interface, so it can't live in the same registry as the others.

Component builders (build_generator, build_advanced_strategy) are
imported from cli_stage2 rather than duplicated, so both entry points
stay driven by the same config-based construction logic for the parts
they share.

Available commands:
    index <path>                    Index a document into the vector store
    query [strategy] <text>         Ask a question (default: current strategy)
    strategy [name]                 Show or switch the default strategy
    chunk <strategy>                Switch the chunking strategy used for future indexing
    status                          Show the current collection, strategy, and configuration
    collections                     List all collections in the vector store
    use <collection>                Switch the current collection
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
from ragforge.config import RagForgeConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.generator.generator import Generator
from ragforge.generator.simple_generator import SimpleGenerator
from ragforge.llm.llm_service import LlmService
from ragforge.parser.parser_router import ParserRouter
from ragforge.pipeline.rag_pipeline import RAGPipeline
from ragforge.query.hyde_generator import HyDEGenerator
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.dense_retriever import DenseRetriever
from ragforge.retriever.hybrid_retriever import HybridRetriever
from ragforge.retriever.retriever import Retriever
from ragforge.store.milvus_vector_store import MilvusVectorStore
from ragforge.strategy.adaptive_rag_strategy import AdaptiveRAGStrategy
from ragforge.strategy.crag_strategy import CRAGStrategy
from ragforge.strategy.rag_strategy import RAGStrategy
from ragforge.strategy.self_rag_strategy import SelfRAGStrategy

CONFIG_PATH = "config/stage3.yml"

BANNER = """
======================================
        RagForge - Stage 3
======================================
"""

HELP_TEXT = """
Available commands:
  index <path>                Index a document into the vector store
  query [strategy] <text>      Ask a question (default: current strategy)
  strategy [name]               Show available strategies, or switch the default
  chunk <strategy>             Switch chunking strategy for future indexing
  status                        Show current collection, strategy, and configuration
  collections                   List all collections
  use <collection>              Switch the current collection
  eval                           Run the evaluation set (not yet implemented)
  compare                        Compare all strategies (not yet implemented)
  help                           Show this help text
  quit / exit                    Exit the program

Strategy names for "query [strategy] <text>": naive, advanced, self_rag, crag, adaptive
"""


def build_self_rag_strategy(
    config: RagForgeConfig,
    hybrid_retriever: Retriever,
    generator: Generator,
    llm_service: LlmService,
    query_engine: QueryEngine | None,
) -> SelfRAGStrategy:
    """Assemble the Self-RAG strategy from already-built collaborators."""
    hyde_generator = HyDEGenerator(llm_service)
    return SelfRAGStrategy(
        hybrid_retriever,
        generator,
        llm_service,
        query_engine,
        hyde_generator,
        config.strategy.self_rag.max_retries,
    )


def build_crag_strategy(
    config: RagForgeConfig,
    dense_retriever: Retriever,
    hybrid_retriever: Retriever,
    generator: Generator,
    query_engine: QueryEngine | None,
) -> CRAGStrategy:
    """Assemble the CRAG strategy from already-built collaborators.

    Both retrievers are passed in: hybrid_retriever is what CRAG
    actually generates from, while dense_retriever is used only to
    score confidence -- see CRAGStrategy's module docstring for why a
    raw dense (cosine-similarity) score, not a hybrid/RRF-fused one, is
    what a threshold-based confidence check needs.
    """
    return CRAGStrategy(
        dense_retriever,
        hybrid_retriever,
        generator,
        query_engine,
        config.strategy.crag.high_threshold,
        config.strategy.crag.low_threshold,
    )


def build_adaptive_strategy(
    config: RagForgeConfig,
    llm_service: LlmService,
    dense_retriever: Retriever,
    hybrid_retriever: Retriever,
    generator: Generator,
    query_engine: QueryEngine | None,
    self_rag_strategy: SelfRAGStrategy,
) -> AdaptiveRAGStrategy:
    """Assemble the Adaptive RAG strategy from already-built collaborators.

    Composes the already-built self_rag_strategy (rather than building
    its own) so the COMPLEX path reuses the exact same Self-RAG
    behavior "query naive|self_rag <text>" would give you directly.
    """
    return AdaptiveRAGStrategy(
        llm_service,
        dense_retriever,
        hybrid_retriever,
        generator,
        query_engine,
        self_rag_strategy,
        config.strategy.adaptive.parallel_sub_queries,
        config.strategy.adaptive.max_sub_queries,
    )


def main() -> None:
    # Configure logging before anything else gets constructed, so that
    # the logger.info(...) diagnostics scattered across the strategy
    # classes (e.g. CRAG's computed confidence score) are actually
    # visible on the console instead of being silently dropped by
    # Python's default WARNING-level root logger.
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
    # registry below, the same way cli_stage2.py treats it -- RAGPipeline
    # predates the RAGStrategy interface and doesn't implement it.
    naive_pipeline = RAGPipeline(
        parser_router,
        chunker,
        embedding_service,
        vector_store,
        DenseRetriever(embedding_service, vector_store, config.retriever),
        SimpleGenerator(config.llm),
    )

    # Both retrievers are needed here (unlike cli_stage2.py, which only
    # ever builds the one config.retriever.strategy picks): Adaptive RAG
    # routes SIMPLE queries through dense retrieval and MEDIUM/COMPLEX
    # queries through hybrid retrieval, while Self-RAG and CRAG both use
    # hybrid retrieval throughout.
    dense_retriever = DenseRetriever(embedding_service, vector_store, config.retriever)
    hybrid_retriever = HybridRetriever(embedding_service, vector_store, config.retriever)

    query_engine = QueryEngine.from_llm(llm_service) if config.query_engine.enabled else None

    # Every Stage 3 strategy shares one citation-capable generator (the
    # same choice cli_stage2.py's build_generator makes for Advanced RAG).
    shared_generator = build_generator(config, llm_service)

    # Stage 2 baseline, included so it stays reachable via "query advanced
    # <text>" for comparison against the Stage 3 strategies.
    advanced_strategy = build_advanced_strategy(config, hybrid_retriever, shared_generator, llm_service)

    self_rag_strategy = build_self_rag_strategy(config, hybrid_retriever, shared_generator, llm_service, query_engine)
    crag_strategy = build_crag_strategy(config, dense_retriever, hybrid_retriever, shared_generator, query_engine)
    adaptive_strategy = build_adaptive_strategy(
        config, llm_service, dense_retriever, hybrid_retriever, shared_generator, query_engine, self_rag_strategy
    )

    strategies: dict[str, RAGStrategy] = {
        "advanced": advanced_strategy,
        "self_rag": self_rag_strategy,
        "crag": crag_strategy,
        "adaptive": adaptive_strategy,
    }

    print("Pipelines ready.\n")

    current_collection = ""
    # config.strategy.name picks the initial default; fall back to
    # "adaptive" if it names a strategy that isn't in the registry
    # (e.g. a typo in stage3.yml).
    current_strategy_name = config.strategy.name if config.strategy.name in strategies else "adaptive"

    print(f"System ready (strategy: {current_strategy_name}). Type 'help' to see available commands.\n")

    while True:
        try:
            raw_input_line = input("ragforge-s3> ").strip()
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
                current_collection = naive_pipeline.index_document(argument)
                print(f"Indexed. Current collection: {current_collection}")

            elif command == "query":
                if not argument:
                    print("Usage: query [strategy] <question>  (strategy: naive, advanced, self_rag, crag, adaptive)")
                    continue
                if not current_collection:
                    print("No collection selected yet. Index a document first.")
                    continue

                # An optional leading strategy name picks a strategy just
                # for this one query; otherwise fall back to whichever
                # strategy is currently the default.
                sub_parts = argument.split(maxsplit=1)
                if len(sub_parts) > 1 and (sub_parts[0] == "naive" or sub_parts[0] in strategies):
                    strategy_name, question = sub_parts[0], sub_parts[1]
                else:
                    strategy_name, question = current_strategy_name, argument

                if strategy_name == "naive":
                    result = naive_pipeline.query(current_collection, question)
                else:
                    result = strategies[strategy_name].execute(question, current_collection)

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
                print(f"Current collection: {current_collection or '(none)'}")
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
                    f"max_sub_queries={config.strategy.adaptive.max_sub_queries}"
                )

            elif command == "collections":
                names = vector_store.list_collections()
                print(f"Collections ({len(names)}):")
                for name in names:
                    print(f"  - {name}")

            elif command == "use":
                if not argument:
                    print("Usage: use <collection name>")
                    continue
                current_collection = argument
                print(f"Current collection: {current_collection}")

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
