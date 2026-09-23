"""Interactive command-line entry point for the RagForge Stage 2 pipeline.

This is the Stage 2 counterpart to cli.py (Stage 1): it reads
config/stage2.yml, decides which concrete implementation to build
for each optional component (query engine, reranker, hallucination
detector), and wires everything into an AdvancedRAGStrategy. A Stage 1
RAGPipeline is built alongside it (sharing the same embedding service,
vector store, and chunker) so "query naive" can still answer with the
plain baseline pipeline -- useful both as a sanity check and, later, as
the "naive" side of the Stage 5 compare command.

Available commands:
    index <path>          Index a document into the vector store
    query <text>           Ask a question using the Stage 2 advanced strategy (default)
    query naive <text>      Ask a question using the Stage 1 naive pipeline
    query adv <text>        Ask a question using the Stage 2 advanced strategy (explicit)
    chunk <strategy>        Switch the chunking strategy used for future indexing
    status                  Show the current collection and component configuration
    collections             List all collections in the vector store
    use <collection>        Switch the current collection
    eval                    Run the evaluation set (not yet implemented)
    compare                 Compare naive vs advanced results (not yet implemented)
    help                    Show this help text
    quit / exit             Exit the program
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ragforge.chunker.chunker_factory import create_chunker
from ragforge.config import RagForgeConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.generator.citation_generator import CitationGenerator
from ragforge.generator.context_compressor import ContextCompressor
from ragforge.generator.generator import Generator
from ragforge.generator.hallucination_detector import HallucinationDetector
from ragforge.generator.simple_generator import SimpleGenerator
from ragforge.llm.llm_service import LlmService
from ragforge.parser.parser_router import ParserRouter
from ragforge.pipeline.rag_pipeline import RAGPipeline
from ragforge.query.query_engine import QueryEngine
from ragforge.reranker.cross_encoder_reranker import CrossEncoderReranker
from ragforge.reranker.no_op_reranker import NoOpReranker
from ragforge.reranker.reranker import Reranker
from ragforge.retriever.dense_retriever import DenseRetriever
from ragforge.retriever.hybrid_retriever import HybridRetriever
from ragforge.retriever.retriever import Retriever
from ragforge.store.milvus_vector_store import MilvusVectorStore
from ragforge.strategy.advanced_rag_strategy import AdvancedRAGStrategy

CONFIG_PATH = "config/stage2.yml"

BANNER = """
======================================
        RagForge - Stage 2
======================================
"""

HELP_TEXT = """
Available commands:
  index <path>        Index a document into the vector store
  query <text>         Ask a question (Stage 2 advanced strategy, default)
  query naive <text>    Ask a question using the Stage 1 naive pipeline
  query adv <text>      Ask a question using the Stage 2 advanced strategy
  chunk <strategy>      Switch chunking strategy for future indexing
  status                Show current collection and component configuration
  collections           List all collections
  use <collection>      Switch the current collection
  eval                  Run the evaluation set (not yet implemented)
  compare               Compare naive vs advanced results (not yet implemented)
  help                  Show this help text
  quit / exit           Exit the program
"""


def build_retriever(
    config: RagForgeConfig,
    embedding_service: EmbeddingService,
    vector_store: MilvusVectorStore,
) -> Retriever:
    """Build the retriever selected by config.retriever.strategy.

    Both DenseRetriever and HybridRetriever take the exact same
    (embedding_service, vector_store, config) signature -- config.retriever
    is handed to whichever one gets built, and each pulls only the fields
    it actually needs out of it.
    """
    if config.retriever.strategy == "hybrid":
        return HybridRetriever(embedding_service, vector_store, config.retriever)
    return DenseRetriever(embedding_service, vector_store, config.retriever)


def build_reranker(config: RagForgeConfig) -> Reranker:
    """Build the reranker selected by config.reranker.

    Note: config.reranker.strategy currently offers "llm" as a value (see
    stage2.yml), but only CrossEncoderReranker (a local Cross-Encoder
    model) is implemented so far -- there's no LLM-prompted reranker
    variant yet. So for now,
    `enabled` is treated as the only real switch: True builds the one
    implementation that exists, False builds NoOpReranker (which exists
    specifically so callers never need an if/else around whether a real
    reranker is present -- see its own docstring).
    """
    if not config.reranker.enabled:
        return NoOpReranker()
    return CrossEncoderReranker()


def build_generator(config: RagForgeConfig, llm_service: LlmService) -> Generator:
    """Build the generator selected by config.generator.enable_citation."""
    if config.generator.enable_citation:
        return CitationGenerator(llm_service)
    return SimpleGenerator(config.llm)


def build_advanced_strategy(
    config: RagForgeConfig,
    retriever: Retriever,
    generator: Generator,
    llm_service: LlmService,
) -> AdvancedRAGStrategy:
    """Assemble the Stage 2 AdvancedRAGStrategy from already-built collaborators.

    query_engine and hallucination_detector are only built when their
    config flag is on -- AdvancedRAGStrategy treats both as optional and
    degrades gracefully when they're None. reranker and context_compressor
    are always built (NoOpReranker covers "reranking disabled" as a real
    object rather than None; context_compressor has no "disabled" switch
    of its own).
    """
    query_engine = QueryEngine.from_llm(llm_service) if config.query_engine.enabled else None
    reranker = build_reranker(config)
    context_compressor = ContextCompressor(
        config.generator.context_compression_threshold,
        config.generator.max_context_tokens,
    )
    hallucination_detector = (
        HallucinationDetector(llm_service) if config.generator.enable_hallucination_detection else None
    )

    return AdvancedRAGStrategy(
        retriever,
        generator,
        query_engine,
        reranker,
        context_compressor,
        hallucination_detector,
        config,
    )


def main() -> None:
    # Load ANTHROPIC_API_KEY (and any other secrets) from .env before
    # anything that needs it gets constructed.
    load_dotenv()

    print(BANNER)

    print(f"Loading config: {CONFIG_PATH}")
    config = RagForgeConfig.load(CONFIG_PATH)
    print("Config loaded.")

    # Shared components -- indexing (and both strategies below) all read
    # from the same embedding service, vector store, and chunker, so a
    # document indexed once is queryable by either strategy.
    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)
    parser_router = ParserRouter()
    chunker = create_chunker(config.chunker, embedding_service)
    llm_service = LlmService(config.llm)

    # Stage 1 baseline: always DenseRetriever + SimpleGenerator, regardless
    # of what config.retriever.strategy says -- this is meant to stay the
    # simplest possible pipeline, so it's a stable point of comparison for
    # the Stage 2 advanced strategy (and, later, for the "compare" command).
    naive_pipeline = RAGPipeline(
        parser_router,
        chunker,
        embedding_service,
        vector_store,
        DenseRetriever(embedding_service, vector_store, config.retriever),
        SimpleGenerator(config.llm),
    )

    # Stage 2 advanced strategy: components are selected/built based on
    # config.
    advanced_retriever = build_retriever(config, embedding_service, vector_store)
    advanced_generator = build_generator(config, llm_service)
    advanced_strategy = build_advanced_strategy(config, advanced_retriever, advanced_generator, llm_service)

    print("Pipelines ready.\n")

    current_collection = ""

    print("System ready. Type 'help' to see available commands.\n")

    while True:
        try:
            raw_input_line = input("ragforge> ").strip()
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
                    print("Usage: query <question> | query naive <question> | query adv <question>")
                    continue
                if not current_collection:
                    print("No collection selected yet. Index a document first.")
                    continue

                # "query naive ..." / "query adv ..." pick a strategy
                # explicitly; bare "query ..." defaults to advanced, since
                # that's the strategy this Stage 2 entry point exists for.
                sub_parts = argument.split(maxsplit=1)
                if sub_parts[0] == "naive" and len(sub_parts) > 1:
                    mode, question = "naive", sub_parts[1]
                elif sub_parts[0] == "adv" and len(sub_parts) > 1:
                    mode, question = "advanced", sub_parts[1]
                else:
                    mode, question = "advanced", argument

                if mode == "naive":
                    result = naive_pipeline.query(current_collection, question)
                else:
                    result = advanced_strategy.execute(question, current_collection)

                print("\n--- Answer ---")
                print(result.answer)
                if result.cited_chunks:
                    print(f"\nCited chunks: {', '.join(result.cited_chunks)}")
                if result.hallucination_score >= 0:
                    print(f"Hallucination rate: {result.hallucination_score:.2f}")
                print(f"\n({len(result.retrieved_chunks)} chunk(s) retrieved, mode={mode})")

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
                print(f"Chunker strategy: {config.chunker.strategy}")
                print(f"Retriever strategy: {config.retriever.strategy}")
                print(f"Query engine: {'enabled' if config.query_engine.enabled else 'disabled'}")
                print(f"Reranker: {'enabled' if config.reranker.enabled else 'disabled'}")
                print(
                    "Hallucination detection: "
                    f"{'enabled' if config.generator.enable_hallucination_detection else 'disabled'}"
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
