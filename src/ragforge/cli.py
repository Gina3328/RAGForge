"""Interactive command-line entry point for the RagForge Stage 1 pipeline.

Available commands:
    index <path>   Index a document into the vector store
    query <text>   Ask a question against the most recently indexed collection
    collections    List all collections currently in the vector store
    help           Show this help text
    quit / exit    Exit the program
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ragforge.chunker.chunker_factory import create_chunker
from ragforge.config import RagForgeConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.generator.simple_generator import SimpleGenerator
from ragforge.parser.parser_router import ParserRouter
from ragforge.pipeline.rag_pipeline import RAGPipeline
from ragforge.retriever.dense_retriever import DenseRetriever
from ragforge.store.milvus_vector_store import MilvusVectorStore

CONFIG_PATH = "config/stage2.yml"

BANNER = """
======================================
        RagForge - Stage 1
======================================
"""

HELP_TEXT = """
Available commands:
  index <path>   Index a document into the vector store
  query <text>   Ask a question against the current collection
  collections    List all collections
  help           Show this help text
  quit / exit    Exit the program
"""


def build_pipeline(config: RagForgeConfig) -> RAGPipeline:
    """Construct every component and wire them into a RAGPipeline."""
    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)
    parser_router = ParserRouter()
    chunker = create_chunker(config.chunker, embedding_service)
    retriever = DenseRetriever(embedding_service, vector_store, config.retriever)
    generator = SimpleGenerator(config.llm)

    return RAGPipeline(
        parser_router,
        chunker,
        embedding_service,
        vector_store,
        retriever,
        generator,
    )


def main() -> None:
    # Load ANTHROPIC_API_KEY (and any other secrets) from .env before
    # anything that needs it gets constructed.
    load_dotenv()

    print(BANNER)

    print(f"Loading config: {CONFIG_PATH}")
    config = RagForgeConfig.load(CONFIG_PATH)
    print("Config loaded.")

    pipeline = build_pipeline(config)
    print("Pipeline ready.\n")

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
                current_collection = pipeline.index_document(argument)
                print(f"Indexed. Current collection: {current_collection}")

            elif command == "query":
                if not argument:
                    print("Usage: query <question>")
                    continue
                if not current_collection:
                    print("No collection selected yet. Index a document first.")
                    continue
                result = pipeline.query(current_collection, argument)
                print("\n--- Answer ---")
                print(result.answer)
                print(f"\n({len(result.retrieved_chunks)} chunk(s) retrieved)")

            elif command == "collections":
                names = pipeline.vector_store.list_collections()
                print(f"Collections ({len(names)}):")
                for name in names:
                    print(f"  - {name}")

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
