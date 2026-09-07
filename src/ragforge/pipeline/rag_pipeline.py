"""Orchestrates the RAG pipeline: indexing documents and answering queries."""

from __future__ import annotations

import re
from pathlib import Path

from ragforge.chunker.chunker import Chunker
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.parser.parser_router import ParserRouter
from ragforge.retriever.retriever import Retriever
from ragforge.store.chunk_embedding import ChunkEmbedding
from ragforge.store.vector_store import VectorStore


class RAGPipeline:
    """Wires the parser, chunker, embedding, store, retriever, and generator
    together into two operations: indexing a document and answering a query.
    """

    def __init__(
            self,
            parser_router: ParserRouter,
            chunker: Chunker,
            embedding_service: EmbeddingService,
            vector_store: VectorStore,
            retriever: Retriever,
            generator: Generator,
    ) -> None:
        self.parser_router = parser_router
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator

    def index_document(self, file_path: str) -> str:
        """Parse, chunk, embed, and store a document. Returns its collection name."""
        # 1. Parse the document.
        parse_result = self.parser_router.parse(file_path)

        # 2. Split it into chunks.
        chunks = self.chunker.chunk(parse_result)

        # 3. Batch-embed every chunk's text.
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        # 4. Pair each chunk with its embedding.
        chunk_embeddings = [
            ChunkEmbedding(
                chunk_id=chunk.id,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # 5. Create the collection and write the data.
        collection_name = self.to_collection_name(file_path)
        self.vector_store.create_collection(collection_name, len(embeddings[0]))
        self.vector_store.upsert(collection_name, chunk_embeddings)

        return collection_name

    def query(self, collection: str, query_text: str) -> GenerationResult:
        """Retrieve relevant chunks and generate an answer from them."""
        retrieval_result = self.retriever.retrieve(query_text, collection)
        return self.generator.generate(query_text, retrieval_result)

    @staticmethod
    def to_collection_name(source: str) -> str:
        """Turn a file path into a valid Milvus collection name."""
        file_name = Path(source).stem
        name = re.sub(r"[^a-zA-Z0-9]", "_", file_name)
        if name and name[0].isdigit():
            name = f"col_{name}"
        return name.lower()
