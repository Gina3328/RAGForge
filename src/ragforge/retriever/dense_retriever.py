"""Dense (embedding-based) retrieval strategy."""

from __future__ import annotations

from ragforge.config import RetrieverConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.store.vector_store import VectorStore

from .parent_expansion import expand_parent_child
from .retrieval_result import RetrievalResult
from .retriever import Retriever


class DenseRetriever(Retriever):
    """Retrieves chunks by embedding the query and searching Milvus."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        config: RetrieverConfig,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = config.top_k
        self.score_threshold = config.score_threshold

    def retrieve(self, query_text: str, collection: str) -> RetrievalResult:
        """Embed the query, search for similar chunks, and package the result."""
        # Turn the query text into a vector.
        query_vector = self.embedding_service.embed(query_text)

        # Search for the most similar chunks in the given collection.
        results = self.vector_store.search_dense(
            collection,
            query_vector,
            self.top_k,
            self.score_threshold,
        )

        # If any hit is a "parent_child" child chunk, swap in its parent's
        # full text -- a no-op for every other chunking strategy. See
        # ParentExpansion's module docstring for why this has to happen
        # here, after the similarity search has already picked its winners.
        results = expand_parent_child(self.vector_store, collection, results)

        # Package the query and its results together.
        return RetrievalResult(
            query=query_text,
            results=results,
            total_retrieved=len(results),
        )
