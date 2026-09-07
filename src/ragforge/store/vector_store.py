"""Base interface for vector store implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chunk_embedding import ChunkEmbedding
from .search_result import SearchResult


class VectorStore(ABC):
    """Defines the basic operations a vector database backend must support."""

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a collection with the given vector dimension."""
        ...

    @abstractmethod
    def upsert(self, collection: str, data: list[ChunkEmbedding]) -> None:
        """Insert or update a batch of chunk embeddings."""
        ...

    @abstractmethod
    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        threshold: float,
    ) -> list[SearchResult]:
        """Find the top_k most similar chunks to the query vector."""
        ...

    @abstractmethod
    def get_by_ids(self, collection: str, chunk_ids: list[str]) -> list[SearchResult]:
        """Fetch chunks directly by chunk_id -- a primary-key lookup, not a
        similarity search. Used by parent-child expansion (see
        ragforge.retriever.ParentExpansion) to fetch a child hit's parent
        chunk by id once retrieval has already picked the child.

        Chunk ids that don't exist are silently skipped rather than raised
        as an error -- callers are expected to tolerate a partial result
        (fewer chunks back than ids requested). The returned SearchResults
        always carry score=0.0, since a direct lookup has no similarity to
        rank by; callers needing a meaningful score should use their
        original hit's score instead of this placeholder.
        """
        ...

    @abstractmethod
    def delete_by_source(self, collection: str, source: str) -> None:
        """Delete all chunks that came from the given source file."""
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete an entire collection."""
        ...

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List all collections managed by this store."""
        ...
