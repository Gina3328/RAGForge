"""Base interface for reranking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragforge.store.search_result import SearchResult


class Reranker(ABC):
    """Re-scores and re-orders a small set of retrieval candidates.

    Takes the (comparatively cheap, comparatively coarse) candidates a
    Retriever already narrowed the corpus down to, and re-ranks them with
    a more precise -- but more expensive -- relevance signal, so only the
    truly best few end up in the final answer's context.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-rank candidates by relevance to the query, return the top_k."""
        ...
