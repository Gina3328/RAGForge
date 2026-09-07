"""A reranker that does no reranking at all."""

from __future__ import annotations

from ragforge.store.search_result import SearchResult

from .reranker import Reranker


class NoOpReranker(Reranker):
    """Passes candidates through unchanged, only truncating to top_k.

    Used when reranking is disabled in config -- keeps the rest of the
    pipeline (which always calls self.reranker.rerank(...)) working
    without needing an if/else around whether a real reranker is present.
    """

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        return candidates[:top_k]
