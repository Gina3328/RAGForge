"""Data class for the result of a retrieval operation."""

from __future__ import annotations

from dataclasses import dataclass

from ragforge.store.search_result import SearchResult


@dataclass
class RetrievalResult:
    """The outcome of retrieving chunks relevant to a query.

    Attributes:
        query: The original query text.
        results: The matched chunks, most similar first.
        total_retrieved: Number of results returned.
    """

    query: str
    results: list[SearchResult]
    total_retrieved: int
