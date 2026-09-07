"""Base interface for retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .retrieval_result import RetrievalResult


class Retriever(ABC):
    """Finds chunks relevant to a query text."""

    @abstractmethod
    def retrieve(self, query_text: str, collection: str) -> RetrievalResult:
        """Retrieve chunks relevant to the query text from a collection."""
        ...
