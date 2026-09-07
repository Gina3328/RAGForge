"""Base interface for answer-generation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragforge.retriever.retrieval_result import RetrievalResult

from .generation_result import GenerationResult


class Generator(ABC):
    """Generates an answer from a query and its retrieved context."""

    @abstractmethod
    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """Generate an answer using the query and retrieved chunks as context."""
        ...
