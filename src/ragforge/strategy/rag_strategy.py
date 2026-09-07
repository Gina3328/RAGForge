"""Base interface for RAG strategies (the strategy pattern).

Every stage's RAG implementation (Naive / Advanced / and later Self-RAG /
CRAG / Adaptive) implements this same interface, so a caller -- an
Evaluator, an API handler, a CLI dispatcher -- can run any of them
interchangeably through `execute()` without needing to know which
concrete strategy it's actually holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragforge.generator.generation_result import GenerationResult


class RAGStrategy(ABC):
    """Common contract every RAG strategy implements."""

    @abstractmethod
    def get_name(self) -> str:
        """Strategy name, used for logging and evaluation report labels."""
        ...

    @abstractmethod
    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the full RAG flow for `query` against `collection` and
        return the generated answer.
        """
        ...
