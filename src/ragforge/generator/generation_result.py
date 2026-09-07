"""Data class for the result of an answer-generation step."""

from __future__ import annotations

from dataclasses import dataclass

from ragforge.store.search_result import SearchResult


@dataclass
class GenerationResult:
    """The outcome of generating an answer from retrieved context.

    Attributes:
        answer: The LLM-generated answer.
        cited_chunks: Chunks the answer explicitly cites (unused until
            Stage 3, always an empty list for now).
        hallucination_score: Hallucination-detection score (unused until
            Stage 3, always -1 for now).
        retrieved_chunks: The chunks that were retrieved and used as context.
    """

    answer: str
    cited_chunks: list[str]
    hallucination_score: float
    retrieved_chunks: list[SearchResult]
