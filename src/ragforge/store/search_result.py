"""Data class for a single vector search hit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single result returned from a similarity search.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        content: The matched chunk's text content.
        score: Similarity score for this match.
        metadata: Metadata carried over from the chunk.
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict
