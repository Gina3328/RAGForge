"""Data class for a chunk paired with its embedding vector, ready to be
written into the vector store.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkEmbedding:
    """A single row to upsert into the vector store.

    Attributes:
        chunk_id: Unique identifier of the source chunk.
        content: The chunk's text content.
        embedding: The chunk's embedding vector.
        metadata: Metadata carried over from the chunk (title, source, etc.).
    """

    chunk_id: str
    content: str
    embedding: list[float]
    metadata: dict
