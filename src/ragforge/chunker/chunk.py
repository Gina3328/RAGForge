"""Text chunk produced by a Chunker."""

from __future__ import annotations

from dataclasses import dataclass

from ragforge.parser.parse_result import Metadata


@dataclass
class Chunk:
    """A chunk of text produced by splitting a parsed document.

    Attributes:
        id: Unique identifier for the chunk.
        content: The chunk's text content.
        metadata: Metadata inherited from the source document.
    """

    id: str
    content: str
    metadata: dict

    @staticmethod
    def metadata_from(meta: Metadata) -> dict:
        """Build chunk metadata from a document's parsed metadata."""
        return {
            "title": meta.title,
            "source": meta.source,
            "sections": "|".join(meta.sections),
        }
