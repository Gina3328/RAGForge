"""Parsed document result: extracted content plus metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metadata:
    """Metadata describing a parsed document.

    Attributes:
        title: Document title (first top-level heading for Markdown,
            falls back to the file name for formats without headings,
            e.g. PDF).
        source: Path to the source file.
        sections: List of top-level heading titles. Only extractable
            from formats with heading structure (e.g. Markdown); left
            empty otherwise.
    """

    title: str
    source: str
    sections: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """Result of parsing a document.

    This is the data handed off from the parsing stage to the chunking
    stage of the RAG pipeline.

    Attributes:
        content: The full extracted text content.
        metadata: Metadata describing the document.
    """

    content: str
    metadata: Metadata
