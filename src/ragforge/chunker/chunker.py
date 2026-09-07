"""Base interface for chunking strategies."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from ragforge.parser.parse_result import ParseResult

from .chunk import Chunk


class Chunker(ABC):
    """Splits parsed document content into a list of Chunks."""

    @abstractmethod
    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Split a parsed document into chunks."""
        ...

    @staticmethod
    def _generate_id(source: str, index: int, kind: str = "") -> str:
        """md5(filename[_kind]_index)[:12]. `kind` disambiguates id sequences
        that share the same source file and index space (e.g. parent vs.
        child chunks in ParentChildChunker)."""
        stem = Path(source).name
        raw = f"{stem}_{kind}_{index}" if kind else f"{stem}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]