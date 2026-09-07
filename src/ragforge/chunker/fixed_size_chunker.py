"""Fixed-size chunking strategy."""

from __future__ import annotations

from ragforge.parser.parse_result import ParseResult

from .chunk import Chunk
from .chunker import Chunker


class FixedSizeChunker(Chunker):
    """Splits text into fixed-size chunks with overlap between them.

    The simplest chunking strategy: text is cut into pieces of at most
    `max_size` characters, with `overlap` characters shared between
    consecutive chunks so information near a boundary isn't lost.
    Where possible, cuts are aligned to a sentence boundary (a period
    or newline) instead of an arbitrary character.
    """

    def __init__(self, max_size: int, overlap: int) -> None:
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        text = parse_result.content
        base_metadata = Chunk.metadata_from(parse_result.metadata)
        chunks: list[Chunk] = []

        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.max_size, len(text))

            # Prefer to cut at a sentence boundary: look for the last
            # period or newline at or before `end` so chunks don't
            # split a sentence in half.
            if end < len(text):
                last_period_cn = text.rfind("。", 0, end + 1)
                last_period_en = text.rfind(".", 0, end + 1)
                last_newline = text.rfind("\n", 0, end + 1)
                split_point = max(last_period_cn, last_period_en, last_newline)
                if split_point > start:
                    end = split_point + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = self._generate_id(parse_result.metadata.source, index)
                chunks.append(Chunk(chunk_id, chunk_text, base_metadata))

            # Advance the next chunk's start by `overlap` characters so
            # consecutive chunks share some context.
            next_start = end - self.overlap
            # Always move forward by at least min_advance to avoid
            # looping forever when overlap is close to max_size.
            min_advance = max(self.max_size - self.overlap, 1)
            if next_start < start + min_advance:
                next_start = start + min_advance
            if next_start <= start:
                next_start = end
            start = next_start
            index += 1

        return chunks


