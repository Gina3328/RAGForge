"""Recursive-separator chunking strategy.

Splits text by trying separators in priority order (paragraph > line >
sentence > word > hard cut), so cuts land on natural boundaries wherever
possible instead of an arbitrary character offset. Pieces that are still
too large after being cut on a separator are recursively split again with
the *next* separator, hence "recursive". Consecutive chunks share
`overlap` characters of context, same idea as FixedSizeChunker.
"""

from __future__ import annotations

import logging
import re

from ragforge.parser.parse_result import ParseResult

from .chunk import Chunk
from .chunker import Chunker

logger = logging.getLogger(__name__)


class RecursiveChunker(Chunker):
    """Recursive-separator chunker with FixedSizeChunker-style overlap."""

    # Tried in priority order, highest first: paragraph break, line break,
    # English sentence end, Chinese sentence end (period/exclamation/
    # question), then whitespace, then "" (no separator left -- caller
    # falls back to a hard cut at max_size).
    SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "。", "！", "？", " ", "")

    def __init__(self, max_size: int, overlap: int) -> None:
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        text = parse_result.content
        base_metadata = Chunk.metadata_from(parse_result.metadata)

        logger.info("Recursive chunking started, text length: %d", len(text))

        # Step 1: recursively split into pieces, each <= max_size where possible.
        pieces = self._recursive_split(text, 0)
        # Drop any blank pieces produced by the split.
        pieces = [p.strip() for p in pieces if p.strip()]

        # Step 2: stitch `overlap` characters of context from the previous
        # piece onto the front of each following piece.
        pieces = self._apply_overlap(pieces)

        # Step 3: wrap into Chunk objects with generated ids.
        chunks: list[Chunk] = []
        for index, piece_text in enumerate(pieces):
            chunk_id = self._generate_id(parse_result.metadata.source, index)
            chunks.append(Chunk(chunk_id, piece_text, base_metadata))
        return chunks

    def _recursive_split(self, text: str, separator_level: int) -> list[str]:
        """Split `text` using SEPARATORS[separator_level]; any resulting
        piece still bigger than max_size recurses to the next separator
        level. Bottoms out at a hard character-count cut once separators
        are exhausted.
        """
        # Already small enough -- nothing to split.
        if len(text) <= self.max_size:
            return [text] if text.strip() else []

        # Ran out of separators (or hit the "" sentinel): hard cut.
        if separator_level >= len(self.SEPARATORS):
            return self._hard_split(text)

        sep = self.SEPARATORS[separator_level]
        if sep == "":
            return self._hard_split(text)

        # Paragraph breaks get a looser regex so a "blank" line that still
        # has stray whitespace on it (e.g. "\n  \n") still counts as one.
        if sep == "\n\n":
            parts = re.split(r"\n\s*\n", text)
        else:
            parts = text.split(sep)

        return self._merge_parts(parts, sep, separator_level)

    def _merge_parts(self, parts: list[str], sep: str, separator_level: int) -> list[str]:
        """Greedily merge consecutive `parts` back together as long as the
        combined text stays within max_size (re-inserting `sep` between
        them so the merged text reads naturally). A single part that's
        already too big on its own can't be merged with anything -- it
        gets recursively split at the next separator level instead.
        """
        merged: list[str] = []
        current = ""  # text accumulated for the chunk currently being built

        for part in parts:
            trimmed = part.strip()
            if not trimmed:
                continue

            if len(trimmed) > self.max_size:
                # Flush whatever's pending, then recurse on the oversized piece.
                if current:
                    merged.append(current)
                    current = ""
                merged.extend(self._recursive_split(trimmed, separator_level + 1))
                continue

            candidate = trimmed if not current else current + sep + trimmed
            if len(candidate) <= self.max_size:
                current = candidate
            else:
                # Adding `trimmed` would overflow -- close out the current
                # chunk and start a new one with `trimmed`.
                merged.append(current)
                current = trimmed

        if current:
            merged.append(current)
        return merged

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        """Prepend the trailing `overlap` characters of each piece onto the
        next one, so consecutive chunks share some context (same idea as
        FixedSizeChunker's overlap, applied after recursive splitting).
        """
        if self.overlap <= 0 or len(pieces) <= 1:
            return pieces

        result = [pieces[0]]
        for i in range(1, len(pieces)):
            prev_tail = pieces[i - 1][-self.overlap:]
            result.append(prev_tail + pieces[i])
        return result

    def _hard_split(self, text: str) -> list[str]:
        """Fallback: cut every max_size characters with no regard for word
        or sentence boundaries. Used once all separators are exhausted.
        """
        pieces: list[str] = []
        pos = 0
        while pos < len(text):
            end = min(pos + self.max_size, len(text))
            pieces.append(text[pos:end])
            pos = end  # NOT `pos += end` -- `end` is already an absolute
            # position, so `pos += end` would skip content and/or blow
            # past the string on the second iteration.
        return pieces


