"""Parent-child chunking strategy.

Splits text in two passes: a first pass cuts large "parent" chunks (used to
give the LLM full context), then each parent is split again into smaller
"child" chunks (used for precise retrieval matching). Both parent and child
chunks are returned in one flat list; a child chunk's metadata carries
`parentId` so its parent's full text can be looked up after retrieval.
"""

from __future__ import annotations

import logging

from .chunk import Chunk
from .chunker import Chunker
from ..parser.parse_result import ParseResult

logger = logging.getLogger(__name__)


class ParentChildChunker(Chunker):
    """Two-level chunker: large parent chunks for full context, small child
    chunks (linked back to their parent via `parentId`) for precise search.
    """

    def __init__(
        self,
        max_size: int,
        overlap: int,
        child_size: int,
        child_overlap: int,
    ) -> None:
        self.max_size = max_size
        self.overlap = overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Two-pass chunking: split the document into parent pieces, then
        split each parent's text into child pieces. Parent and child chunks
        are returned together in one flat list -- a child chunk's metadata
        carries `parentId` so its parent's full text can be looked up once
        the child is retrieved.
        """
        text = parse_result.content
        source = parse_result.metadata.source
        base_metadata = Chunk.metadata_from(parse_result.metadata)

        logger.info("Parent-child chunking started, text length: %d", len(text))

        # Step 1: split the whole document into parent-sized pieces.
        parent_pieces = self._split_parent(text)
        logger.info("Split into %d parent pieces", len(parent_pieces))

        chunks: list[Chunk] = []

        for i, parent_text in enumerate(parent_pieces):
            # Step 2: wrap the parent piece into its own Chunk. Each chunk
            # gets its own metadata copy -- base_metadata must never be
            # shared by reference here, since parent and child chunks each
            # add different extra keys on top of it.
            parent_id = self._generate_id(source, i, "p")
            parent_metadata = {**base_metadata, "isParent": True}
            chunks.append(Chunk(parent_id, parent_text, parent_metadata))

            # Step 3: split this parent's text into child-sized pieces, and
            # wrap each into its own Chunk, linked back to the parent via
            # `parentId`. The child id's hash input includes `parent_id`
            # (instead of a combined index like `i * 1000 + j`) so ids stay
            # unique no matter how many children a parent ends up with.
            child_pieces = self._split_child(parent_text)
            for j, child_text in enumerate(child_pieces):
                child_id = self._generate_id(source, j, f"c_{parent_id}")
                child_metadata = {
                    **base_metadata,
                    "isParent": False,
                    "parentId": parent_id,
                }
                chunks.append(Chunk(child_id, child_text, child_metadata))

        logger.info(
            "Parent-child chunking complete: %d parent pieces, %d chunks total",
            len(parent_pieces),
            len(chunks),
        )
        return chunks

    def _split_parent(self, text: str) -> list[str]:
        """Split `text` into parent-sized pieces using a sliding window:
        each piece is at most `max_size` characters, cuts prefer a sentence
        boundary (period/newline) over an arbitrary offset, and consecutive
        pieces share `overlap` characters of context.
        """
        pieces: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.max_size, len(text))

            # Prefer to cut at a sentence boundary: look for the last
            # period or newline at or before `end` so chunks don't split a
            # sentence in half.
            if end < len(text):
                last_period_cn = text.rfind("。", 0, end + 1)
                last_period_en = text.rfind(".", 0, end + 1)
                last_newline = text.rfind("\n", 0, end + 1)
                split_point = max(last_period_cn, last_period_en, last_newline)
                if split_point > start:
                    end = split_point + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                pieces.append(chunk_text)

            # Three-step advance guard: makes sure `start` always moves
            # forward -- even when `overlap` is large relative to
            # `max_size` -- so the loop can't spin forever.
            next_start = end - self.overlap
            min_advance = max(self.max_size - self.overlap, 1)
            if next_start < start + min_advance:
                next_start = start + min_advance
            if next_start <= start:
                next_start = end
            start = next_start

        return pieces

    def _split_child(self, parent_text: str) -> list[str]:
        """Split a single parent chunk's text into child-sized pieces.

        Same sliding-window-with-overlap idea as `_split_parent`, but
        without sentence-boundary snapping -- children are meant to be
        small, precisely-searched fragments, so a plain fixed-size cut on
        `child_size`/`child_overlap` is enough.
        """
        pieces: list[str] = []
        start = 0

        while start < len(parent_text):
            end = min(start + self.child_size, len(parent_text))

            chunk_text = parent_text[start:end].strip()
            if chunk_text:
                pieces.append(chunk_text)

            # Three-step advance guard: makes sure `start` always moves
            # forward -- even when `child_overlap` is large relative to
            # `child_size` -- so the loop can't spin forever.
            next_start = end - self.child_overlap
            min_advance = max(self.child_size - self.child_overlap, 1)
            if next_start < start + min_advance:
                next_start = start + min_advance
            if next_start <= start:
                next_start = end
            start = next_start

        return pieces
