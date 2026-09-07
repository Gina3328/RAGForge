"""Structure-aware chunking strategy.

Uses the document's own heading structure (`metadata.sections`) to cut
text: each top-level section becomes one chunk, so a chunk's content maps
onto one complete semantic unit instead of an arbitrary slice. Works well
for formats with reliable heading extraction (Markdown); documents with no
usable heading structure (e.g. PDF, where `sections` is typically empty)
fall back to `RecursiveChunker` for the whole document.
"""

from __future__ import annotations

import logging

from .chunk import Chunk
from .chunker import Chunker
from .recursive_chunker import RecursiveChunker
from ..parser.parse_result import Metadata, ParseResult

logger = logging.getLogger(__name__)


class StructureAwareChunker(Chunker):
    """Splits text along its heading structure: one chunk per section,
    with sections too large for a single chunk delegated to
    `RecursiveChunker`. Falls back to `RecursiveChunker` entirely when no
    usable section structure is available.
    """

    def __init__(self, max_size: int, overlap: int) -> None:
        # `max_size` doubles as the per-section size threshold (a section
        # larger than this gets split further) and as the `max_size` used
        # by the `RecursiveChunker` this chunker falls back to.
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Split the document along its section headings.

        Each section becomes one chunk if it fits within `max_size`;
        oversized sections are handed to `RecursiveChunker` for a second
        pass. A document with no usable section structure -- no sections
        at all, or none of them locatable in the text -- falls back to
        `RecursiveChunker` for the whole document.
        """
        text = parse_result.content
        source = parse_result.metadata.source
        base_metadata = Chunk.metadata_from(parse_result.metadata)
        sections = parse_result.metadata.sections

        logger.info("Structure-aware chunking started, text length: %d", len(text))

        if not sections:
            # No section structure at all (e.g. PDF) -- degrade to
            # RecursiveChunker for the whole document.
            logger.info("No sections found, falling back to RecursiveChunker")
            return RecursiveChunker(self.max_size, self.overlap).chunk(parse_result)

        located_sections = self._extract_sections(text, sections)

        if not located_sections:
            # Sections were declared but none could actually be located in
            # the text -- same fallback as having no sections at all.
            logger.info(
                "None of the %d declared sections could be located, "
                "falling back to RecursiveChunker",
                len(sections),
            )
            return RecursiveChunker(self.max_size, self.overlap).chunk(parse_result)

        logger.info("Located %d of %d declared sections", len(located_sections), len(sections))

        chunks: list[Chunk] = []
        index = 0

        for title, content in located_sections:
            if not content:
                continue

            if len(content) <= self.max_size:
                # Section fits in one chunk as-is.
                chunk_id = self._generate_id(source, index)
                metadata = {**base_metadata, "sectionTitle": title}
                chunks.append(Chunk(chunk_id, content, metadata))
                index += 1
            else:
                # Section too large for one chunk -- split it with
                # RecursiveChunker, then re-wrap its pieces as our own
                # chunks (fresh ids, sectionTitle/subChunkIndex metadata).
                sub_parse_result = ParseResult(
                    content=content,
                    metadata=Metadata(title=title, source=source, sections=[]),
                )
                sub_chunks = RecursiveChunker(self.max_size, self.overlap).chunk(sub_parse_result)

                for sub_index, sub_chunk in enumerate(sub_chunks):
                    chunk_id = self._generate_id(source, index)
                    metadata = {
                        **base_metadata,
                        "sectionTitle": title,
                        "subChunkIndex": sub_index,
                    }
                    chunks.append(Chunk(chunk_id, sub_chunk.content, metadata))
                    index += 1

        logger.info("Structure-aware chunking complete: %d chunks total", len(chunks))
        return chunks

    def _extract_sections(self, text: str, sections: list[str]) -> list[tuple[str, str]]:
        """Locate each section title in `text`, in order, and slice out
        its content.

        Searches for each title starting from where the previous title's
        occurrence ended, so titles are always matched in document order
        -- this also avoids matching an earlier, unrelated occurrence of
        the same text (e.g. a title mentioned in passing inside an
        earlier section's body). A title that can't be found at all as a
        literal substring is skipped: it doesn't produce a chunk, but
        the rest of the document is still processed normally.

        Returns a list of (title, content) pairs, one per title that was
        actually located, in document order. Each piece's content runs
        from its own title's position up to the start of the next
        *located* title (or the end of the text, for the last one).
        """
        located: list[tuple[str, int]] = []  # (title, start_pos), in order
        search_from = 0

        for title in sections:
            pos = text.find(title, search_from)
            if pos == -1:
                # Not found as a literal substring -- skip it rather
                # than guessing at a position.
                continue
            located.append((title, pos))
            search_from = pos + len(title)

        result: list[tuple[str, str]] = []
        for i, (title, start) in enumerate(located):
            end = located[i + 1][1] if i + 1 < len(located) else len(text)
            content = text[start:end].strip()
            result.append((title, content))

        return result
