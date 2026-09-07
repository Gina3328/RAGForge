"""Semantic-similarity chunking strategy.

Splits text where meaning shifts: adjacent sentences are embedded, and a
cut is made wherever the cosine similarity between two consecutive
sentences drops below `similarity_threshold` (below that, they're treated
as belonging to different topics).
"""

from __future__ import annotations

import logging

from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.parser.parse_result import ParseResult

from .chunk import Chunk
from .chunker import Chunker

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """Semantic-similarity chunker: splits text where sentence-to-sentence
    embedding similarity drops below a threshold.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> None:
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Split `parse_result.content` into sentences, embed them, group
        consecutive sentences whose similarity stays >= similarity_threshold,
        then hand the groups to `_merge_and_split` before wrapping the
        result into Chunk objects.
        """
        text = parse_result.content
        base_metadata = Chunk.metadata_from(parse_result.metadata)

        logger.info("Semantic chunking started, text length: %d", len(text))

        # Step 1: split into sentences.
        sentences = self._split_sentence(text)
        logger.info("Split into %d sentences", len(sentences))

        if not sentences:
            return []

        # Step 2: batch-embed every sentence in one call.
        logger.info("Fetching sentence embeddings...")
        embeddings = self.embedding_service.embed_batch(sentences)

        # Step 3: cosine similarity between each pair of adjacent sentences.
        # similarities[i] is the similarity between sentences[i] and
        # sentences[i + 1].
        similarities = []
        for i in range(len(embeddings) - 1):
            similarities.append(self._cosine_similarity(embeddings[i], embeddings[i + 1]))

        # Step 4: group sentences, cutting a new group wherever similarity
        # to the previous sentence drops below the threshold. The first
        # sentence has no "previous" sentence to compare against, so it
        # unconditionally starts the first group.
        groups: list[list[str]] = [[sentences[0]]]
        for i, similarity in enumerate(similarities):
            if similarity >= self.similarity_threshold:
                # Still on-topic -- keep it in the current group.
                groups[-1].append(sentences[i + 1])
            else:
                # Topic shifted -- start a new group.
                groups.append([sentences[i + 1]])

        # Step 5: merge groups that ended up too small, split ones too big.
        merged_chunks = self._merge_and_split(groups)

        # Step 6: assign ids and wrap into Chunk objects.
        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(merged_chunks):
            chunk_id = self._generate_id(parse_result.metadata.source, index)
            chunks.append(Chunk(chunk_id, chunk_text, base_metadata))
        return chunks

    def _split_sentence(self, text: str) -> list[str]:
        """Split text on sentence-ending punctuation (period, exclamation
        mark, question mark)."""
        sentences: list[str] = []
        pos = 0

        for i, char in enumerate(text):
            is_terminator = False

            if char in ("。", "！", "？", "!", "?"):
                # Chinese terminators (and English "!"/"?") end a sentence
                # unambiguously on their own -- no lookahead needed.
                is_terminator = True
            elif char == ".":
                # English "." is ambiguous (decimals like "3.14",
                # abbreviations, filenames like "test.py") -- only treat
                # it as a sentence end when followed by whitespace, or
                # when it's the last character in the text.
                next_char = text[i + 1] if i + 1 < len(text) else " "
                is_terminator = next_char.isspace()

            if is_terminator:
                sentence = text[pos : i + 1].strip()
                if sentence:
                    sentences.append(sentence)
                pos = i + 1

        # Flush whatever's left after the last terminator -- this also
        # covers text that has no terminating punctuation at all.
        tail = text[pos:].strip()
        if tail:
            sentences.append(tail)

        return sentences


    def _merge_and_split(self, groups: list[list[str]]) -> list[str]:
        """Merge groups smaller than `self.min_chunk_size` into a neighbor;
        split groups larger than `self.max_chunk_size` via `_hard_split`."""
        # Step 1: join each group's sentences into a single string.
        texts = ["".join(group) for group in groups]

        # Step 2: greedily accumulate consecutive texts *while still too
        # small* -- the opposite condition from RecursiveChunker's
        # _merge_parts, which accumulates *while it still fits under a
        # max*. Here we keep gluing until we've reached min_chunk_size,
        # then close out and start the next chunk from scratch.
        merged: list[str] = []
        current = ""
        for text in texts:
            current += text
            if len(current) >= self.min_chunk_size:
                merged.append(current)
                current = ""

        if current:
            # Leftover tail never reached min_chunk_size on its own --
            # fold it into the previous chunk instead of leaving it as an
            # undersized chunk by itself.
            if merged:
                merged[-1] += current
            else:
                merged.append(current)

        # Step 3: anything still too big -- either a single group was
        # already oversized, or merging in step 2 pushed a piece past
        # max_chunk_size -- gets hard-split. Build a fresh list rather
        # than mutating the one being iterated.
        result: list[str] = []
        for text in merged:
            if len(text) > self.max_chunk_size:
                result.extend(self._hard_split(text))
            else:
                result.append(text)

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Hard split: cut every `self.max_chunk_size` characters while
        still preferring to land on a sentence boundary."""
        result = []
        pos = 0
        while pos < len(text):
            end = min(pos + self.max_chunk_size, len(text))

            # Try to land the cut on a sentence boundary instead of an
            # arbitrary offset. Reset every iteration -- otherwise a stale
            # value from a previous loop pass could leak in, or this could
            # be undefined entirely on an iteration where `end` already
            # equals `len(text)`.
            last_end = -1
            if end < len(text):
                for i in range(end, pos, -1):
                    c = text[i - 1]
                    if c in ("。", "！", "？", "!", "?", "\n"):
                        last_end = i
                        break

            if last_end > pos:
                end = last_end

            result.append(text[pos:end])
            pos = end

        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors.

        cos(A, B) = (A . B) / (||A|| * ||B||)

        Result range is [-1, 1]; the closer to 1, the more similar the two
        vectors' directions (i.e. the more semantically similar). Used here
        to judge how coherent two consecutive sentences are -- below
        `similarity_threshold`, a cut is made between them.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        # A zero vector has no direction, so "similarity" is undefined --
        # treat it as "not similar" rather than raising a ZeroDivisionError.
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

