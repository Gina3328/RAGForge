"""Factory for building the Chunker selected by configuration.

`create_chunker` reads `config.strategy` and constructs the matching
Chunker implementation, pulling whatever fields that particular strategy
needs from the rest of `config`. Callers (the pipeline builder, evaluation
scripts) go through this instead of hardcoding a specific Chunker class, so
switching strategies is a one-line config change.
"""

from __future__ import annotations

from ragforge.config import ChunkerConfig
from ragforge.embedding.embedding_service import EmbeddingService

from .chunker import Chunker
from .fixed_size_chunker import FixedSizeChunker
from .parent_child_chunker import ParentChildChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker
from .structure_aware_chunker import StructureAwareChunker


def create_chunker(config: ChunkerConfig, embedding_service: EmbeddingService) -> Chunker:
    """Build the Chunker selected by `config.strategy`.

    `embedding_service` is only used by the "semantic" strategy -- it's
    taken as a separate argument (rather than living on ChunkerConfig)
    because it's a shared service instance, not a plain config value.
    """
    match config.strategy:
        case "fixed_size":
            return FixedSizeChunker(config.max_size, config.overlap)
        case "recursive":
            return RecursiveChunker(config.max_size, config.overlap)
        case "semantic":
            return SemanticChunker(
                embedding_service,
                config.similarity_threshold,
                config.min_chunk_size,
                config.max_chunk_size,
            )
        case "parent_child":
            return ParentChildChunker(
                config.max_size,
                config.overlap,
                config.child_size,
                config.child_overlap,
            )
        case "structure_aware":
            # StructureAwareChunker takes only (max_size, overlap) -- it
            # has no size field of its own, it just reuses these same two.
            return StructureAwareChunker(config.max_size, config.overlap)
        case _:
            # Unlike the Java reference (which silently falls back to
            # FixedSizeChunker for an unrecognized strategy), we raise --
            # a typo'd strategy name in the config should be caught
            # immediately, not silently swapped for a different strategy.
            raise ValueError(f"Unknown chunker strategy: {config.strategy!r}")
