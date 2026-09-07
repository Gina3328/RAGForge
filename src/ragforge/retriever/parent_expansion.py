"""Parent-chunk expansion for the "parent_child" chunking strategy.

ParentChildChunker indexes small "child" chunks for precise retrieval
matching (see its own docstring) alongside their larger "parent" chunks,
linking each child back to its parent via metadata["parentId"]. Retrieval
naturally tends to surface children over parents -- they're shorter and
more topically focused, so they often score higher in a similarity search
-- but a raw child fragment (as short as ChunkerConfig.child_size
characters) is too little context to generate a good answer from on its
own. This module's job is the step ParentChildChunker's docstring says has
to happen "after retrieval": look up each child hit's parent by id and
swap in the parent's full text, so the LLM always gets full-context
chunks, regardless of which granularity actually won the similarity
search.

Shared by DenseRetriever and HybridRetriever (called at the end of each
one's retrieve()) so both retrieval strategies get this for free, and so a
future retrieval strategy only needs to call it too rather than
reimplementing it.
"""

from __future__ import annotations

from ragforge.store.search_result import SearchResult
from ragforge.store.vector_store import VectorStore


def expand_parent_child(
    vector_store: VectorStore, collection: str, results: list[SearchResult]
) -> list[SearchResult]:
    """Expand any child-chunk hit in `results` to its parent's full content.

    Results that aren't child chunks (metadata has no "parentId" -- true
    for every chunking strategy other than "parent_child", and for parent
    chunks that won the search on their own) pass through unchanged. If a
    child's parent can't be found in the store for some reason, that hit
    also passes through unchanged rather than being dropped.

    Multiple child hits can share the same parent; after expansion those
    collapse into a single entry (keeping the highest score among them)
    instead of repeating the same parent content multiple times and
    wasting context budget on duplicates. The returned list is sorted by
    score, descending.
    """
    parent_ids = {r.metadata.get("parentId") for r in results if r.metadata.get("parentId")}
    if not parent_ids:
        # Nothing to expand: either this isn't parent_child chunking, or
        # every hit here was already a parent chunk.
        return results

    parents = vector_store.get_by_ids(collection, list(parent_ids))
    parent_content_by_id = {p.chunk_id: p.content for p in parents}

    expanded_by_key: dict[str, SearchResult] = {}
    for r in results:
        parent_id = r.metadata.get("parentId")
        if parent_id and parent_id in parent_content_by_id:
            key = parent_id
            chunk_id = parent_id
            content = parent_content_by_id[parent_id]
        else:
            # Not a child hit, or its parent is missing -- keep as-is.
            key = r.chunk_id
            chunk_id = r.chunk_id
            content = r.content

        existing = expanded_by_key.get(key)
        if existing is None or r.score > existing.score:
            expanded_by_key[key] = SearchResult(
                chunk_id=chunk_id, content=content, score=r.score, metadata=r.metadata
            )

    return sorted(expanded_by_key.values(), key=lambda r: r.score, reverse=True)
