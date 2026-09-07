"""Hybrid (Dense + Sparse) retrieval strategy with RRF fusion."""

from __future__ import annotations

import re

from ragforge.config import RetrieverConfig
from ragforge.embedding.embedding_service import EmbeddingService
from ragforge.store.search_result import SearchResult
from ragforge.store.vector_store import VectorStore

from .parent_expansion import expand_parent_child
from .retriever import Retriever
from .retrieval_result import RetrievalResult


class HybridRetriever(Retriever):
    """Retrieves chunks by fusing a dense (embedding) search with a
    simplified keyword-match score, via RRF.

    Unlike a "true" hybrid retriever with an independent sparse index,
    the sparse signal here is computed over the same candidate set the
    dense search already returned (over-fetched beyond top_k), not from a
    separate full-corpus scan -- see _score_by_keyword_match for why.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        config: RetrieverConfig,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = config.top_k
        self.score_threshold = config.score_threshold
        self.dense_weight = config.dense_weight
        self.sparse_weight = config.sparse_weight
        self.dynamic_weight = config.dynamic_weight
        self.rrf_k = config.rrf_k

    def retrieve(self, query_text: str, collection: str) -> RetrievalResult:
        """Retrieve chunks by fusing a dense search with a keyword-match
        score, via RRF.

        Steps:
          1. Decide the dense/sparse fusion weights -- either the
             configured base weights, or (if dynamic_weight is enabled)
             weights scaled by query length via _compute_dynamic_weights.
          2. Embed the query text.
          3. Run a dense (vector similarity) search, over-fetching well
             beyond top_k so there's a real candidate pool for the sparse
             signal and RRF fusion to work with -- not just the final
             top_k dense hits.
          4. Re-score that SAME candidate pool by keyword-match ratio.
             This is the simplified "sparse" signal -- see
             _score_by_keyword_match's docstring for why it reuses the
             dense candidates instead of scanning the full corpus.
          5. Fuse the dense ranking and the sparse ranking into a single
             ranked list via Reciprocal Rank Fusion (RRF).
          6. Truncate to top_k.
          7. If any surviving hit is a "parent_child" child chunk, swap in
             its parent's full text (no-op for every other chunking
             strategy -- see ParentExpansion's module docstring), then
             package the result.
        """
        if self.dynamic_weight:
            dense_weight, sparse_weight = self._compute_dynamic_weights(query_text)
        else:
            dense_weight, sparse_weight = self.dense_weight, self.sparse_weight

        query_vector = self.embedding_service.embed(query_text)

        # Over-fetch beyond top_k: RRF and the keyword-match re-scoring
        # need a real pool of candidates to work with, not just the final
        # top_k dense hits.
        candidate_k = max(self.top_k * 3, 30)
        dense_results = self.vector_store.search_dense(
            collection,
            query_vector,
            candidate_k,
            self.score_threshold,
        )

        sparse_results = self._score_by_keyword_match(query_text, dense_results)

        fused_results = self._rrf_fuse(
            dense_results, sparse_results, dense_weight, sparse_weight
        )

        top_results = fused_results[: self.top_k]
        top_results = expand_parent_child(self.vector_store, collection, top_results)
        return RetrievalResult(
            query=query_text,
            results=top_results,
            total_retrieved=len(top_results),
        )

    def _compute_dynamic_weights(self, query_text: str) -> tuple[float, float]:
        """Scale the configured dense/sparse weights based on query length.

        Word count (splitting on whitespace), not character count, is the
        length signal here -- this project's queries are primarily
        English, and Java's original char-count thresholds (tuned for
        Chinese, where a single character already carries close to a full
        word's worth of meaning) don't translate well: an English
        sentence's character count is inflated by word length and spaces,
        so it doesn't track query complexity the same way word count does.

        Short queries (< 4 words, e.g. "What is RAG?") are usually a
        single term/concept lookup, where exact keyword matching tends to
        help more than semantic search -- so sparse gets scaled up and
        dense down. Long queries (> 15 words) are more likely a
        multi-clause, elaborate question where semantic understanding
        matters more -- so dense gets scaled up and sparse down. Anything
        in between keeps the configured base weights unscaled.

        The scaled weights are then normalized to sum to 1, so the result
        can be used directly as RRF fusion weights.
        """
        word_count = len(query_text.split())
        dense_weight = self.dense_weight
        sparse_weight = self.sparse_weight

        if word_count < 4:
            dense_weight = self.dense_weight * 0.6
            sparse_weight = self.sparse_weight * 1.4
        elif word_count > 15:
            dense_weight = self.dense_weight * 1.4
            sparse_weight = self.sparse_weight * 0.6
        # else: a normal-length query keeps the configured base weights.

        # Normalize so dense_weight + sparse_weight == 1.
        total = dense_weight + sparse_weight
        if total > 0:
            dense_weight /= total
            sparse_weight /= total

        return dense_weight, sparse_weight

    def _score_by_keyword_match(
        self, query_text: str, candidates: list[SearchResult]
    ) -> list[SearchResult]:
        """Simplified "sparse" score: what fraction of the query's
        keywords appear (as a substring) in each candidate's content.

        This is NOT full BM25 sparse retrieval -- deliberately simplified
        versus the book's original design, to avoid needing a real Milvus
        sparse vector field/index (see the book's "设计与实现的差异说明"
        callout in 7.4). Notably, it also does NOT scan the full corpus
        for keyword matches: `candidates` here is the SAME over-fetched
        result list the dense search already returned in retrieve(). So
        this re-scores/re-ranks that pool by a different signal, rather
        than independently sourcing its own candidate set.

        Returns a new list of SearchResult (same chunk_id/content/
        metadata, but .score replaced by the keyword-match ratio),
        sorted descending by that score.
        """
        # Lowercase and split on whitespace/punctuation (English + the
        # common Chinese punctuation marks, in case a query mixes both).
        words = re.split(r"[\s,.;!?，。；！？、]+", query_text.lower())
        # Keep only words of length >= 2 (drop single letters/noise), and
        # dedupe via the set -- order doesn't matter, since each keyword
        # is only ever checked for substring membership below.
        keywords = {w for w in words if len(w) >= 2}

        if not keywords:
            # No usable keywords to score by -- every candidate is
            # equally (un)informative here, so leave the pool as-is.
            return list(candidates)

        scored: list[SearchResult] = []
        for candidate in candidates:
            content_lower = candidate.content.lower()
            match_count = sum(1 for keyword in keywords if keyword in content_lower)
            score = match_count / len(keywords)
            scored.append(
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=score,
                    metadata=candidate.metadata,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    def _rrf_fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        dense_weight: float,
        sparse_weight: float,
    ) -> list[SearchResult]:
        """Fuse the dense ranking and the sparse (keyword-match) ranking
        into a single ranked list, via Reciprocal Rank Fusion (RRF).

        RRF scores a candidate by its RANK within each ranking, not its
        raw score -- this sidesteps the fact that dense similarity scores
        and keyword-match ratios live on completely different scales and
        aren't directly comparable:

            rrf_score = dense_weight  * 1/(rrf_k + dense_rank)   [if present]
                      + sparse_weight * 1/(rrf_k + sparse_rank)  [if present]

        where dense_rank / sparse_rank are 1-based positions within each
        list, and rrf_k (60, the default recommended by the original RRF
        paper) damps how much rank #1 vs. rank #2 differ.

        In this project, every candidate is guaranteed to appear in BOTH
        rankings, since sparse_results is just dense_results re-scored and
        re-sorted -- but the lookups below are still written defensively
        (checking for None) so this method doesn't secretly depend on
        that always being true.
        """
        dense_ranks = {
            result.chunk_id: rank for rank, result in enumerate(dense_results, start=1)
        }
        sparse_ranks = {
            result.chunk_id: rank for rank, result in enumerate(sparse_results, start=1)
        }

        # chunk_id -> original SearchResult, so the final output can carry
        # over content/metadata (rrf_score below replaces .score, but
        # content and metadata still need to come from somewhere).
        by_chunk_id: dict[str, SearchResult] = {}
        for result in dense_results:
            by_chunk_id[result.chunk_id] = result
        for result in sparse_results:
            by_chunk_id.setdefault(result.chunk_id, result)

        fused: list[SearchResult] = []
        for chunk_id, original in by_chunk_id.items():
            rrf_score = 0.0

            dense_rank = dense_ranks.get(chunk_id)
            if dense_rank is not None:
                rrf_score += dense_weight * (1 / (self.rrf_k + dense_rank))

            sparse_rank = sparse_ranks.get(chunk_id)
            if sparse_rank is not None:
                rrf_score += sparse_weight * (1 / (self.rrf_k + sparse_rank))

            fused.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    score=rrf_score,
                    metadata=original.metadata,
                )
            )

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused
