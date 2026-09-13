"""CRAG strategy (Stage 3 -- confidence-based routing).

Corrective RAG's core idea: score how confident we are in what got
retrieved, then route to a different handling path based on that score:
  - High confidence (>= high_threshold): the retrieval is reliable
    enough to generate from directly.
  - Medium confidence: trigger a supplemental retrieval (rewrite the
    query, retrieve again), merge both result sets, then generate.
  - Low confidence (< low_threshold): decline to answer, optionally
    surfacing a couple of the closest (but unreliable) chunks as a
    partial reference.

Confidence is deliberately evaluated on a *dense* retrieval rather than
the hybrid one used for generation. The hybrid retriever fuses its
dense and keyword-match rankings via Reciprocal Rank Fusion (RRF),
which scores a candidate by its RANK within each ranking, not by how
similar it actually is -- so a perfect, rank-1-in-both-rankings match
and a mediocre one can end up with nearly the same RRF score, and the
score's absolute magnitude is capped by the RRF formula itself
(roughly 1 / rrf_k) regardless of match quality. That makes RRF scores
meaningless when compared against an absolute threshold like
high_threshold/low_threshold. Raw dense (cosine-similarity) scores
don't have that problem: a genuinely close match and a loose one
produce meaningfully different numbers, which is exactly what a
threshold-based confidence check needs.

Paper reference: Corrective Retrieval Augmented Generation.
"""

from __future__ import annotations

import logging

from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.retriever.retriever import Retriever
from ragforge.store.search_result import SearchResult

from .rag_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class CRAGStrategy(RAGStrategy):
    """Routes between direct generation, supplemental retrieval, and
    declining to answer, based on a confidence score computed from a
    dense retrieval's results.
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        hybrid_retriever: Retriever,
        generator: Generator,
        query_engine: QueryEngine | None,
        high_threshold: float,
        low_threshold: float,
    ) -> None:
        # dense_retriever: used only to score confidence (raw cosine
        # similarity has a meaningful absolute magnitude).
        # hybrid_retriever: used for the actual retrieval that gets
        # generated from (keyword-aware fusion gives it better recall).
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.query_engine = query_engine
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def get_name(self) -> str:
        return "CRAG (Stage 3)"

    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the CRAG flow: retrieve, score confidence, then route to
        direct generation / supplemental retrieval / refusal.
        """
        logger.info("CRAG strategy executing")

        # 1. Retrieve. hybrid_retriever's result is what actually gets
        # generated from; dense_retriever's result is retrieved
        # separately and used only to score confidence (see the module
        # docstring for why the two need different score semantics).
        retrieval_result = self.hybrid_retriever.retrieve(query, collection)
        dense_result = self.dense_retriever.retrieve(query, collection)

        # 2. Score confidence (a weighted combination of three signals).
        confidence = self._evaluate_confidence(dense_result)
        logger.info("Confidence score: %.2f", confidence)

        # 3. Route based on confidence.
        if confidence >= self.high_threshold:
            logger.info("Confidence high (>= %.2f), generating directly", self.high_threshold)
            return self.generator.generate(query, retrieval_result)

        if confidence >= self.low_threshold:
            logger.info("Confidence medium, triggering supplemental retrieval")
            supplemental = self._supplemental_retrieval(query, collection)

            merged = self._merge_results(retrieval_result.results, supplemental.results)
            merged_result = RetrievalResult(query=query, results=merged, total_retrieved=len(merged))
            return self.generator.generate(query, merged_result)

        logger.info("Confidence low, declining to answer")
        partial_chunks = [r.content for r in retrieval_result.results[:2]]
        partial_info = "\n".join(partial_chunks)

        answer = "Sorry, I couldn't find enough relevant information in the knowledge base to answer your question."
        if partial_info:
            answer += "\n\nHere is some possibly related information for reference:\n" + partial_info

        return GenerationResult(
            answer=answer,
            cited_chunks=[],
            hallucination_score=-1.0,
            retrieved_chunks=retrieval_result.results,
        )

    def _evaluate_confidence(self, result: RetrievalResult | None) -> float:
        """Score how confident we should be in `result`, as a weighted
        sum of three signals:
          - Top-1 score (weight 0.4): how relevant the single best
            match is.
          - Top-5 average score (weight 0.3): overall retrieval quality.
          - Effective-result ratio (weight 0.3): via score-gap
            detection, what fraction of the results are still relevant.

        `result` must come from a dense retrieval (raw cosine
        similarity, roughly 0..1 with meaningful absolute magnitude),
        not a hybrid/RRF-fused one -- see the module docstring for why.
        """
        if result is None or not result.results:
            return 0.0

        results = result.results

        # Signal 1: the single most relevant result's score.
        top1_score = results[0].score

        # Signal 2: average score of the top 5 results.
        top5_scores = [r.score for r in results[:5]]
        avg_score = sum(top5_scores) / len(top5_scores) if top5_scores else 0.0

        # Signal 3: score-gap detection -- once two adjacent results'
        # scores differ by more than 0.2, treat everything from that
        # point on as no longer relevant.
        effective_chunks = len(results)
        for i in range(1, len(results)):
            gap = results[i - 1].score - results[i].score
            if gap > 0.2:
                effective_chunks = i
                break

        effective_ratio = effective_chunks / len(results)
        confidence = 0.4 * top1_score + 0.3 * avg_score + 0.3 * effective_ratio
        logger.info(
            "Confidence breakdown: top1=%.3f, avg5=%.3f, effective_chunks=%d/%d (ratio=%.3f) -> %.3f",
            top1_score,
            avg_score,
            effective_chunks,
            len(results),
            effective_ratio,
            confidence,
        )
        return confidence

    def _supplemental_retrieval(self, query: str, collection: str) -> RetrievalResult:
        """Rewrite `query` via QueryEngine (if configured) and retrieve
        again, to surface additional candidates for a medium-confidence
        retrieval to merge with.
        """
        rewritten_query = query
        if self.query_engine is not None:
            try:
                processed = self.query_engine.process_query(query)
                if processed.rewritten:
                    rewritten_query = processed.rewritten
            except Exception:
                logger.warning("Query rewrite failed, retrying with the original query", exc_info=True)

        return self.hybrid_retriever.retrieve(rewritten_query, collection)

    @staticmethod
    def _merge_results(
        result_a: list[SearchResult],
        result_b: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge two result lists, deduplicating by chunk_id and keeping
        whichever copy has the higher score, then sort by score
        descending.
        """
        merged: dict[str, SearchResult] = {}
        for sr in result_a + result_b:
            existing = merged.get(sr.chunk_id)
            if existing is None or sr.score > existing.score:
                merged[sr.chunk_id] = sr

        return sorted(merged.values(), key=lambda r: r.score, reverse=True)
