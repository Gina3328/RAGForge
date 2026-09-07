"""Advanced RAG strategy (Stage 2 -- the enhanced pipeline).

Builds on Naive RAG with a full query-understanding, retrieval-quality,
and generation-quality pipeline:
  1. Query understanding: intent classification -> rewrite / HyDE /
     multi-query / decomposition.
  2. Hybrid retrieval + reranking.
  3. Context compression: filter low-score results, cap the context size.
  4. Hallucination detection: verify the generated answer against what
     was actually retrieved.
"""

from __future__ import annotations

import logging

from ragforge.config import RagForgeConfig
from ragforge.generator.context_compressor import ContextCompressor
from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.generator.hallucination_detector import HallucinationDetector
from ragforge.query.intent import Intent
from ragforge.query.processed_query import ProcessedQuery
from ragforge.query.query_engine import QueryEngine
from ragforge.reranker.reranker import Reranker
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.retriever.retriever import Retriever
from ragforge.store.search_result import SearchResult

from .rag_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class AdvancedRAGStrategy(RAGStrategy):
    """Ties QueryEngine, a Retriever, a Reranker, a ContextCompressor, a
    Generator, and a HallucinationDetector together into one pipeline.

    Every collaborator except `retriever`/`generator`/`config` is optional
    (may be None) -- mirroring how each of those components is itself an
    optional, config-gated feature. When a collaborator is None, its step
    is skipped rather than the whole pipeline failing: no QueryEngine
    means no query understanding (fall back to the raw query text), no
    Reranker means keep the retriever's own ranking (just truncated to
    final_top_k), no ContextCompressor means use the reranked candidates
    as-is, no HallucinationDetector means the answer's hallucination_score
    stays at GenerationResult's default (-1, "not computed").
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        query_engine: QueryEngine | None,
        reranker: Reranker | None,
        context_compressor: ContextCompressor | None,
        hallucination_detector: HallucinationDetector | None,
        config: RagForgeConfig,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_engine = query_engine
        self.reranker = reranker
        self.context_compressor = context_compressor
        self.hallucination_detector = hallucination_detector
        self.config = config

    def get_name(self) -> str:
        return "Advanced RAG (Stage 2)"

    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the full Advanced RAG pipeline.

        Flow: query understanding -> intent branch -> pick the retrieval
        query -> hybrid retrieval (+ multi-query variants) -> reranking ->
        context compression -> generation -> hallucination detection.
        """
        # 1. Query understanding (intent classification + rewrite + HyDE +
        # multi-query + decomposition). Skipped entirely when query_engine
        # wasn't configured -- processed_query then stays None, and every
        # step below already knows how to treat that as "no query
        # understanding happened" rather than crashing on it.
        processed_query = self.query_engine.process_query(query) if self.query_engine is not None else None

        # 2. Intent branch: chitchat skips retrieval entirely -- there's no
        # knowledge-base question to answer, so there's nothing to look up.
        # Note this builds an EMPTY RetrievalResult directly instead of
        # calling self.retriever.retrieve(): retrieving here would defeat
        # the point of skipping retrieval for chitchat in the first place.
        if processed_query is not None and processed_query.intent == Intent.CHITCHAT:
            logger.info("Intent classified as chitchat, skipping retrieval")
            empty_result = RetrievalResult(query=query, results=[], total_retrieved=0)
            return self.generator.generate(query, empty_result)

        # 3. Pick the best query text for retrieval (HyDE answer > rewritten
        # query > original), then run hybrid retrieval, expanded with any
        # multi-query variants.
        retrieval_query = self._select_retrieval_query(processed_query, query)
        all_candidates = self._retrieve_with_variants(retrieval_query, processed_query, collection)

        # 5. Reranking: use the (comparatively expensive, comparatively
        # precise) Reranker to re-score and trim the candidates down to
        # final_top_k. Without a Reranker configured, fall back to just
        # truncating the retriever's own ranking.
        final_top_k = self.config.retriever.final_top_k
        if self.reranker is not None:
            reranked = self.reranker.rerank(query, all_candidates, final_top_k)
        else:
            reranked = all_candidates[:final_top_k]

        # 6. Context compression: filter out low-score results and cap the
        # total context size before it goes into the generation prompt.
        if self.context_compressor is not None:
            compressed = self.context_compressor.compress(reranked)
        else:
            compressed = reranked

        # 7. Generate the answer from the compressed context.
        retrieval_result = RetrievalResult(query=query, results=compressed, total_retrieved=len(compressed))
        result = self.generator.generate(query, retrieval_result)

        # 8. Hallucination detection: verify the generated answer against
        # the same context it was generated from, independently of
        # anything the generator itself claimed. Only runs when both a
        # detector is wired in AND the config feature flag is on.
        if self.hallucination_detector is not None and self.config.generator.enable_hallucination_detection:
            hallucination = self.hallucination_detector.detect(result.answer, compressed)
            result.hallucination_score = hallucination.hallucination_rate

        return result

    def _select_retrieval_query(self, processed_query: ProcessedQuery | None, original: str) -> str:
        """Pick the query text to retrieve with.

        Priority: HyDE's hypothetical answer > the rewritten query >
        the original query text. HyDE's generated document-style text is
        a better match for dense retrieval than a short query, because
        its semantic space sits closer to the real documents being
        searched for than a short question does.
        """
        if processed_query is None:
            return original
        if processed_query.hyde_answer:
            return processed_query.hyde_answer
        if processed_query.rewritten:
            return processed_query.rewritten
        return original

    def _retrieve_with_variants(
        self,
        retrieval_query: str,
        processed_query: ProcessedQuery | None,
        collection: str,
    ) -> list[SearchResult]:
        """Retrieve using the main query plus any multi-query variants,
        then deduplicate.

        Multi-query's core idea: the same question can be phrased several
        different ways, and retrieving separately for each phrasing
        compensates for one phrasing's semantic blind spots, improving
        recall. When the same chunk comes back from more than one
        phrasing, the higher-scoring copy is kept.
        """
        all_results: list[SearchResult] = []

        # Retrieve with the main query.
        main_result = self.retriever.retrieve(retrieval_query, collection)
        if main_result and main_result.results:
            all_results.extend(main_result.results)

        # Retrieve with each multi-query variant, if any were generated.
        # A single variant's retrieval failing shouldn't take down the
        # whole request -- log it and move on to the next variant.
        if processed_query is not None and processed_query.variants:
            for variant in processed_query.variants:
                try:
                    variant_result = self.retriever.retrieve(variant, collection)
                    if variant_result and variant_result.results:
                        all_results.extend(variant_result.results)
                except Exception:
                    logger.warning("Retrieving variant %r failed", variant, exc_info=True)

        return self._dedupe_by_chunk_id(all_results)

    @staticmethod
    def _dedupe_by_chunk_id(results: list[SearchResult]) -> list[SearchResult]:
        """Collapse duplicate chunk_ids (the same chunk retrieved by more
        than one query/variant) down to one entry each, keeping whichever
        copy has the higher score, then return them sorted by score
        descending.

        A plain dict keyed by chunk_id does the deduplication: for each
        result, only overwrite what's already in the dict if this one
        scores higher, so the dict ends up holding exactly one --
        the best -- SearchResult per chunk_id.
        """
        best_by_chunk_id: dict[str, SearchResult] = {}
        for result in results:
            existing = best_by_chunk_id.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                best_by_chunk_id[result.chunk_id] = result

        return sorted(best_by_chunk_id.values(), key=lambda r: r.score, reverse=True)
