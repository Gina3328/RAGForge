"""Adaptive RAG strategy (Stage 3 -- complexity-based routing).

Dynamically picks a processing path based on how complex the query is:
  - SIMPLE: a simple factual query -> dense retrieval + direct
    generation (fastest).
  - MEDIUM: needs some understanding or a couple of reasoning steps ->
    query understanding + hybrid retrieval (a balance of quality and
    speed).
  - COMPLEX: multi-hop reasoning, comparison -> query decomposition +
    per-sub-question Self-RAG (run in parallel) + synthesis (most
    thorough).

Complexity is assessed automatically by the LLM, so the retrieval
strategy adapts itself to each query instead of using one fixed pipeline
for everything.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.llm import LlmService
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.dense_retriever import DenseRetriever
from ragforge.retriever.hybrid_retriever import HybridRetriever
from ragforge.store.search_result import SearchResult

from .rag_strategy import RAGStrategy
from .self_rag_strategy import SelfRAGStrategy

logger = logging.getLogger(__name__)


@dataclass
class SubResult:
    """One sub-question's outcome, produced while answering a COMPLEX
    query broken down into several sub-questions.

    Attributes:
        query: The sub-question's text.
        answer: The answer generated for this sub-question (or an
            error message if answering it failed).
        chunks: The chunks retrieved and used to answer this sub-question.
    """

    query: str
    answer: str
    chunks: list[SearchResult] = field(default_factory=list)


class Complexity(Enum):
    """Query complexity classification, used to route to the right
    RAG path: simple direct answers, standard retrieval, or the full
    decompose-and-synthesize pipeline for harder questions.
    """

    SIMPLE = "SIMPLE"    # Simple factual lookup, answerable with a single keyword search.
    MEDIUM = "MEDIUM"    # Requires some understanding or a couple of reasoning steps.
    COMPLEX = "COMPLEX"  # Requires multi-hop reasoning, comparison, or synthesis.


class AdaptiveRAGStrategy(RAGStrategy):
    """Assesses query complexity via the LLM, then routes to the
    matching path: simple, medium, or complex.
    """

    def __init__(
        self,
        llm: LlmService,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        generator: Generator,
        query_engine: QueryEngine | None,
        self_rag_strategy: SelfRAGStrategy,
        parallel_sub_queries: bool,
        max_sub_queries: int,
    ) -> None:
        self.llm = llm
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.query_engine = query_engine
        self.self_rag_strategy = self_rag_strategy
        self.parallel_sub_queries = parallel_sub_queries
        self.max_sub_queries = max_sub_queries

    def get_name(self) -> str:
        return "Adaptive RAG (Stage 3)"

    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the Adaptive RAG flow: assess complexity, then route to
        the matching path.
        """
        logger.info("Adaptive RAG strategy executing")

        # Use the LLM to assess query complexity, which decides the path.
        complexity = self._assess_complexity(query)
        logger.info("Query complexity: %s", complexity)

        if complexity == Complexity.SIMPLE:
            return self._execute_simple(query, collection)
        if complexity == Complexity.MEDIUM:
            return self._execute_medium(query, collection)
        return self._execute_complex(query, collection)

    def _execute_simple(self, query: str, collection: str) -> GenerationResult:
        """Simple path: dense retrieval + direct generation, for the
        fastest possible response.
        """
        logger.info("Simple path: dense retrieval + direct generation")
        retrieval_result = self.dense_retriever.retrieve(query, collection)
        return self.generator.generate(query, retrieval_result)

    def _execute_medium(self, query: str, collection: str) -> GenerationResult:
        """Medium path: query understanding + hybrid retrieval + generation.

        Uses QueryEngine to improve the retrieval query text -- HyDE's
        hypothetical answer takes priority, falling back to the
        rewritten query -- then retrieves with the hybrid retriever,
        which combines semantic and keyword matching.
        """
        logger.info("Medium path: query understanding + hybrid retrieval + generation")
        retrieval_query = query
        if self.query_engine is not None:
            try:
                processed = self.query_engine.process_query(query)
                # HyDE first: a hypothetical document passage is a
                # better match for dense retrieval than a short query.
                if processed.hyde_answer:
                    retrieval_query = processed.hyde_answer
                elif processed.rewritten:
                    retrieval_query = processed.rewritten
            except Exception:
                logger.warning("Query understanding failed, retrying with the original query", exc_info=True)

        retrieval_result = self.hybrid_retriever.retrieve(retrieval_query, collection)
        return self.generator.generate(query, retrieval_result)

    def _execute_complex(self, query: str, collection: str) -> GenerationResult:
        """Complex path: decompose into sub-questions, answer each one
        with Self-RAG (in parallel or sequentially), then synthesize a
        final answer.

        For multi-hop reasoning or comparison-style queries, break the
        query down into several sub-questions, answer each one
        independently through the full Self-RAG flow (retrieval
        decision + relevance check), then use the LLM to combine all
        the sub-answers into one coherent answer.
        """
        logger.info("Complex path: query decomposition + per-sub-question Self-RAG + synthesis")

        # Decompose into sub-questions, falling back to the original
        # query as a single sub-question if decomposition isn't
        # available or fails.
        sub_queries = [query]
        if self.query_engine is not None:
            try:
                decomposed = self.query_engine.process_query(query).sub_queries
                if decomposed:
                    sub_queries = decomposed
            except Exception:
                logger.warning("Query decomposition failed, using the original query", exc_info=True)

        # Cap the number of sub-questions to keep latency bounded.
        if len(sub_queries) > self.max_sub_queries:
            sub_queries = sub_queries[: self.max_sub_queries]

        # Run Self-RAG for each sub-question, in parallel or sequentially.
        if self.parallel_sub_queries and len(sub_queries) > 1:
            sub_results = self._execute_parallel(sub_queries, collection)
        else:
            sub_results = self._execute_sequential(sub_queries, collection)

        return self._execute_synthesize(query, sub_results)

    def _execute_parallel(self, sub_queries: list[str], collection: str) -> list[SubResult]:
        """Run each sub-question's Self-RAG pass concurrently, then
        collect the results back in the original sub_queries order.

        Suited to sub-questions that don't depend on each other. Any
        single sub-question failing doesn't affect the others -- the
        failure is recorded in its own SubResult instead.
        """
        logger.info("Running %d sub-question(s) in parallel", len(sub_queries))

        def run_one(sub_query: str) -> SubResult:
            try:
                result = self.self_rag_strategy.execute(sub_query, collection)
                return SubResult(sub_query, result.answer, result.retrieved_chunks)
            except Exception as e:
                logger.error("Sub-question execution failed: %s", e)
                return SubResult(sub_query, f"Unable to answer: {e}", [])

        with ThreadPoolExecutor() as executor:
            return list(executor.map(run_one, sub_queries))

    def _execute_sequential(self, sub_queries: list[str], collection: str) -> list[SubResult]:
        """Run each sub-question's Self-RAG pass one after another (the
        fallback used when parallel execution is disabled).
        """
        logger.info("Running %d sub-question(s) sequentially", len(sub_queries))
        results: list[SubResult] = []
        for sub_query in sub_queries:
            try:
                result = self.self_rag_strategy.execute(sub_query, collection)
                results.append(SubResult(sub_query, result.answer, result.retrieved_chunks))
            except Exception as e:
                logger.error("Sub-question execution failed: %s", e)
                results.append(SubResult(sub_query, f"Unable to answer: {e}", []))
        return results

    def _execute_synthesize(self, original_query: str, sub_results: list[SubResult]) -> GenerationResult:
        """Synthesize a final answer from every sub-question's answer.

        Fallback strategy: if the LLM call to synthesize fails, fall
        back to a plain concatenation of each sub-question and its answer.
        """
        logger.info("Synthesizing a final answer from %d sub-result(s)", len(sub_results))

        prompt_parts = [
            "Based on the answers to the following sub-questions, synthesize "
            "a complete answer to the original question.\n\n",
            f"Original question: {original_query}\n\n",
        ]
        for i, sub_result in enumerate(sub_results, start=1):
            prompt_parts.append(f"Sub-question {i}: {sub_result.query}\nAnswer {i}: {sub_result.answer}\n\n")
        prompt_parts.append("Please synthesize the information above into a complete, coherent answer:")

        try:
            synthesized = self.llm.generate("".join(prompt_parts))
            all_chunks = [chunk for sub_result in sub_results for chunk in sub_result.chunks]
            return GenerationResult(
                answer=synthesized,
                cited_chunks=[],
                hallucination_score=-1.0,
                retrieved_chunks=all_chunks,
            )
        except RuntimeError:
            logger.error("Synthesis failed, falling back to a plain concatenation of sub-answers", exc_info=True)
            if sub_results:
                fallback = "\n\n".join(f"Q: {sr.query}\nA: {sr.answer}" for sr in sub_results)
            else:
                fallback = "Unable to generate a synthesized answer."
            return GenerationResult(
                answer=fallback,
                cited_chunks=[],
                hallucination_score=-1.0,
                retrieved_chunks=[],
            )

    def _assess_complexity(self, query: str) -> Complexity:
        """Assess `query`'s complexity via the LLM, to decide which
        path to route through. Defaults to MEDIUM on error -- a safe
        middle ground that doesn't wrongly commit to the cheapest or
        the most expensive path.
        """
        try:
            prompt = f"""Assess the complexity of the following question:
- SIMPLE: a simple factual lookup, answerable with a single keyword search.
- MEDIUM: requires some understanding or a couple of reasoning steps.
- COMPLEX: requires multi-hop reasoning, comparison, or synthesizing
  information across multiple sources.

Question: {query}
Output only the level name."""

            response = self.llm.generate(prompt).strip().upper()
            if "COMPLEX" in response:
                return Complexity.COMPLEX
            if "MEDIUM" in response:
                return Complexity.MEDIUM
            return Complexity.SIMPLE
        except RuntimeError:
            logger.warning("Complexity assessment failed, defaulting to MEDIUM", exc_info=True)
            return Complexity.MEDIUM
