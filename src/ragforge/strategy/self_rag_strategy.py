"""Self-RAG strategy (Stage 3).

Wraps retrieval and generation with an LLM self-assessment loop, adding
three decision points on top of the Naive/Advanced RAG pipeline:
  1. Retrieval decision (_should_retrieve): ask the LLM whether the query
     needs the knowledge base at all, or can be answered directly.
  2. Relevance check (_is_relevant): ask the LLM whether what was
     retrieved actually contains enough information to answer the query.
  3. Retry: when retrieval isn't relevant, retry with a better query --
     first a rewritten query, then a HyDE hypothetical-answer query --
     up to max_retries times before falling back to the last attempt.
"""

from __future__ import annotations

import logging

from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.llm import LlmService
from ragforge.query.hyde_generator import HyDEGenerator
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.retriever.retriever import Retriever

from .rag_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class SelfRAGStrategy(RAGStrategy):
    """Adds an LLM self-assessment loop around retrieval and generation.

    Unlike AdvancedRAGStrategy, every collaborator here is required (not
    optional) -- Self-RAG's design depends on all three decision points
    (retrieval, rewrite-retry, HyDE-retry) actually being available.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        llm: LlmService,
        query_engine: QueryEngine,
        hyde_generator: HyDEGenerator,
        max_retries: int,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.llm = llm
        self.query_engine = query_engine
        self.hyde_generator = hyde_generator
        self.max_retries = max_retries

    def get_name(self) -> str:
        return "Self-RAG (Stage 3)"

    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the Self-RAG flow: decide whether to retrieve, retrieve,
        assess relevance, retry with a better query if needed, generate.
        """
        logger.info("Self-RAG strategy executing for query: %r", query)

        # 1. Ask the LLM whether this query needs the knowledge base at
        # all. Skip retrieval entirely for common-sense/chitchat queries.
        if not self._should_retrieve(query):
            logger.info("Retrieval not needed, generating directly")
            empty_result = RetrievalResult(query=query, results=[], total_retrieved=0)
            return self.generator.generate(query, empty_result)

        logger.info("Retrieval needed")
        retrieval_query = query
        retrieval_result = self.retriever.retrieve(retrieval_query, collection)

        # 2. Relevance check + retry loop. The first retry rewrites the
        # query; every retry after that uses a HyDE hypothetical answer.
        for attempt in range(self.max_retries):
            if self._is_relevant(query, retrieval_result):
                logger.info("Retrieval relevant on attempt %d", attempt + 1)
                return self.generator.generate(query, retrieval_result)

            logger.info("Retrieval not relevant on attempt %d", attempt + 1)
            if attempt == 0:
                rewritten = self.query_engine.process_query(query).rewritten
                if rewritten:
                    retrieval_query = rewritten
                    logger.info("Retry strategy: using rewritten query")
            else:
                hyde_answer = self.hyde_generator.generate(query)
                if hyde_answer:
                    retrieval_query = hyde_answer
                    logger.info("Retry strategy: using HyDE hypothetical answer")

            retrieval_result = self.retriever.retrieve(retrieval_query, collection)

        # Retries exhausted -- generate from whatever the last attempt
        # retrieved rather than failing outright.
        logger.info("Retries exhausted, generating from the last retrieval")
        return self.generator.generate(query, retrieval_result)

    def _should_retrieve(self, query: str) -> bool:
        """Ask the LLM whether `query` needs the knowledge base.

        Common-sense questions, small talk, and simple calculations don't
        need retrieval; questions about company policy, technical
        documentation, or business processes do. Defaults to True on
        error -- an unnecessary retrieval is cheaper than a missed one.
        """
        try:
            prompt = f"""Determine whether the following question requires retrieving the
knowledge base to answer it.

Common-sense questions, small talk, and simple calculations do not
require retrieval. Questions about company policies, technical
documentation, or business processes do require retrieval.

Question: {query}
Answer only YES or NO."""

            response = self.llm.generate(prompt)
            return response.strip().upper().startswith("YES")
        except RuntimeError:
            logger.warning("Retrieval-decision call failed, defaulting to True", exc_info=True)
            return True

    def _is_relevant(self, query: str, result: RetrievalResult | None) -> bool:
        """Ask the LLM whether `result` contains enough information to
        answer `query`.

        Uses the top 3 retrieved chunks. Defaults to True on error, to
        avoid retrying unnecessarily.
        """
        if result is None or not result.results:
            return False

        try:
            chunks = "\n---\n".join(r.content for r in result.results[:3])
            prompt = f"""Do the retrieved results below contain enough information to answer
the user's question?

Question: {query}
Retrieved results: {chunks}

Answer only RELEVANT or IRRELEVANT."""

            response = self.llm.generate(prompt)
            # NOTE: check startswith(), not `in` -- "IRRELEVANT" contains
            # "RELEVANT" as a substring, so a plain `in` check would read
            # every IRRELEVANT verdict as relevant.
            return response.strip().upper().startswith("RELEVANT")
        except RuntimeError:
            logger.warning("Relevance-check call failed, defaulting to True", exc_info=True)
            return True
