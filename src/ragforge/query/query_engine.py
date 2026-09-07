"""Orchestrates the query-understanding pipeline.

QueryEngine classifies a query's intent, then runs whichever of
rewrite / HyDE / multi-query / decomposition that intent calls for,
assembling the results into a ProcessedQuery for the retrieval layer.
"""

from __future__ import annotations

from ragforge.llm import LlmService

from .hyde_generator import HyDEGenerator
from .intent import Intent
from .intent_classifier import IntentClassifier
from .multi_query_generator import MultiQueryGenerator
from .processed_query import ProcessedQuery
from .query_decomposer import QueryDecomposer
from .query_rewriter import QueryRewriter


class QueryEngine:
    """Ties the query-understanding components together, routed by intent."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        query_rewriter: QueryRewriter,
        hyde_generator: HyDEGenerator,
        multi_query_generator: MultiQueryGenerator,
        query_decomposer: QueryDecomposer,
    ) -> None:
        """Build the engine from its five already-constructed components.

        QueryEngine's job is to orchestrate these five -- deciding which
        ones a given query needs, in what order, and how to assemble their
        results -- not to also be responsible for constructing them. Taking
        them as parameters (rather than building them internally from a
        raw LlmService) keeps that separation clean, and makes the routing
        logic in process_query() easy to unit test: a test can pass in
        five lightweight fakes with canned return values, instead of
        faking raw LLM text replies that each real component would need to
        correctly parse.

        For everyday construction (e.g. wiring the app up at startup,
        where there's no need for that flexibility), use the from_llm()
        classmethod below instead of calling this directly.
        """
        self.intent_classifier = intent_classifier
        self.query_rewriter = query_rewriter
        self.hyde_generator = hyde_generator
        self.multi_query_generator = multi_query_generator
        self.query_decomposer = query_decomposer

    @classmethod
    def from_llm(cls, llm: LlmService) -> QueryEngine:
        """Convenience constructor: build all five components from one
        shared LlmService.

        All five sub-components currently take nothing but an LlmService
        to construct, so this is just a shortcut for the common case --
        equivalent to constructing all five yourself and passing them to
        __init__(), but without the boilerplate of doing so at every call
        site. All five end up sharing the same LlmService instance, and
        therefore the same underlying anthropic.Anthropic() client.
        """
        return cls(
            intent_classifier=IntentClassifier(llm),
            query_rewriter=QueryRewriter(llm),
            hyde_generator=HyDEGenerator(llm),
            multi_query_generator=MultiQueryGenerator(llm),
            query_decomposer=QueryDecomposer(llm),
        )

    def process_query(self, query: str) -> ProcessedQuery:
        """Run the query-understanding pipeline for `query`.

        Always classifies intent first, since that decides which of the
        other four steps are worth running:

        - CHITCHAT: nothing else runs. Retrieval is going to be skipped
          entirely downstream, so rewriting/HyDE/multi-query/decomposition
          would just be wasted LLM calls -- the returned ProcessedQuery
          only has `original` and `intent` set, everything else keeps its
          default (None / empty list).
        - FACTUAL / PROCEDURAL: rewrite and HyDE both run -- they improve
          retrieval quality regardless of the specific topic -- plus
          MultiQueryGenerator, since paraphrase variants help find a single
          matching fact/procedure phrased differently than the user asked.
          Decomposition doesn't run: a single fact or procedure isn't made
          of independent sub-questions the way a comparison is.
        - COMPARISON: rewrite and HyDE still run, but QueryDecomposer
          replaces MultiQueryGenerator -- splitting "X vs Y" into targeted
          sub-questions already covers multiple retrieval angles, so also
          generating generic paraphrase variants on top would just be a
          redundant extra LLM call.

        rewrite / HyDE / multi-query / decomposition are all called with
        the original `query`, not chained off each other's output: per the
        retrieval-time fallback chain (hyde_answer -> rewritten ->
        original), they're independent, equally-ranked candidates for the
        dense-retrieval query text, not a pipeline where one depends on
        another's result.
        """
        intent = self.intent_classifier.classify(query)

        if intent == Intent.CHITCHAT:
            return ProcessedQuery(original=query, intent=intent)

        rewritten = self.query_rewriter.rewrite(query)
        hyde_answer = self.hyde_generator.generate(query)

        if intent == Intent.COMPARISON:
            variants = []
            sub_queries = self.query_decomposer.decompose(query)
        else:
            variants = self.multi_query_generator.generate(query)
            sub_queries = []

        return ProcessedQuery(
            original=query,
            intent=intent,
            rewritten=rewritten,
            hyde_answer=hyde_answer,
            variants=variants,
            sub_queries=sub_queries,
        )
