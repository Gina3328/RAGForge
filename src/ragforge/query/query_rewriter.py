"""Rewrites a user query into a clearer, more retrieval-friendly form.

Unlike IntentClassifier, this doesn't need to parse the LLM's reply into
a fixed set of values -- the output is free text, used as-is (after
basic cleanup) as the query text QueryEngine hands off to retrieval.
"""

from __future__ import annotations

import logging

from ragforge.llm import LlmService

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites a user query for better retrieval, via a single LLM call."""

    _REWRITE_PROMPT_TEMPLATE = """You are a query rewriting expert. Rewrite the following user query into a form better suited for information retrieval.
Requirements:
1. Preserve the core intent of the original query.
2. Add any necessary context.
3. Make the phrasing clearer and more explicit.
4. Expand relevant keywords where appropriate to improve retrieval recall.
5. Output only the rewritten query, and nothing else.

Original query: {query}"""

    def __init__(self, llm: LlmService) -> None:
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """Return a retrieval-friendly rewrite of `query`.

        Never raises, and never returns an empty string: if the LLM call
        fails, or its reply turns out to be empty/whitespace-only, this
        falls back to the original, un-rewritten `query` -- worst case,
        retrieval behaves as if this step hadn't run at all, rather than
        the pipeline breaking or a blank query reaching retrieval.
        """
        prompt = self._REWRITE_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(prompt)
        except RuntimeError:
            logger.warning(
                "Rewrite LLM call failed for query %r; falling back to the original query",
                query,
            )
            return query

        rewritten = response.strip()
        if not rewritten:
            logger.warning(
                "Rewrite LLM call for query %r returned an empty reply; falling back to the original query",
                query,
            )
            return query

        return rewritten
