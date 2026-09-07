"""Decomposes a complex query into independent, separately-answerable
sub-questions.

Complementary to MultiQueryGenerator, but with a different goal: instead of
several equivalent rephrasings of the same question (for wider recall
around the same underlying topic), this splits one complex question --
especially comparison-type ones -- into sub-questions that each cover a
distinct part of it, so each can be retrieved (and later answered) against
its own matching documents, then the sub-answers combined into a full
answer to the original query.
"""

from __future__ import annotations

import logging
import re

from ragforge.llm import LlmService

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Splits a complex query into independent sub-questions, via one LLM call."""

    _DECOMPOSE_PROMPT_TEMPLATE = """You are a query decomposition expert. Decompose the following complex query into multiple independent sub-questions.
Requirements:
1. Each sub-question should be independently answerable.
2. Together, the sub-questions should cover every aspect of the original query.
3. If the original query is already simple and doesn't need decomposition, just return it as-is.
4. Output one sub-question per line, with no numbering and no other explanation.

Original query: {query}"""

    def __init__(self, llm: LlmService) -> None:
        self.llm = llm

    def decompose(self, query: str) -> list[str]:
        """Return a list of independent sub-questions covering `query`.

        Never raises, and never returns None: on LLM failure, or if the
        reply doesn't yield any usable sub-question lines, this returns an
        empty list -- the same fallback philosophy as MultiQueryGenerator.
        `sub_queries` maps to ProcessedQuery.sub_queries, which already
        defaults to an empty list. QueryEngine should treat an empty list
        as "decomposition didn't produce anything usable" and just retrieve
        normally with the rewritten/original query instead, rather than
        this class faking a single-item list that just repeats the raw
        query.

        Note requirement 3 in the prompt: for a query that's already
        simple, the LLM is asked to return it as-is rather than force a
        split. That's a legitimate, successful result -- a one-item list
        -- and is different from the failure case above.
        """
        prompt = self._DECOMPOSE_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(prompt)
        except RuntimeError:
            logger.warning(
                "Query decomposition LLM call failed for query %r; no sub-questions produced",
                query,
            )
            return []

        sub_queries = self._parse_sub_queries(response)
        if not sub_queries:
            logger.warning(
                "Query decomposition LLM call for query %r returned no usable sub-questions",
                query,
            )
            return []

        return sub_queries

    def _parse_sub_queries(self, response: str) -> list[str]:
        """Split the LLM's reply into one sub-question per non-empty line.

        Same approach as MultiQueryGenerator._parse_variants(): strip
        common list-marker prefixes ("1.", "1)", "-", "*", "•") in case the
        model ignores the "no numbering" instruction.
        """
        sub_queries = []
        for line in response.splitlines():
            cleaned = line.strip()
            cleaned = re.sub(r"^(\d+[.)]|[-*•])\s*", "", cleaned).strip()
            if cleaned:
                sub_queries.append(cleaned)

        return sub_queries
