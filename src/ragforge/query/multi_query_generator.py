"""Generates multiple differently-worded variants of a user query.

Unlike QueryRewriter (one query -> one better query) or HyDEGenerator (one
query -> one hypothetical passage), this expands a single query into
several semantically-equivalent variants, each phrased differently. The
idea is to widen recall: a single query's embedding only points in one
semantic direction, but the documents that actually answer it may use
quite different wording. Retrieving with several phrasings and merging
the results covers more of that space than any one phrasing alone.
"""

from __future__ import annotations

import logging
import re

from ragforge.llm import LlmService

logger = logging.getLogger(__name__)


class MultiQueryGenerator:
    """Generates query variants for a single query, via one LLM call."""

    _MULTI_QUERY_PROMPT_TEMPLATE = """You are a query expansion expert. Generate 3 variant questions for the following user query, each phrased differently.
Requirements:
1. Preserve the same semantic meaning as the original query.
2. Use different wording and keywords for each variant.
3. Output one variant per line, with no numbering and no other explanation.

Original query: {query}"""

    def __init__(self, llm: LlmService) -> None:
        self.llm = llm

    def generate(self, query: str) -> list[str]:
        """Return a list of reworded variants of `query`.

        Never raises, and never returns None: on LLM failure, or if the
        reply doesn't yield any usable variant lines, this returns an
        empty list rather than stuffing the original query in as a fake
        "variant". `variants` maps to ProcessedQuery.variants, which
        already defaults to an empty list -- retrieval code that loops
        over it to fire extra searches should simply do nothing extra
        when it's empty, the same as if this step hadn't run. There's no
        need to fall back to the original query here the way
        QueryRewriter does: the original query is retrieved separately
        regardless of what this returns, so re-adding it as a "variant"
        would just cause a duplicate search.
        """
        prompt = self._MULTI_QUERY_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(prompt)
        except RuntimeError:
            logger.warning(
                "Multi-query generation LLM call failed for query %r; no variants produced",
                query,
            )
            return []

        variants = self._parse_variants(response)
        if not variants:
            logger.warning(
                "Multi-query generation LLM call for query %r returned no usable variants",
                query,
            )
            return []

        return variants

    def _parse_variants(self, response: str) -> list[str]:
        """Split the LLM's reply into one variant per non-empty line.

        The prompt asks for no numbering, but small/local models don't
        always comply -- strip common list-marker prefixes ("1.", "1)",
        "-", "*", "•") so a variant isn't polluted with leftover markup.
        """
        variants = []
        for line in response.splitlines():
            cleaned = line.strip()
            cleaned = re.sub(r"^(\d+[.)]|[-*•])\s*", "", cleaned).strip()
            if cleaned:
                variants.append(cleaned)

        return variants
