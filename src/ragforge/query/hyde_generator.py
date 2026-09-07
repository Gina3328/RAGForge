"""Generates a hypothetical document passage for a query (HyDE).

Short queries embed poorly against long document chunks -- their
vocabulary and structure differ too much from the documents they're
meant to match. HyDE works around this by having the LLM write a
plausible-looking passage that might contain the answer (it doesn't need
to be factually correct, only stylistically similar to a real document)
and embedding that instead of the raw query. See:
Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (ACL 2023), https://arxiv.org/abs/2212.10496.
"""

from __future__ import annotations

import logging

from ragforge.llm import LlmService

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """Generates a hypothetical answer passage for a query, for embedding."""

    _HYDE_PROMPT_TEMPLATE = """Given the following question, write a hypothetical passage of text that might contain the answer.
Note: the content does not need to be factually accurate, but it should read like an excerpt from a real, professionally written document -- precise and matter-of-fact in style.
Output only the passage itself, with no explanation or preamble.

Question: {query}"""

    def __init__(self, llm: LlmService) -> None:
        self.llm = llm

    def generate(self, query: str) -> str | None:
        """Return a hypothetical document passage for `query`, or None.

        Unlike QueryRewriter.rewrite() (which falls back to the original
        query on failure, since a rewritten and an un-rewritten query
        play the same role downstream), this returns None -- not the
        original query -- when the LLM call fails or its reply is empty.
        That's deliberate: HybridRetriever's dense-embedding step prefers
        hyde_answer, then falls back to rewritten, then to the original
        query. If this quietly returned the raw query as a stand-in
        "hyde_answer", it would short-circuit that chain and skip the
        rewritten-query tier entirely. Returning None instead lets
        ProcessedQuery.hyde_answer correctly stay unset, so that
        fallback chain still works as designed.
        """
        prompt = self._HYDE_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(prompt)
        except RuntimeError:
            logger.warning(
                "HyDE generation LLM call failed for query %r; no hypothetical answer produced",
                query,
            )
            return None

        hyde_answer = response.strip()
        if not hyde_answer:
            logger.warning(
                "HyDE generation LLM call for query %r returned an empty reply; no hypothetical answer produced",
                query,
            )
            return None

        return hyde_answer
