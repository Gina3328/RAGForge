"""Classifies a user query's intent.

QueryEngine uses the result to decide which of the other query-
understanding steps (rewrite / HyDE / multi-query / decomposition) to run
for a given query, and later which retrieval strategy to route it to.
"""

from __future__ import annotations

import logging

from ragforge.llm import LlmService

from .intent import Intent

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies a user query into one of the four Intent categories."""

    _CLASSIFY_PROMPT_TEMPLATE = """You are a query intent classifier. Classify the following user query into exactly one of four intents:
- FACTUAL: a factual query, looking for a specific fact, definition, or piece of data.
- PROCEDURAL: a procedural query, asking how to perform a task or what the steps of a process are.
- COMPARISON: a comparison query, comparing similarities or differences between two or more things.
- CHITCHAT: casual conversation that doesn't require retrieval.

User query: {query}

Reply with only the intent category name (FACTUAL / PROCEDURAL / COMPARISON / CHITCHAT) and nothing else."""

    # Safe default when the LLM call fails, or its reply can't be matched
    # to a known Intent. FACTUAL still routes into normal retrieval
    # (unlike CHITCHAT, which skips it) -- an unnecessary extra retrieval
    # is a much cheaper mistake than silently skipping one a question
    # actually needed.
    _FALLBACK_INTENT = Intent.FACTUAL

    def __init__(self, llm: LlmService) -> None:
        self.llm = llm

    def classify(self, query: str) -> Intent:
        """Classify `query` into one of the four Intent categories.

        Never raises: if the LLM call itself fails, or its reply can't be
        parsed into a known Intent, this falls back to _FALLBACK_INTENT
        rather than letting a classification miss take down the whole
        query pipeline.
        """
        prompt = self._CLASSIFY_PROMPT_TEMPLATE.format(query=query)

        try:
            response = self.llm.generate(prompt)
        except RuntimeError:
            logger.warning(
                "Intent classification LLM call failed for query %r; falling back to %s",
                query, self._FALLBACK_INTENT,
            )
            return self._FALLBACK_INTENT

        return self._parse_intent(response)

    def _parse_intent(self, response: str) -> Intent:
        """Match the LLM's raw reply text to an Intent value.

        The prompt asks for just the category name, but small models
        don't always comply exactly -- the reply may carry extra
        whitespace, punctuation, different casing, or a short explanation
        wrapped around the label (e.g. "FACTUAL." or "This is FACTUAL.").
        Rather than requiring an exact match, this looks for each
        Intent's value as a substring of the cleaned, uppercased reply --
        none of the four labels are substrings of each other, so this
        can't misfire between them.
        """
        cleaned = response.strip().upper()

        for intent in Intent:
            if intent.value in cleaned:
                return intent

        logger.warning(
            "Could not parse an Intent from LLM reply %r; falling back to %s",
            response, self._FALLBACK_INTENT,
        )
        return self._FALLBACK_INTENT
