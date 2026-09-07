"""Aggregated result of QueryEngine.process().

Captures the user's original query plus whatever optional processing
steps (intent classification, rewriting, HyDE, multi-query expansion,
decomposition) were actually run for it. Not every query goes through
every step -- IntentClassifier's result decides which of the other steps
QueryEngine runs (e.g. CHITCHAT skips retrieval entirely). Fields for
steps that didn't run stay at their default (None, or an empty list for
the two list-valued fields), so downstream code can check "did this step
run?" by checking for None/empty instead of re-deriving it from intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .intent import Intent


@dataclass
class ProcessedQuery:
    """Everything QueryEngine knows about one user query after processing."""

    # The user's original, unmodified query text. Always set.
    original: str

    # Result of IntentClassifier.classify(). None until that step has run.
    intent: Intent | None = None

    # Query rewritten into a clearer, more retrieval-friendly form by
    # QueryRewriter. None if that step didn't run.
    rewritten: str | None = None

    # LLM-generated hypothetical answer from HyDEGenerator, used in place
    # of the query text itself when embedding for dense retrieval (HyDE).
    # None if that step didn't run.
    hyde_answer: str | None = None

    # Alternate phrasings of the query from MultiQueryGenerator, used to
    # broaden retrieval recall. Empty list if that step didn't run.
    variants: list[str] = field(default_factory=list)

    # Sub-questions the original query was split into by QueryDecomposer
    # (e.g. for COMPARISON-intent queries). Empty list if that step didn't
    # run.
    sub_queries: list[str] = field(default_factory=list)
