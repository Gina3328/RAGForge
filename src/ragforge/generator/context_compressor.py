"""Context compression: trims reranked results down to what fits the
LLM's context budget, favoring the higher-scoring chunks.
"""

from __future__ import annotations

from ragforge.store.search_result import SearchResult


class ContextCompressor:
    """Filters and truncates a scored SearchResult list for a generation call.

    Runs after Reranker, on already-scored candidates -- it doesn't compute
    any new relevance score itself, only filters/sorts/truncates by the
    scores it's handed.

    Those incoming scores are NOT on one consistent scale: depending on
    which Reranker ran before this (see AdvancedRAGStrategy.execute()),
    candidates arrive scored as either the retriever's own similarity/RRF
    fusion score (NoOpReranker just passes those through -- dense cosine
    similarity is roughly in [0, 1], RRF fusion scores are typically much
    smaller, well under 0.1) or a Cross-Encoder's raw regression-head
    output (CrossEncoderReranker -- an unbounded logit, often negative,
    with no fixed range at all). A single absolute `threshold` tuned for
    one of those scales is meaningless against the others: compared
    against typical RRF scores it never lets anything through (silently
    degrading to a no-op via the "nothing survived" fallback below),
    while compared against Cross-Encoder logits it can just as easily
    reject the genuinely best-matching chunk in a batch purely because
    that batch's scores happened to cluster below the cutoff (observed in
    practice -- e.g. a warranties/liability question where the one
    correct chunk was cross-encoder-scored below 0.3 and dropped, leaving
    only an adjacent-but-wrong chunk that happened to score above it).

    To make `threshold` mean the same thing regardless of what produced
    the raw scores, compress() first min-max normalizes THIS candidate
    batch's own scores to [0, 1] -- the threshold then reads as "how far
    toward the best-scoring candidate in this batch", which is comparable
    whether the underlying scores are similarity scores, RRF scores, or
    Cross-Encoder logits.
    """

    def __init__(self, threshold: float, max_context_chars: int) -> None:
        self.threshold = threshold
        self.max_context_chars = max_context_chars

    def compress(self, candidates: list[SearchResult]) -> list[SearchResult]:
        """Filter low-score candidates, then truncate to the character budget.

        Steps:
          1. Min-max normalize this batch's scores to [0, 1], so
             `threshold` means the same thing no matter which upstream
             stage (retriever vs. reranker) produced these scores -- see
             the class docstring for why this step exists.
          2. Keep only candidates whose normalized score >= threshold.
          3. Fallback: if EVERY candidate was filtered out, return the
             original candidates unchanged -- an empty context would be
             worse than a low-quality one.
          4. Sort the survivors by their original score, descending (best
             first) -- min-max normalization is order-preserving, so
             sorting by the raw score gives the identical order sorting
             by the normalized score would.
          5. Greedily add chunks (highest score first) until the next one
             would push the total content length over max_context_chars,
             then stop -- so the budget is spent on the best-scoring chunks.
        """
        if not candidates:
            return candidates

        # Step 1: min-max normalize this batch's own scores to [0, 1].
        scores = [c.score for c in candidates]
        lo, hi = min(scores), max(scores)
        if hi > lo:
            normalized_by_id = {id(c): (c.score - lo) / (hi - lo) for c in candidates}
        else:
            # Every candidate has the identical score -- there's nothing
            # to rank between them, so treat them all as equally "best".
            normalized_by_id = {id(c): 1.0 for c in candidates}

        # Step 2: filter by normalized score threshold.
        filtered_candidates = [c for c in candidates if normalized_by_id[id(c)] >= self.threshold]

        # Fallback: nothing survived the filter -- keep the original
        # candidates rather than handing the generator an empty context.
        if not filtered_candidates:
            return candidates

        # Step 3: highest (original) score first -- equivalent to sorting
        # by normalized score, since the normalization is order-preserving.
        filtered_candidates.sort(key=lambda c: c.score, reverse=True)

        # Step 4: greedily fill the character budget, best-scoring first.
        total_chars = 0
        results: list[SearchResult] = []
        for candidate in filtered_candidates:
            if total_chars + len(candidate.content) > self.max_context_chars:
                break
            total_chars += len(candidate.content)
            results.append(candidate)

        return results
