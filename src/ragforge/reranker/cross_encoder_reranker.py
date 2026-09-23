"""Cross-Encoder based reranking, run locally via sentence-transformers.

Loads a real Cross-Encoder model directly in-process for reranking --
no external reranking service or API call involved at all.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from ragforge.store.search_result import SearchResult

from .reranker import Reranker


class CrossEncoderReranker(Reranker):
    """Reranks candidates with a locally-loaded Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Load the Cross-Encoder model once, up front.

        Model loading is a heavy operation (possibly downloading weights
        from Hugging Face on first use, then loading them into memory), so
        it happens exactly once here in __init__ -- not per rerank() call.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Score every candidate against the query with the Cross-Encoder,
        re-sort by that score, and return the top_k.

        Falls back to the original candidates (truncated to top_k, order
        unchanged) if scoring fails for any reason -- e.g. the model isn't
        loaded correctly, or a candidate's content trips up inference.
        A reranking failure should never break the whole request, just
        skip the precision boost this step was meant to add.
        """
        if not candidates:
            return []

        try:
            # Cross-Encoder scoring needs the query paired with EACH
            # candidate's content -- that's the whole point of "cross"
            # encoding (see Reranker's docstring): the model looks at
            # query and document together, not as two separate vectors.
            pairs = [(query, candidate.content) for candidate in candidates]
            scores = self.model.predict(pairs)

            reranked = [
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=float(score),
                    metadata=candidate.metadata,
                )
                for candidate, score in zip(candidates, scores)
            ]
            reranked.sort(key=lambda r: r.score, reverse=True)
            return reranked[:top_k]

        except Exception:
            return self._degrade(candidates, top_k)

    def _degrade(self, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Graceful fallback: return the original candidates, truncated to
        top_k, with their original order/scores untouched."""
        return candidates[:top_k]
