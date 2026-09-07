"""Data class for the result of a hallucination-detection pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HallucinationResult:
    """The outcome of checking a generated answer against its retrieved
    context for unsupported claims.

    Attributes:
        claims: Every factual statement the answer was split into.
        hallucination_rate: unsupported_claims count / claims count.
            0.0 means every claim was supported (or there was nothing to
            check). -1.0 is a distinct sentinel meaning detection itself
            failed (an LLM call errored, or its output couldn't be
            parsed) -- NOT "zero hallucination found".
        unsupported_claims: The subset of `claims` that could not be
            verified against the retrieved context.
    """

    claims: list[str]
    hallucination_rate: float
    unsupported_claims: list[str]
