"""Hallucination detection (Stage 2/3).

Checks whether a generated answer contains "hallucinations" -- factual
statements with no support in the retrieved context. Two-step process:
  1. Ask the LLM to split the answer into independent factual statements
     (claims).
  2. Verify each claim, one at a time, against the retrieved context.
The final hallucination rate = unsupported claims / total claims.
"""

from __future__ import annotations

import logging
import re

from ragforge.generator.hallucination_result import HallucinationResult
from ragforge.llm.llm_service import LlmService
from ragforge.store.search_result import SearchResult

# Module-level logger, named after this module (e.g.
# "ragforge.generator.HallucinationDetector") so log lines show exactly
# which component produced them.
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Splits a generated answer into claims and verifies each against
    the retrieved context, independently of anything CitationGenerator's
    [n] markers claimed -- this re-derives and re-checks from scratch.
    """

    SPLIT_PROMPT = """Split the following answer into a list of independent factual statements.
    Each statement must be a complete fact, without any reasoning or speculation.

    Answer: {answer}

    Output as a JSON array: ["statement1", "statement2", ...]
    """

    VERIFY_PROMPT = """Is the following factual statement supported by the retrieved context?

    Statement: {claim}
    Context: {context}

    Answer only YES or NO.
    """

    JSON_ARRAY_PATTERN = re.compile(r'\["[\s\S]*?]')
    STRING_PATTERN = re.compile(r'"([^"]+)"')

    def __init__(self, llm_service: LlmService) -> None:
        self.llm_service = llm_service

    def detect(self, answer: str, chunks: list[SearchResult]) -> HallucinationResult:
        """Check whether `answer` contains claims unsupported by `chunks`.

        Two-phase process, both phases going through the LLM:
          1. `_split_claims` breaks the answer into independent factual
             statements.
          2. Each claim is verified, one at a time, against the top 5
             retrieved chunks (not the full context, and not all of
             `chunks`) -- chunks are already ranked by relevance, so the
             highest-scoring few are the ones most likely to actually
             support (or fail to support) a claim, and keeping the
             verification prompt short matters since this makes one LLM
             call PER claim.

        hallucination_rate is 0.0 when every claim was supported (or
        there was nothing to check), and the distinct sentinel -1.0 when
        detection itself failed (an LLM call errored, or a response
        couldn't be parsed) -- see HallucinationResult's docstring for
        why these two are kept separate.
        """
        # If there's no answer or no context to check it against, there's
        # nothing to detect -- skip the LLM calls entirely.
        if not answer or not chunks:
            return HallucinationResult(claims=[], hallucination_rate=0.0, unsupported_claims=[])

        try:
            claims = self._split_claims(answer)
            if not claims:
                # An empty list here just means "nothing to check" (or
                # _split_claims already degraded and logged a warning
                # about why) -- not itself a detection failure, so 0.0,
                # not -1.0.
                return HallucinationResult(claims=[], hallucination_rate=0.0, unsupported_claims=[])

            context = "\n---\n".join(chunk.content for chunk in chunks[:5])

            # Verify each claim, one at a time, against that context.
            unsupported_claims: list[str] = []
            for claim in claims:
                prompt = self.VERIFY_PROMPT.format(claim=claim, context=context)
                response = self.llm_service.generate(prompt).strip().upper()
                if "YES" not in response:
                    unsupported_claims.append(claim)

            # Hallucination rate = unsupported claims / total claims.
            rate = len(unsupported_claims) / len(claims)
            logger.info(
                "Hallucination detection: %d claims, %d unsupported, rate=%.2f",
                len(claims),
                len(unsupported_claims),
                rate,
            )

            return HallucinationResult(claims=claims, hallucination_rate=rate, unsupported_claims=unsupported_claims)
        except Exception:
            # Any failure here (an LLM call erroring mid-verification,
            # etc.) means detection itself could not complete -- distinct
            # from "ran fine, found nothing wrong". -1.0 signals that to
            # the caller, matching HallucinationResult's documented
            # sentinel.
            logger.error("Hallucination detection failed", exc_info=True)
            return HallucinationResult(claims=[], hallucination_rate=-1.0, unsupported_claims=[])

    def _split_claims(self, answer: str) -> list[str]:
        """Ask the LLM to split `answer` into independent factual
        statements, then defensively parse the JSON array it replies with.

        The LLM's reply isn't guaranteed to be clean JSON -- it may wrap
        the array in markdown fences or add extra commentary -- so this
        first locates the "[...]"-shaped substring (JSON_ARRAY_PATTERN),
        then pulls each quoted string out of THAT substring
        (STRING_PATTERN), rather than trying to json.loads() the whole
        reply directly.

        Returns an empty list if the LLM call fails, or if no parseable
        array could be found -- callers treat "no claims found" the same
        as "nothing to check", not as an error.
        """
        try:
            prompt = self.SPLIT_PROMPT.format(answer=answer)
            response = self.llm_service.generate(prompt)

            array_match = self.JSON_ARRAY_PATTERN.search(response)
            if array_match:
                array_str = array_match.group()
                claims: list[str] = []
                for str_match in self.STRING_PATTERN.finditer(array_str):
                    claims.append(str_match.group(1))
                return claims
        except Exception:
            # Same graceful-degradation contract as the rest of this
            # project: swallow the error (an LLM call failure, a bad
            # response format, etc.) and fall through to `return []`
            # below, rather than letting it crash the caller. We still
            # log it -- exc_info=True attaches the full traceback -- so
            # the failure is visible in logs instead of silently vanishing.
            logger.warning("Failed to split claims from the LLM response", exc_info=True)

        return []
