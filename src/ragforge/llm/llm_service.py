"""Shared helper for one-shot LLM text completions.

Wraps the single Claude API call that IntentClassifier, QueryRewriter,
HyDEGenerator, MultiQueryGenerator, and QueryDecomposer all need: send one
user-role prompt, get back the model's text reply. Centralizing it here
avoids each of those classes repeating client setup and error handling,
mirroring how EmbeddingService centralizes embedding calls for the
chunker strategies.

Deliberately thin: `generate()` either returns the model's text or raises
-- it does NOT catch the API error and substitute a fallback value,
because what counts as a safe fallback differs per caller (e.g.
IntentClassifier wants to fall back to Intent.FACTUAL, QueryRewriter
wants to fall back to the original, un-rewritten query). Each caller
decides that for itself; this class only reports success or failure.
"""

from __future__ import annotations

import anthropic

from ragforge.config import LlmConfig

# Default reply length cap. Matches SimpleGenerator.MAX_TOKENS for
# consistency -- note this is just a ceiling, not a target: Claude is
# billed by tokens actually generated, not by this cap, so there's no
# cost downside to leaving headroom for callers that need a longer reply
# (e.g. HyDE's hypothetical-document generation). Callers needing more
# than this can still pass a larger max_tokens explicitly.
DEFAULT_MAX_TOKENS = 1024


class LlmService:
    """Thin wrapper around a single Claude text-completion call."""

    def __init__(self, config: LlmConfig) -> None:
        # anthropic.Anthropic() reads ANTHROPIC_API_KEY from the
        # environment (see SimpleGenerator) -- config.base_url/api_key/
        # timeout aren't used here either, for the same reason they
        # aren't used there: this project's LLM calls go to the Claude
        # API, not the local Ollama endpoint those fields describe.
        self.client = anthropic.Anthropic()
        self.model = config.model

    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send `prompt` as a single user message and return the reply text.

        Raises RuntimeError (wrapping the original anthropic.APIError) if
        the request fails, so callers don't need to import anthropic
        themselves just to catch it. Callers are responsible for deciding
        what a safe fallback looks like for their own use case.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            raise RuntimeError(f"LLM completion request failed: {e}") from e

        return response.content[0].text
