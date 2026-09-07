"""Text embedding service backed by the Ollama embedding API."""

from __future__ import annotations

import requests

from ragforge.config import EmbeddingConfig


class EmbeddingService:
    """Turns text into vectors by calling Ollama's embedding API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.base_url = config.base_url
        self.model = config.model
        # requests.timeout accepts a (connect_timeout, read_timeout) tuple.
        self.timeout = (config.timeout, config.timeout)

    def embed(self, text: str) -> list[float]:
        """Embed a single text (convenience wrapper around embed_batch)."""
        # Wrap the single text in a one-element list so we can reuse the
        # batch implementation, then unwrap its only result.
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single API call.

        Returns one vector per input text, in the same order as `texts`.
        """
        # Ollama's /api/embed expects {"model": ..., "input": [...]}.
        payload = {"model": self.model, "input": texts}

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            # Network-level failure: connection refused, timeout, DNS, etc.
            raise RuntimeError("Embedding request failed") from e

        if not response.ok:
            # HTTP-level failure: request reached the server but it
            # responded with an error status code.
            raise RuntimeError(f"Embedding request failed: {response.status_code}")

        # requests.Response.json() parses the response body straight into
        # Python dicts/lists, no manual tree-walking needed.
        data = response.json()
        embeddings = data["embeddings"]

        # Coerce every value to float in case the JSON numbers decoded as int.
        return [[float(v) for v in emb] for emb in embeddings]
