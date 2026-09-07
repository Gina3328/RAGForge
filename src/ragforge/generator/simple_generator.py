"""Simple single-turn RAG answer generator backed by the Claude API."""

from __future__ import annotations

import anthropic

from ragforge.config import LlmConfig
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.store.search_result import SearchResult

from .generation_result import GenerationResult
from .generator import Generator


class SimpleGenerator(Generator):
    """Generates answers by sending retrieved context + query to Claude."""

    SYSTEM_PROMPT = (
        "You are a professional knowledge-base QA assistant. Answer the "
        "user's question using only the retrieved context provided.\n"
        "Requirements:\n"
        "1. Answer strictly based on the given context; do not make up information.\n"
        "2. If the context has no relevant information, say so clearly.\n"
        "3. Keep answers concise, accurate, and well-organized.\n"
        "4. Cite key information from the context where appropriate."
    )
    MAX_TOKENS = 1024

    def __init__(self, config: LlmConfig) -> None:
        self.client = anthropic.Anthropic()
        self.model = config.model

    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """Build a prompt from the retrieved context and ask Claude to answer it."""
        user_prompt = self._build_prompt(query, retrieval_result.results)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            answer = response.content[0].text
        except anthropic.APIError as e:
            answer = f"Sorry, an error occurred while generating the answer: {e}"

        # Stage 1: cited_chunks stays empty, hallucination_score stays -1.
        return GenerationResult(
            answer=answer,
            cited_chunks=[],
            hallucination_score=-1.0,
            retrieved_chunks=retrieval_result.results,
        )

    @staticmethod
    def _build_prompt(query: str, chunks: list[SearchResult]) -> str:
        """Assemble the retrieved chunks and the user's question into one prompt."""
        if chunks:
            lines = ["[Retrieved context]"]
            for i, chunk in enumerate(chunks, start=1):
                lines.append(f"[{i}] {chunk.content}\n")
            context = "\n".join(lines)
        else:
            context = "[No relevant context retrieved]"

        return f"{context}\n[User question]\n{query}"
