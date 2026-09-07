"""Answer generation with inline [n] citation markers, tracing each cited
chunk back to its chunk_id.
"""

from __future__ import annotations

import re

from ragforge.llm.llm_service import LlmService
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.store.search_result import SearchResult

from .generation_result import GenerationResult
from .generator import Generator


class CitationGenerator(Generator):
    """Generates an answer that cites its sources with [n] markers.

    A drop-in alternative to SimpleGenerator -- both implement Generator,
    so either can be used wherever a Generator is expected. The difference
    is entirely in the prompt (ask the LLM to mark sources as [n]) plus a
    post-processing step that maps those markers back to chunk_ids.
    """

    CITATION_PROMPT = (
        "You are a professional knowledge-base QA assistant. Answer the "
        "user's question based on the reference materials provided.\n"
        "\n"
        "Requirements:\n"
        "1. Answer the question as completely and accurately as possible "
        "based on the reference materials.\n"
        "2. Wherever possible, cite the source of each piece of information "
        "using the format [number], where the number corresponds to the "
        "reference material's number below.\n"
        "3. If the reference materials contain no relevant information, "
        "state that clearly.\n"
        "4. Keep the answer concise, accurate, and well-organized.\n"
        "\n"
        "Reference materials:\n"
        "{context}\n"
        "\n"
        "User question: {query}\n"
    )

    # Matches citation markers like "[1]", "[12]" in the LLM's answer.
    CITATION_PATTERN = re.compile(r"\[(\d+)]")

    def __init__(self, llm_service: LlmService) -> None:
        self.llm_service = llm_service

    def generate(self, query: str, retrieval_result: RetrievalResult) -> GenerationResult:
        """Ask the LLM to answer with [n] citation markers, then figure out
        which chunks it actually cited.
        """
        chunks = retrieval_result.results

        numbered_context = self._build_numbered_context(chunks)
        prompt = self.CITATION_PROMPT.format(context=numbered_context, query=query)

        try:
            answer = self.llm_service.generate(prompt)
        except RuntimeError:
            # Same graceful-degradation idea used elsewhere in this
            # project: an LLM call failure shouldn't crash the request,
            # just produce a clearly-labeled placeholder answer instead.
            answer = "Sorry, an error occurred while generating the answer."

        cited_chunks = self._extract_citations(answer, chunks)

        return GenerationResult(
            answer=answer,
            cited_chunks=cited_chunks,
            hallucination_score=-1.0,  # not computed until HallucinationDetector runs.
            retrieved_chunks=chunks,
        )

    @staticmethod
    def _build_numbered_context(chunks: list[SearchResult]) -> str:
        """Assemble the chunks into a numbered block, e.g. "[1] ...\\n\\n[2]
        ...", so the prompt's [number] citation markers have something to
        refer back to.
        """
        lines = [f"[{i}] {chunk.content}\n" for i, chunk in enumerate(chunks, start=1)]
        return "\n".join(lines)

    def _extract_citations(self, answer: str, chunks: list[SearchResult]) -> list[str]:
        """Find every [n] marker in the answer and map it back to the
        chunk_id of the nth chunk (1-based, matching how the prompt
        numbered them), deduping repeats and ignoring out-of-range numbers.
        """
        cited: list[str] = []
        for match in self.CITATION_PATTERN.finditer(answer):
            index = int(match.group(1)) - 1
            if 0 <= index < len(chunks):
                chunk_id = chunks[index].chunk_id
                if chunk_id not in cited:
                    cited.append(chunk_id)
        return cited
