"""Agentic RAG strategy (Stage 4 -- Agent-driven dynamic retrieval).

Embeds retrieval in an agent loop: instead of deciding once up front how
to retrieve (Adaptive RAG's approach), the LLM re-evaluates the
accumulated context after every retrieval round and decides what to do
next -- generate the final answer now, retrieve again with a new query,
or rewrite the original question and retrieve with that. The loop keeps
going until the LLM says the context is sufficient, or a maximum number
of rounds is reached (whichever comes first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from ragforge.generator.generation_result import GenerationResult
from ragforge.generator.generator import Generator
from ragforge.llm import LlmService
from ragforge.query.query_engine import QueryEngine
from ragforge.retriever.retrieval_result import RetrievalResult
from ragforge.retriever.retriever import Retriever
from ragforge.store.search_result import SearchResult
from ragforge.store.vector_store import VectorStore

from .rag_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class Action(Enum):
    """The Agent loop's next move: generate the final answer now,
    retrieve with a new query, or rewrite the original question and
    retrieve with that.
    """

    GENERATE = "GENERATE"
    RETRIEVE = "RETRIEVE"
    REWRITE_AND_RETRIEVE = "REWRITE_AND_RETRIEVE"


@dataclass
class AgentDecision:
    """The Agent loop's decision for one round: what to do next, and
    (for RETRIEVE) the query to retrieve with.
    """

    action: Action
    new_query: str | None = None


class AgenticRAGStrategy(RAGStrategy):
    """Runs a multi-round agent loop: retrieve, let the LLM decide
    whether the accumulated context is enough, and either generate the
    final answer or retrieve again -- up to max_iterations rounds.
    """

    # How many results each individual collection is allowed to
    # contribute to a single cross-collection retrieval round. Kept
    # deliberately small -- see _retrieve_all_collections's docstring
    # for why a large per-collection contribution defeats score-based
    # sorting once several collections are merged together. 3 turned
    # out too aggressive in testing: a chunk that names something
    # explicitly (e.g. a short enumeration) can score lower than
    # several chunks that merely discuss the same topic in passing,
    # so capping too tightly risks dropping the one chunk that actually
    # answers the question. 5 trades back a little of the noise
    # reduction for better recall.
    MAX_RESULTS_PER_COLLECTION = 5

    def __init__(
        self,
        dense_retriever: Retriever,
        hybrid_retriever: Retriever,
        generator: Generator,
        llm: LlmService,
        query_engine: QueryEngine | None,
        vector_store: VectorStore | None,
        collection_prefix: str,
        max_iterations: int,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.llm = llm
        self.query_engine = query_engine
        self.vector_store = vector_store
        self.collection_prefix = collection_prefix
        self.max_iterations = max_iterations

    def get_name(self) -> str:
        return "Agentic RAG (Stage 4)"

    def execute(self, query: str, collection: str) -> GenerationResult:
        """Run the Agent loop: retrieve, decide, and either generate the
        final answer or retrieve again, up to max_iterations rounds.
        """
        logger.info("Agentic RAG executing query: %s", query)

        # Accumulated context: chunks collected across every round so far.
        all_chunks: list[SearchResult] = []

        # Queries already used, to avoid retrieving with the same query twice.
        used_queries: set[str] = set()
        used_queries.add(query.lower().strip())

        # Initial retrieval: search every collection, merge and dedupe.
        initial_chunks = self._retrieve_all_collections(query, collection)
        recent_chunks = self._add_new_chunks(all_chunks, initial_chunks)
        logger.info("Initial retrieval complete: %d chunk(s) collected", len(all_chunks))

        # Agent loop.
        for i in range(1, self.max_iterations + 1):
            # Let the LLM judge whether the current context is enough,
            # and if not, what to do next. recent_chunks (whatever the
            # previous round just added, if anything) is always shown
            # first, so the Agent can actually see what it just asked
            # for -- see _decide_action's docstring for why that matters.
            # used_queries is passed through too, so the Agent doesn't
            # waste a round re-proposing something it already tried.
            decision = self._decide_action(query, all_chunks, i, recent_chunks, used_queries)
            logger.info(
                "Round %d Agent decision: %s%s",
                i,
                decision.action,
                f" | new_query: {decision.new_query}" if decision.new_query else "",
            )

            match decision.action:
                case Action.GENERATE:
                    logger.info("Context is sufficient, generating the final answer")
                    return self._generate_answer(query, all_chunks)

                case Action.RETRIEVE:
                    retrieve_query = decision.new_query
                    if not retrieve_query or not retrieve_query.strip():
                        logger.warning("Agent returned an empty query, falling back to rewrite")
                        retrieve_query = self._try_rewrite(query)

                    if retrieve_query and retrieve_query.lower().strip() not in used_queries:
                        used_queries.add(retrieve_query.lower().strip())
                        new_chunks = self._retrieve_all_collections(retrieve_query, collection)
                        recent_chunks = self._add_new_chunks(all_chunks, new_chunks)
                        logger.info(
                            "Retrieved %d new chunk(s), %d total",
                            len(new_chunks),
                            len(all_chunks),
                        )
                    else:
                        logger.debug("Query already used or empty, skipping this round's retrieval")
                        recent_chunks = []

                case Action.REWRITE_AND_RETRIEVE:
                    rewritten = self._try_rewrite(query)
                    if rewritten and rewritten.lower().strip() not in used_queries:
                        used_queries.add(rewritten.lower().strip())
                        new_chunks = self._retrieve_all_collections(rewritten, collection)
                        recent_chunks = self._add_new_chunks(all_chunks, new_chunks)
                        logger.info("Rewritten retrieval found %d new chunk(s)", len(new_chunks))
                    else:
                        recent_chunks = []

        # Reached the iteration cap without the Agent choosing to
        # generate -- fall back to answering with whatever was collected.
        logger.info("Reached max iterations (%d), generating the final answer", self.max_iterations)
        return self._generate_answer(query, all_chunks)

    def _generate_answer(self, query: str, all_chunks: list[SearchResult]) -> GenerationResult:
        """Generate the final answer from the accumulated context."""
        result = RetrievalResult(query=query, results=all_chunks, total_retrieved=len(all_chunks))
        return self.generator.generate(query, result)

    def _decide_action(
        self,
        original_query: str,
        chunks: list[SearchResult],
        iteration: int,
        recent_chunks: list[SearchResult] | None = None,
        used_queries: set[str] | None = None,
    ) -> AgentDecision:
        """Ask the LLM whether the accumulated context is sufficient, and
        if not, what to do next. Falls back to GENERATE if the LLM call
        or the response parsing fails, so the loop can never get stuck.

        recent_chunks -- the chunks newly added by the immediately
        preceding retrieval round, if any -- are always shown first in
        the context summary, ahead of chunks pulled in from earlier
        rounds. Without this, whatever the Agent just asked for could
        easily land past the 10-chunk summary cutoff even after the
        score-based sort in _add_new_chunks, since RRF-fused hybrid
        retrieval scores are rank-based and compressed into a narrow
        range -- ties are common, and a tie doesn't guarantee the chunk
        that answers what the Agent is actively looking for right now
        ends up near the front. Surfacing it unconditionally, regardless
        of how it happened to score, is what actually lets the Agent
        notice "I already have this" instead of re-requesting the same
        thing round after round.

        used_queries -- every query tried so far, including the
        original question -- is listed explicitly in the prompt so the
        LLM doesn't burn a round proposing something it (or execute's
        own dedup check) already tried. Without this the Agent has no
        way to know what it's already asked for, since that state lives
        in execute()'s local variable, not anywhere the LLM can see it.
        """
        recent_chunks = recent_chunks or []
        recent_ids = {id(c) for c in recent_chunks}
        older_chunks = [c for c in chunks if id(c) not in recent_ids]
        summary_chunks = (recent_chunks + older_chunks)[:10]

        context_summary = "\n---\n".join(
            c.content[:200] + "..." if len(c.content) > 200 else c.content
            for c in summary_chunks
        )

        tried_queries = "\n".join(f"- {q}" for q in used_queries) if used_queries else "(none yet)"

        prompt = f"""You are an intelligent retrieval assistant. Based on the original question and the information collected so far, decide the next action.

Original question: {original_query}
Current iteration: {iteration}

Queries already tried (do not repeat any of these; if you choose RETRIEVE, new_query must target information genuinely missing from what's below, not restate a query already on this list):
{tried_queries}

Summary of information collected so far:
{context_summary}

Choose one of the following actions:
- GENERATE: the current information is already sufficient to answer the original question -- including the case where some part of the question genuinely isn't answerable from what's been retrieved after multiple attempts, and no new query is likely to find it
- RETRIEVE: more retrieval is needed; provide a new search query or question in new_query that is meaningfully different from every query already tried above
- REWRITE_AND_RETRIEVE: the original question needs to be rewritten before retrieving again

Reply using exactly the following JSON format (no other content):
{{"action": "GENERATE"}} or
{{"action": "RETRIEVE", "new_query": "the new retrieval question"}}
{{"action": "REWRITE_AND_RETRIEVE"}}
"""

        try:
            response = self.llm.generate(prompt).strip()
            return self._parse_decision(response)
        except Exception as e:
            logger.warning("Agent decision failed: %s, defaulting to GENERATE", e)
            return AgentDecision(Action.GENERATE, None)

    def _parse_decision(self, response: str) -> AgentDecision:
        """Parse the LLM's JSON reply into an AgentDecision. Falls back
        to GENERATE if the action is missing or unrecognized.
        """
        # Extract the JSON portion, in case the model wrapped it in
        # extra text (a code fence, a leading "Here's the answer:", etc).
        json_str = response
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            json_str = response[start : end + 1]

        action_str = self._extract_json_value(json_str, "action") or ""
        new_query_str = self._extract_json_value(json_str, "new_query")

        try:
            action = Action(action_str.upper().strip())
        except ValueError:
            action = Action.GENERATE

        return AgentDecision(action, new_query_str)

    def _extract_json_value(self, json_str: str, key: str) -> str | None:
        """Naive fallback extractor: find "key": "value" inside json_str and
        return value, without requiring the surrounding text to be valid
        JSON. Only handles quoted string values with no embedded/escaped
        quotes; returns None if the key is absent or the value isn't a
        plain quoted string.
        """
        pattern = f'"{key}"'
        idx = json_str.find(pattern)
        if idx < 0:
            return None

        colon_idx = json_str.find(":", idx + len(pattern))
        if colon_idx < 0:
            return None

        value_start = json_str.find('"', colon_idx + 1)
        if value_start < 0:
            return None

        value_end = json_str.find('"', value_start + 1)
        if value_end < 0:
            return None

        return json_str[value_start + 1 : value_end]

    def _try_rewrite(self, query: str) -> str | None:
        """Try to get a better query via QueryEngine: prefer a rewritten
        query, fall back to a HyDE hypothetical answer, or None if
        neither is available (no query engine configured, or the call
        fails).
        """
        if self.query_engine is None:
            return None
        try:
            processed = self.query_engine.process_query(query)
            if processed.rewritten:
                return processed.rewritten
            if processed.hyde_answer:
                return processed.hyde_answer
        except Exception:
            logger.warning("Query rewrite failed", exc_info=True)
        return None

    def _add_new_chunks(
        self, all_chunks: list[SearchResult], new_chunks: list[SearchResult]
    ) -> list[SearchResult]:
        """Merge chunks from new_chunks into all_chunks, deduplicating by
        exact (stripped) content match so the same chunk collected in an
        earlier round isn't added twice. Returns the subset of new_chunks
        that was actually added (post-dedup), so the caller can track
        what's new this round -- see _decide_action's recent_chunks
        parameter.

        Re-sorts the combined list by score, descending, after merging --
        not just appending new_chunks at the end -- so a highly-relevant
        chunk found by a later, more targeted round doesn't get stuck
        behind the (possibly less relevant) results from an earlier,
        broader round.
        """
        existing_contents = {c.content.strip() for c in all_chunks}
        added: list[SearchResult] = []
        for chunk in new_chunks:
            content = chunk.content.strip()
            if content not in existing_contents:
                all_chunks.append(chunk)
                existing_contents.add(content)
                added.append(chunk)
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return added

    def _retrieve_all_collections(self, query: str, collection: str) -> list[SearchResult]:
        """Cross-collection retrieval: search every collection the
        vector store knows about and merge the results, deduplicated by
        content. This is what sets Agentic RAG apart from the other
        (single-collection) strategies -- every round pulls in
        information from every source, not just the collection the
        caller selected. Falls back to single-collection retrieval if
        no vector store was configured, or if listing collections fails.

        Each collection contributes at most MAX_RESULTS_PER_COLLECTION of
        its own top-scoring results, not its full result set. This
        matters because the score attached to each result comes from
        hybrid_retriever's RRF fusion, which is rank-based and
        compressed into a narrow range (see CRAGStrategy's module
        docstring for the same characteristic causing a different
        problem there) -- every collection's own #1 result ends up
        clustered near the same ceiling score regardless of how
        genuinely relevant it is to this query. Merging each
        collection's full top_k (e.g. 20) results together would mean
        dozens of near-tied, mostly-irrelevant candidates drown out the
        one collection that actually matters; capping each collection's
        contribution keeps the merged pool small enough that sorting it
        by score afterward is still meaningful.

        The merged (and capped) results are sorted by score, descending,
        before being returned, so _decide_action's summary of the first
        10 accumulated chunks is at least biased toward the
        highest-scoring candidates rather than whatever order
        list_collections() happened to return collections in.
        """
        if self.vector_store is None:
            return self.hybrid_retriever.retrieve(query, collection).results

        try:
            cols = self.vector_store.list_collections()
        except Exception:
            logger.warning("Failed to list collections, falling back to single-collection search", exc_info=True)
            return self.hybrid_retriever.retrieve(query, collection).results

        all_results: list[SearchResult] = []
        seen: set[str] = set()
        for col in cols:
            short_name = (
                col[len(self.collection_prefix) :]
                if self.collection_prefix and col.startswith(self.collection_prefix)
                else col
            )
            try:
                collection_results = sorted(
                    self.hybrid_retriever.retrieve(query, short_name).results,
                    key=lambda r: r.score,
                    reverse=True,
                )[: self.MAX_RESULTS_PER_COLLECTION]
                for s in collection_results:
                    key = s.content.strip()
                    if key not in seen:
                        seen.add(key)
                        all_results.append(s)
            except Exception:
                logger.debug("Collection %s failed to retrieve, skipping", col, exc_info=True)

        all_results.sort(key=lambda s: s.score, reverse=True)
        return all_results
