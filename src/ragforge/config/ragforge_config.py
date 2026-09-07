"""Global configuration for the RagForge engine.

Configuration is loaded from a YAML file and grows progressively across stages:

- Stage 1: llm, embedding, vector_store, chunker, retriever, evaluation
- Stage 2 (implemented here): query_engine, reranker, generator
- Stage 3 (later): strategy (Self-RAG / CRAG / Adaptive)
- Stage 4 (later): api, cache

Naming note: every dataclass field here uses idiomatic Python snake_case,
even though the underlying YAML files (stage1.yml/stage2.yml) keep their
camelCase keys (inherited from the Java reference project this is ported
from -- there's no reason to churn the YAML itself just to match Python
convention). Because the two naming styles differ, `load()` below can't
just unpack a YAML section straight into its dataclass with
`SomeConfig(**raw.get("section"))` -- each field is instead looked up by
its YAML (camelCase) key and passed in under its Python (snake_case) name,
falling back to the dataclass's own default when the YAML file doesn't set
it. It's more verbose than a straight unpack, but keeps every Python-side
name idiomatic regardless of what the YAML calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LlmConfig:
    """Chat LLM configuration (defaults to a local Ollama model).

    Note: `base_url` and `api_key` are not yet used by Stage 1's
    SimpleGenerator (which calls Anthropic directly via the
    ANTHROPIC_API_KEY env var). They're reserved for Stage 2, when the
    generator is expected to support switching between multiple LLM
    providers.
    """

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5"
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    """Text embedding model configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dimension: int = 768
    timeout: int = 30


@dataclass
class VectorStoreConfig:
    """Milvus vector database connection settings."""

    host: str = "localhost"
    port: int = 19530
    collection_prefix: str = "ragforge_"


@dataclass
class ChunkerConfig:
    """Text chunking strategy configuration.

    max_size/overlap are used by every strategy. The rest are each used
    by exactly one non-default strategy, and ignored otherwise:
    child_size/child_overlap by "parent_child", similarity_threshold/
    min_chunk_size/max_chunk_size by "semantic". "structure_aware" needs
    no extra fields of its own -- it just reuses max_size/overlap.
    """

    strategy: str = "fixed_size"
    max_size: int = 500
    overlap: int = 50
    # "parent_child" strategy: size/overlap for the smaller "child" chunks
    # nested inside each larger "parent" chunk (which itself still uses
    # max_size/overlap).
    child_size: int = 128
    child_overlap: int = 30
    # "semantic" strategy: cosine-similarity cutoff between consecutive
    # sentences below which a new chunk starts, plus the chunk size
    # bounds it's kept within regardless of where similarity would cut.
    similarity_threshold: float = 0.5
    min_chunk_size: int = 100
    max_chunk_size: int = 500


@dataclass
class RetrieverConfig:
    """Retriever configuration.

    dense_weight/sparse_weight/dynamic_weight/rrf_k are used by
    HybridRetriever (Stage 2) only -- DenseRetriever only reads
    top_k/score_threshold and ignores the rest.
    """

    strategy: str = "dense"
    top_k: int = 5
    # Candidates surviving to the final answer after reranking -- smaller
    # than top_k on purpose: top_k is how many candidates come OUT of
    # retrieval (cast a wide net), final_top_k is how many survive
    # reranking to actually go into the generation prompt (keep only the
    # best few). Used by AdvancedRAGStrategy's reranking step.
    final_top_k: int = 5
    score_threshold: float = 0.5
    # Base weights HybridRetriever's RRF fusion assigns to the dense and
    # sparse (keyword-match) paths. When dynamic_weight is True, these
    # serve as the "center" that gets scaled up/down based on query
    # length, rather than being used as-is.
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    # Whether HybridRetriever should scale dense_weight/sparse_weight
    # based on query length (short queries lean sparse, long queries lean
    # dense) instead of using the configured weights unchanged.
    dynamic_weight: bool = True
    # RRF's rank-discount constant. 60 is the value recommended by the
    # original RRF paper; rarely needs tuning in practice.
    rrf_k: int = 60


@dataclass
class QueryEngineConfig:
    """Query-understanding configuration for QueryEngine (Stage 2).

    `enabled` is the master switch (AdvancedRAGStrategy skips building a
    QueryEngine at all when this is False); the other four flags turn
    individual query-processing steps on or off once QueryEngine is built.
    """

    enabled: bool = True
    enable_rewrite: bool = True
    enable_hyde: bool = True
    enable_multi_query: bool = True
    enable_decomposition: bool = True


@dataclass
class RerankerConfig:
    """Reranker configuration (Stage 2).

    `strategy` selects which Reranker implementation to build (e.g. "llm"
    for CrossEncoderReranker/LLM-based scoring); `base_url`/`model`/
    `timeout` configure that implementation's backing service.
    """

    enabled: bool = True
    strategy: str = "llm"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5"
    timeout: int = 60


@dataclass
class GeneratorConfig:
    """Generation-layer configuration (Stage 2): which optional
    generation-quality features are turned on, and their tuning knobs.
    """

    enable_citation: bool = True
    enable_hallucination_detection: bool = True
    context_compression_threshold: float = 0.3
    max_context_tokens: int = 3000


@dataclass
class EvaluationConfig:
    """Evaluation configuration: dataset path and report output directory."""

    dataset_path: str = "evaluation/eval_set.json"
    report_path: str = "evaluation/reports"


@dataclass
class RagForgeConfig:
    """Top-level RagForge configuration, composed of the sub-configs above."""

    llm: LlmConfig = field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    query_engine: QueryEngineConfig = field(default_factory=QueryEngineConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "RagForgeConfig":
        """Load configuration from a YAML file.

        Any section missing from the YAML file falls back to the defaults
        defined on the corresponding dataclass; any key missing within a
        present section falls back the same way, field by field.
        """
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        llm_raw = raw.get("llm") or {}
        llm = LlmConfig(
            base_url=llm_raw.get("baseUrl", LlmConfig.base_url),
            api_key=llm_raw.get("apiKey", LlmConfig.api_key),
            model=llm_raw.get("model", LlmConfig.model),
            timeout=llm_raw.get("timeout", LlmConfig.timeout),
        )

        embedding_raw = raw.get("embedding") or {}
        embedding = EmbeddingConfig(
            base_url=embedding_raw.get("baseUrl", EmbeddingConfig.base_url),
            model=embedding_raw.get("model", EmbeddingConfig.model),
            dimension=embedding_raw.get("dimension", EmbeddingConfig.dimension),
            timeout=embedding_raw.get("timeout", EmbeddingConfig.timeout),
        )

        vector_store_raw = raw.get("vectorStore") or {}
        vector_store = VectorStoreConfig(
            host=vector_store_raw.get("host", VectorStoreConfig.host),
            port=vector_store_raw.get("port", VectorStoreConfig.port),
            collection_prefix=vector_store_raw.get("collectionPrefix", VectorStoreConfig.collection_prefix),
        )

        chunker_raw = raw.get("chunker") or {}
        chunker = ChunkerConfig(
            strategy=chunker_raw.get("strategy", ChunkerConfig.strategy),
            max_size=chunker_raw.get("maxSize", ChunkerConfig.max_size),
            overlap=chunker_raw.get("overlap", ChunkerConfig.overlap),
            child_size=chunker_raw.get("childSize", ChunkerConfig.child_size),
            child_overlap=chunker_raw.get("childOverlap", ChunkerConfig.child_overlap),
            similarity_threshold=chunker_raw.get("similarityThreshold", ChunkerConfig.similarity_threshold),
            min_chunk_size=chunker_raw.get("minChunkSize", ChunkerConfig.min_chunk_size),
            max_chunk_size=chunker_raw.get("maxChunkSize", ChunkerConfig.max_chunk_size),
        )

        retriever_raw = raw.get("retriever") or {}
        retriever = RetrieverConfig(
            strategy=retriever_raw.get("strategy", RetrieverConfig.strategy),
            top_k=retriever_raw.get("topK", RetrieverConfig.top_k),
            final_top_k=retriever_raw.get("finalTopK", RetrieverConfig.final_top_k),
            score_threshold=retriever_raw.get("scoreThreshold", RetrieverConfig.score_threshold),
            dense_weight=retriever_raw.get("denseWeight", RetrieverConfig.dense_weight),
            sparse_weight=retriever_raw.get("sparseWeight", RetrieverConfig.sparse_weight),
            dynamic_weight=retriever_raw.get("dynamicWeight", RetrieverConfig.dynamic_weight),
            rrf_k=retriever_raw.get("rrfK", RetrieverConfig.rrf_k),
        )

        query_engine_raw = raw.get("queryEngine") or {}
        query_engine = QueryEngineConfig(
            enabled=query_engine_raw.get("enabled", QueryEngineConfig.enabled),
            enable_rewrite=query_engine_raw.get("enableRewrite", QueryEngineConfig.enable_rewrite),
            enable_hyde=query_engine_raw.get("enableHyDE", QueryEngineConfig.enable_hyde),
            enable_multi_query=query_engine_raw.get("enableMultiQuery", QueryEngineConfig.enable_multi_query),
            enable_decomposition=query_engine_raw.get(
                "enableDecomposition", QueryEngineConfig.enable_decomposition
            ),
        )

        reranker_raw = raw.get("reranker") or {}
        reranker = RerankerConfig(
            enabled=reranker_raw.get("enabled", RerankerConfig.enabled),
            strategy=reranker_raw.get("strategy", RerankerConfig.strategy),
            base_url=reranker_raw.get("baseUrl", RerankerConfig.base_url),
            model=reranker_raw.get("model", RerankerConfig.model),
            timeout=reranker_raw.get("timeout", RerankerConfig.timeout),
        )

        generator_raw = raw.get("generator") or {}
        generator = GeneratorConfig(
            enable_citation=generator_raw.get("enableCitation", GeneratorConfig.enable_citation),
            enable_hallucination_detection=generator_raw.get(
                "enableHallucinationDetection", GeneratorConfig.enable_hallucination_detection
            ),
            context_compression_threshold=generator_raw.get(
                "contextCompressionThreshold", GeneratorConfig.context_compression_threshold
            ),
            max_context_tokens=generator_raw.get("maxContextTokens", GeneratorConfig.max_context_tokens),
        )

        evaluation_raw = raw.get("evaluation") or {}
        evaluation = EvaluationConfig(
            dataset_path=evaluation_raw.get("datasetPath", EvaluationConfig.dataset_path),
            report_path=evaluation_raw.get("reportPath", EvaluationConfig.report_path),
        )

        return cls(
            llm=llm,
            embedding=embedding,
            vector_store=vector_store,
            chunker=chunker,
            retriever=retriever,
            query_engine=query_engine,
            reranker=reranker,
            generator=generator,
            evaluation=evaluation,
        )
