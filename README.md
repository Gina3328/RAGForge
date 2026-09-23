# RAGForge

A retrieval-augmented generation (RAG) system built from scratch across four
progressively more sophisticated stages: a naive retrieve-then-generate
baseline, a fully optimized retrieval pipeline, three self-decision RAG
architectures (Self-RAG, CRAG, Adaptive-RAG), and an autonomous Agentic RAG
controller.

## Architecture

| Stage | Entry point | What it adds |
|---|---|---|
| 1 | `src/ragforge/cli.py` | Naive retrieve-then-generate baseline. |
| 2 | `src/ragforge/cli_stage2.py` | Document parsing, 5 chunking strategies, hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion (RRF), a multi-stage query understanding pipeline (rewriting, HyDE, multi-query generation, decomposition), Cross-Encoder reranking, and citation / hallucination detection. |
| 3 | `src/ragforge/cli_stage3.py` | Self-RAG (retrieval gating + relevance evaluation + retry), CRAG (confidence-based corrective retrieval), and Adaptive-RAG (query-complexity routing) as pluggable strategies, selected via config. |
| 4 | `src/ragforge/cli_stage4.py` | Agentic RAG: an LLM controller that autonomously chooses GENERATE / RETRIEVE / REWRITE_AND_RETRIEVE across multiple rounds and multiple document collections, with bounded retry and fallback logic. |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (to run Milvus, the vector database)
- [Ollama](https://ollama.com/) (to serve the embedding model)
- An Anthropic API key (used for generation, query understanding, and evaluation)

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/Gina3328/RAGForge.git
cd RAGForge
uv sync
```

**2. Start Milvus**

```bash
docker compose -f docker/docker-compose.yml up -d
```

**3. Install Ollama and pull the embedding model**

```bash
# see https://ollama.com/download for platform-specific install instructions
ollama pull nomic-embed-text
```

**4. Configure your API key**

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Running

Each stage has its own interactive CLI. Run these from the repo root --
each config path is resolved relative to the current directory:

```bash
uv run python src/ragforge/cli.py          # Stage 1: naive RAG
uv run python src/ragforge/cli_stage2.py   # Stage 2: full optimized pipeline
uv run python src/ragforge/cli_stage3.py   # Stage 3: Self-RAG / CRAG / Adaptive-RAG
uv run python src/ragforge/cli_stage4.py   # Stage 4: Agentic RAG
```

Inside any CLI session:

```
index <path/to/document.pdf>
query <your question>
help
quit
```

Each stage reads its own config file (`config/stage1.yml` ... `config/stage4.yml`).
For Stage 3 and Stage 4, which strategy is active is set via the `strategy.name`
field in the corresponding config file (e.g. `self_rag`, `crag`, `adaptive`, or
`agentic`).

## Evaluation

`evaluation/` contains two separate harnesses:

- `collect_predictions.py` + `run_ragas_eval.py` (isolated dependencies in
  `evaluation/.ragas_venv`) -- a [RAGAS](https://github.com/explodinggradients/ragas)-based
  harness that measures answer quality (faithfulness, answer relevancy,
  context precision/recall) across pipeline variants (baseline, chunking,
  hybrid retrieval, reranking, query understanding, generation).
- `collect_decision_traces.py` + `analyze_decision_traces.py` (isolated
  dependencies in `evaluation/.analysis_venv`) -- non-invasively records each
  Stage 3/4 strategy's internal decision state (confidence scores,
  complexity routing, round counts) alongside its answers, for evaluating
  routing/decision correctness rather than just final-answer quality.

See `evaluation/README.md` for detailed usage. Evaluation source documents
and generated reports are not included in this repository -- point the
scripts at your own PDFs, or build an `eval_questions_*.json` file following
the existing examples in `evaluation/`.

## Notes

- Reranking currently runs a local Cross-Encoder model
  (`sentence-transformers`, downloaded automatically on first use) --
  no external reranking service is required, independent of the
  `reranker.model` value in the config files.
- Configuration YAML files use camelCase keys; the Python config loader
  (`src/ragforge/config/ragforge_config.py`) maps them to idiomatic
  snake_case fields internally.
