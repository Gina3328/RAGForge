"""Pick the dense-cosine minimum for Agentic RAG's cross-collection merge
from data, not by hand. Read-only: no indexing, no writes to Milvus, no
LLM calls -- only query embeddings + vector searches. Needs Milvus (and the
embedding model) running, and the 3 eval documents already indexed (any
earlier evaluation run leaves them in the store).

What it does
------------
For every answerable eval question, it runs the SAME per-collection
retrieval Agentic RAG runs (HybridRetriever, so the same candidate pool the
strategy sees), and for every collection in the store keeps that
collection's top-MAX_RESULTS_PER_COLLECTION candidates by raw dense cosine
score (metadata["dense_score"]). Each candidate is labelled:

    target  -- it came from the collection of the document the question is about
    foreign -- it came from any other collection (leftover test data, or
               one of the other eval documents)

Collection identity is a coarse proxy for "relevant", but it is exactly the
distinction a cross-collection filter has to make, and it needs no extra
labelling. Then, for a grid of candidate thresholds, it reports how much of
the target side survives and how much of the foreign side is removed.

Selection rule (fixed in advance, so the threshold isn't tuned to look good)
---------------------------------------------------------------------------
    threshold = the LARGEST grid value at which at least 95% of the
                answerable questions still keep their target-collection
                top-1 candidate.

i.e. be as strict as possible without throwing away the chunk most likely
to hold the answer. The full grid is printed too, so the sensitivity of
that choice is visible (and can be reported).

Usage (from the project root):
    uv run python evaluation/calibrate_dense_threshold.py
Writes evaluation/reports/dense_threshold_calibration.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from ragforge.config import RagForgeConfig  # noqa: E402
from ragforge.embedding.embedding_service import EmbeddingService  # noqa: E402
from ragforge.retriever.hybrid_retriever import HybridRetriever  # noqa: E402
from ragforge.store.milvus_vector_store import MilvusVectorStore  # noqa: E402
from ragforge.strategy.agentic_rag_strategy import AgenticRAGStrategy  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = PROJECT_ROOT / "config" / "stage4.yml"
REPORTS = PROJECT_ROOT / "evaluation" / "reports"
CAP = AgenticRAGStrategy.MAX_RESULTS_PER_COLLECTION
GRID = [round(0.30 + 0.05 * i, 2) for i in range(12)]  # 0.30 .. 0.85
KEEP_TARGET_TOP1 = 0.95

DOCS = {
    # doc_key: (questions file, existing trace file that records the collection name)
    "ccby": ("eval_questions_ccby.json", "decision_traces_ccby_agentic_post_flush_fix.json"),
    "cdc": ("eval_questions_cdc_program_eval.json", "decision_traces_cdc_agentic_post_flush_fix.json"),
    "ragbench": ("eval_questions_ragbench.json", "decision_traces_ragbench_agentic_post_flush_fix.json"),
}


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else float("nan")


def main() -> None:
    config = RagForgeConfig.load(str(CONFIG_PATH))
    embedding_service = EmbeddingService(config.embedding)
    store = MilvusVectorStore(config.vector_store)
    retriever = HybridRetriever(embedding_service, store, config.retriever)
    prefix = config.vector_store.collection_prefix

    full_names = store.list_collections()
    shorts = [c[len(prefix):] if prefix and c.startswith(prefix) else c for c in full_names]
    print(f"Collections in the store ({len(shorts)}): {shorts}\n")

    target_top1: list[float] = []  # one per answerable question
    target_chunks: list[float] = []
    foreign_chunks: list[float] = []
    foreign_by_collection: dict[str, list[float]] = {}
    per_doc: dict[str, dict] = {}

    for doc_key, (q_file, trace_file) in DOCS.items():
        target = json.loads((REPORTS / trace_file).read_text(encoding="utf-8"))["collection"]
        if target not in shorts:
            print(f"!! {doc_key}: target collection {target!r} is not in the store -- index it first; skipping.")
            continue
        items = json.loads((PROJECT_ROOT / "evaluation" / q_file).read_text(encoding="utf-8"))
        doc_top1: list[float] = []
        for item in items:
            if not ("reference" in item and item["reference"] is not None):
                continue  # unanswerable question: no "right" chunk to protect
            q = item["question"]
            for short in shorts:
                cands = retriever.retrieve(q, short).results
                scores = sorted(
                    (c.metadata["dense_score"] for c in cands if "dense_score" in c.metadata),
                    reverse=True,
                )[:CAP]
                if short == target:
                    if scores:
                        target_top1.append(scores[0])
                        doc_top1.append(scores[0])
                    else:
                        target_top1.append(0.0)
                        doc_top1.append(0.0)
                    target_chunks.extend(scores)
                else:
                    foreign_chunks.extend(scores)
                    foreign_by_collection.setdefault(short, []).extend(scores)
        per_doc[doc_key] = {
            "target_collection": target,
            "n_questions": len(doc_top1),
            "target_top1_min": round(min(doc_top1), 4) if doc_top1 else None,
            "target_top1_median": round(statistics.median(doc_top1), 4) if doc_top1 else None,
        }
        print(f"{doc_key}: {len(doc_top1)} answerable questions scored")

    if not target_top1:
        print("Nothing scored -- is Milvus up and are the documents indexed?")
        return

    print("\nPer-document target top-1 cosine:", json.dumps(per_doc, indent=2))
    print("\nForeign collections, mean of their top-5 cosine scores (high = leftover data that looks 'relevant' to the embedder):")
    for name, sc in sorted(foreign_by_collection.items(), key=lambda kv: -statistics.mean(kv[1])):
        print(f"  {name:55s} mean={statistics.mean(sc):.3f}  max={max(sc):.3f}  n={len(sc)}")

    print("\nthreshold | target top-1 kept | target top-5 kept | foreign kept (lower is better)")
    grid_rows = []
    chosen = None
    for t in GRID:
        t1 = pct(sum(s >= t for s in target_top1), len(target_top1))
        tc = pct(sum(s >= t for s in target_chunks), len(target_chunks))
        fc = pct(sum(s >= t for s in foreign_chunks), len(foreign_chunks))
        grid_rows.append({"threshold": t, "target_top1_kept_pct": t1, "target_top5_kept_pct": tc, "foreign_kept_pct": fc})
        print(f"  {t:.2f}    |      {t1:5.1f}%       |      {tc:5.1f}%       |   {fc:5.1f}%")
        if t1 >= KEEP_TARGET_TOP1 * 100:
            chosen = t

    print(f"\nRule: largest threshold keeping >= {int(KEEP_TARGET_TOP1 * 100)}% of target top-1 candidates")
    print(f"=> recommended --agentic-dense-min-score {chosen}")

    out = REPORTS / "dense_threshold_calibration.json"
    out.write_text(
        json.dumps(
            {
                "collections": shorts,
                "per_doc": per_doc,
                "grid": grid_rows,
                "rule": f"largest threshold keeping >= {int(KEEP_TARGET_TOP1 * 100)}% of answerable questions' target-collection top-1 candidate",
                "recommended_threshold": chosen,
                "n_target_top1": len(target_top1),
                "n_foreign_chunks": len(foreign_chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
