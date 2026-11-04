"""One-off diagnostic (not part of the eval pipeline, safe to delete after use).

Answers a narrow question: for the 7 cdc questions where CRAG's confidence
collapsed to exactly 0.0 (dense retrieval returned zero results under the
current scoreThreshold=0.3), what were the *actual* raw cosine similarity
scores before the threshold filtered them out?

Does a threshold=0.0 dense search against the collection that's already
indexed in Milvus from the earlier evaluation runs -- no re-indexing, no
writes, purely read-only. Prints top-5 raw scores per question so we can
see how far below 0.3 they really sit before touching any config or code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from ragforge.config import RagForgeConfig  # noqa: E402
from ragforge.embedding.embedding_service import EmbeddingService  # noqa: E402
from ragforge.store.milvus_vector_store import MilvusVectorStore  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = PROJECT_ROOT / "config" / "stage4.yml"
COLLECTION = "cdc_program_evaluation_framework_2024"

# The 7 questions from decision_traces_cdc_crag.json whose trace showed
# confidence == 0.0 (pulled directly from that file, not retyped by hand).
QUESTIONS = [
    "What is the purpose of this report, and how does it relate to the 1999 CDC framework?",
    "What two key principles guided the 2024 update process?",
    'Why was the term "stakeholder" replaced with "interest holder" in this updated framework?',
    "What is the most substantial change from the 1999 framework to the 2024 update?",
    "What are the five Federal program evaluation standards described in this report?",
    "What are the three cross-cutting actions, and how are they depicted in relation to the framework's six steps?",
    "What are the five types of evaluation defined in this report?",
]

CURRENT_THRESHOLD = 0.3  # the value actually in effect (config/stage4.yml)


def main() -> None:
    config = RagForgeConfig.load(str(CONFIG_PATH))
    embedding_service = EmbeddingService(config.embedding)
    vector_store = MilvusVectorStore(config.vector_store)

    print(f"Collection: {COLLECTION}")
    print(f"Current scoreThreshold in effect: {CURRENT_THRESHOLD}")
    print("=" * 100)

    for q in QUESTIONS:
        query_vector = embedding_service.embed(q)

        # threshold=0.0 -- bypass filtering entirely, see every raw score
        # Milvus actually returned for this query.
        raw_results = vector_store.search_dense(COLLECTION, query_vector, top_k=5, threshold=0.0)

        print(f"\nQ: {q}")
        if not raw_results:
            print("  -> Milvus returned ZERO hits even with threshold=0.0 "
                  "(collection/query embedding issue, not a threshold issue).")
            continue

        scores = [r.score for r in raw_results]
        top1 = scores[0]
        avg_top5 = sum(scores) / len(scores)
        survive_03 = sum(1 for s in scores if s >= CURRENT_THRESHOLD)

        print(f"  raw top-{len(scores)} scores: {[round(s, 4) for s in scores]}")
        print(f"  top1={top1:.4f}  avg_top5={avg_top5:.4f}  "
              f"survive threshold={CURRENT_THRESHOLD}: {survive_03}/{len(scores)}")


if __name__ == "__main__":
    main()
