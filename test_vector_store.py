"""Standalone smoke test for MilvusVectorStore.

Run from the project root with:
    uv run python test_vector_store.py

This bypasses chunker/embedding entirely and uses small fake vectors,
so it only tells you whether store/ itself works against your local
Milvus instance.
"""

from ragforge.config import RagForgeConfig
from ragforge.store.chunk_embedding import ChunkEmbedding
from ragforge.store.milvus_vector_store import MilvusVectorStore

TEST_COLLECTION = "smoke_test"
TEST_DIMENSION = 4  # small on purpose, has nothing to do with nomic-embed-text's 768


def main() -> None:
    config = RagForgeConfig.load("config/stage1.yml")
    store = MilvusVectorStore(config.vector_store)

    print("1. create_collection")
    store.create_collection(TEST_COLLECTION, TEST_DIMENSION)
    print("   collections:", store.list_collections())

    print("2. upsert")
    fake_data = [
        ChunkEmbedding(
            chunk_id="chunk_1",
            content="RagForge is a personal RAG learning project.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"title": "doc1", "source": "fake_source.md", "sections": "intro"},
        ),
        ChunkEmbedding(
            chunk_id="chunk_2",
            content="Milvus stores vectors and supports similarity search.",
            embedding=[0.9, 0.8, 0.7, 0.6],
            metadata={"title": "doc1", "source": "fake_source.md", "sections": "intro"},
        ),
    ]
    store.upsert(TEST_COLLECTION, fake_data)

    print("3. search_dense (query close to chunk_1)")
    results = store.search_dense(
        TEST_COLLECTION,
        query_vector=[0.1, 0.2, 0.3, 0.35],
        top_k=5,
        threshold=0.0,
    )
    for r in results:
        print("   ->", r)

    print("4. delete_by_source")
    store.delete_by_source(TEST_COLLECTION, "fake_source.md")
    results_after_delete = store.search_dense(
        TEST_COLLECTION,
        query_vector=[0.1, 0.2, 0.3, 0.35],
        top_k=5,
        threshold=0.0,
    )
    print("   results after delete:", results_after_delete)

    print("5. drop_collection (cleanup)")
    store.drop_collection(TEST_COLLECTION)
    print("   collections:", store.list_collections())


if __name__ == "__main__":
    main()
