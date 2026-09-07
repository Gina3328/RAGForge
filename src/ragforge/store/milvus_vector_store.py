"""Milvus-backed implementation of VectorStore."""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from ragforge.config import VectorStoreConfig

from .chunk_embedding import ChunkEmbedding
from .search_result import SearchResult
from .vector_store import VectorStore


class MilvusVectorStore(VectorStore):
    """Vector store implementation backed by Milvus."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.client = MilvusClient(uri=f"http://{config.host}:{config.port}")
        self.collection_prefix = config.collection_prefix

    def create_collection(self, name: str, dimension: int) -> None:
        """Create a collection with the schema chunks are stored in.

        If a collection with the same name already exists, it is dropped
        and recreated so we never insert into a collection with a
        mismatched schema.
        """
        full_name = self.collection_prefix + name

        # Idempotency: start from a clean slate if the collection exists.
        if self.client.has_collection(collection_name=full_name):
            self.client.drop_collection(collection_name=full_name)

        # Define the schema: one row per chunk.
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
        )
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        # AUTOINDEX + cosine similarity: a sensible default for dense vectors.
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=full_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, collection: str, data: list[ChunkEmbedding]) -> None:
        """Insert a batch of chunk embeddings into a collection.

        This is a plain insert, not a true upsert: re-inserting the same
        source file will duplicate rows unless the old ones are removed
        first (see delete_by_source).
        """
        if not data:
            return

        full_name = self.collection_prefix + collection

        # Each row is a plain dict matching the collection's schema fields.
        rows = [
            {
                "chunk_id": item.chunk_id,
                "content": item.content,
                "embedding": item.embedding,
                "metadata": item.metadata,
            }
            for item in data
        ]

        self.client.insert(collection_name=full_name, data=rows)

    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        threshold: float,
    ) -> list[SearchResult]:
        """Find the top_k chunks most similar to the query vector.

        Hits scoring below `threshold` are dropped from the results, so
        fewer than top_k (or zero) results may be returned.
        """
        full_name = self.collection_prefix + collection

        resp = self.client.search(
            collection_name=full_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "content", "metadata"],
        )
        # Only one query vector was submitted, so we only need its hits.
        hits = resp[0]

        results: list[SearchResult] = []
        for hit in hits:
            score = hit["distance"]
            if score < threshold:
                continue

            entity = hit["entity"]
            results.append(
                SearchResult(
                    chunk_id=entity.get("chunk_id", ""),
                    content=entity.get("content", ""),
                    score=score,
                    metadata=entity.get("metadata") or {},
                )
            )

        return results

    def get_by_ids(self, collection: str, chunk_ids: list[str]) -> list[SearchResult]:
        """Fetch chunks directly by chunk_id (primary-key lookup, not a
        similarity search). Returns fewer entries than requested if some
        ids don't exist -- never raises for a missing id."""
        if not chunk_ids:
            return []

        full_name = self.collection_prefix + collection
        rows = self.client.get(
            collection_name=full_name,
            ids=chunk_ids,
            output_fields=["chunk_id", "content", "metadata"],
        )

        return [
            SearchResult(
                chunk_id=row.get("chunk_id", ""),
                content=row.get("content", ""),
                score=0.0,
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]

    def delete_by_source(self, collection: str, source: str) -> None:
        """Delete every chunk that came from the given source file."""
        full_name = self.collection_prefix + collection
        filter_expr = f'metadata["source"] == "{source}"'
        self.client.delete(collection_name=full_name, filter=filter_expr)

    def drop_collection(self, name: str) -> None:
        """Delete an entire collection."""
        full_name = self.collection_prefix + name
        self.client.drop_collection(collection_name=full_name)

    def list_collections(self) -> list[str]:
        """List all collections managed by this store."""
        names = self.client.list_collections()
        return [n for n in names if n.startswith(self.collection_prefix)]
