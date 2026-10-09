"""C12：用上游最终 ID 规则把 Chunk 和向量写入存储。"""

from hashlib import sha256
from typing import Any

from src.core.settings import Settings
from src.core.types import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """默认使用配置的 Factory，也允许注入已有存储。"""

    def __init__(self, settings: Settings, collection_name: str | None = None,
                 *, vector_store: BaseVectorStore | None = None):
        self.settings = settings
        overrides = {"collection_name": collection_name} if collection_name else {}
        self.vector_store = vector_store if vector_store is not None else VectorStoreFactory.create(settings, **overrides)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], trace: Any = None) -> list[str]:
        """只更新同 ID 记录；正文变化后的旧 ID 不会在此自动删除。"""
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Nonempty chunks and vectors must have matching counts")
        ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        records = [{"id": chunk_id, "vector": vector,
                    "metadata": {**chunk.metadata, "text": chunk.text, "chunk_id": chunk_id}}
                   for chunk, vector, chunk_id in zip(chunks, vectors, ids)]
        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as exc:
            raise RuntimeError(f"Vector upsert failed: {type(exc).__name__}") from exc
        return ids

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """来源哈希＋块序号＋增强后正文哈希；不是 C4 的临时 ID。"""
        if "source_path" not in chunk.metadata or "chunk_index" not in chunk.metadata:
            raise ValueError("Chunk metadata requires source_path and chunk_index")
        source = sha256(chunk.metadata["source_path"].encode("utf-8")).hexdigest()[:8]
        content = sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        return f"{source}_{chunk.metadata['chunk_index']:04d}_{content}"

    def upsert_batch(self, batches: list[tuple[list[Chunk], list[list[float]]]], trace: Any = None) -> list[str]:
        """先检查每批数量，再展平；避免一批多、一批少相互抵消。"""
        chunks, vectors = [], []
        for batch_chunks, batch_vectors in batches:
            if len(batch_chunks) != len(batch_vectors):
                raise ValueError("Each batch must have matching chunk and vector counts")
            chunks.extend(batch_chunks)
            vectors.extend(batch_vectors)
        return self.upsert(chunks, vectors, trace=trace)
