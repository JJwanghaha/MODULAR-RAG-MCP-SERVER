"""本地 Chroma adapter，沿用上游 ID、向量、metadata.text 的存储映射。"""

from typing import Any

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """持久化已计算好的向量；不自动调用或下载 Embedding 模型。"""

    def __init__(self, settings: Any, **kwargs: Any) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed; install .[vector-stores]"
            ) from exc

        self.persist_directory = resolve_path(
            kwargs.get("persist_directory", settings.vector_store.persist_directory)
        )
        self.collection_name = kwargs.get(
            "collection_name", settings.vector_store.collection_name
        )
        create_if_missing = kwargs.get("create_if_missing", True)
        if not create_if_missing and not (self.persist_directory / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Chroma database does not exist")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                _env_file=None, anonymized_telemetry=False, allow_reset=False
            ),
        )
        try:
            if create_if_missing:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None,
                )
            else:
                self.collection = self.client.get_collection(name=self.collection_name, embedding_function=None)
        except Exception as exc:
            self.client.close()
            if not create_if_missing and isinstance(exc, chromadb.errors.NotFoundError):
                raise FileNotFoundError("Chroma collection does not exist") from None
            raise

    def upsert(self, records: list[dict[str, Any]], trace=None, **kwargs: Any) -> None:
        """按上游约定，同时存文本和 metadata 中的文本副本。"""
        self.validate_records(records)
        ids, vectors, metadatas, documents = [], [], [], []
        for record in records:
            metadata = record.get("metadata", {})
            ids.append(str(record["id"]))
            vectors.append(record["vector"])
            documents.append(str(metadata.get("text", record["id"])))
            metadatas.append(
                self._sanitize_metadata(metadata) or {"_placeholder": "true"}
            )
        try:
            self.collection.upsert(
                ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(f"Chroma upsert failed: {type(exc).__name__}") from exc

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace=None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """按余弦距离检索，再转换为上游的 score，不是答案可信度。"""
        self.validate_query_vector(vector, top_k)
        where = filters or None
        if filters and len(filters) > 1:
            where = {"$and": [{key: value} for key, value in filters.items()]}
        try:
            result = self.collection.query(
                query_embeddings=[vector], n_results=top_k, where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"Chroma query failed: {type(exc).__name__}") from exc
        return [
            {
                "id": identifier,
                "score": max(0.0, 1.0 - distance / 2.0),
                "text": document or "",
                "metadata": metadata or {},
            }
            for identifier, distance, document, metadata in zip(
                result["ids"][0],
                result["distances"][0],
                result["documents"][0],
                result["metadatas"][0],
            )
        ]

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """沿用上游转换；复杂结构转字符串的限制留待后续改造。"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                continue
            elif isinstance(value, (list, tuple)):
                sanitized[key] = ",".join(str(item) for item in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    def get_by_ids(self, ids: list[str], trace=None, **kwargs: Any) -> list[dict[str, Any]]:
        """按请求顺序取回原文；缺失 ID 保留空位置，不压缩列表。"""
        if not ids:
            raise ValueError("IDs list cannot be empty")
        requested = [str(identifier) for identifier in ids]
        try:
            result = self.collection.get(ids=list(dict.fromkeys(requested)), include=["documents", "metadatas"])
        except Exception as exc:
            raise RuntimeError(f"Chroma get_by_ids failed: {type(exc).__name__}") from None
        records = {identifier: {"id": identifier, "text": document or "", "metadata": metadata or {}}
                   for identifier, document, metadata in zip(result["ids"], result["documents"], result["metadatas"])}
        return [records.get(identifier, {}) for identifier in requested]
