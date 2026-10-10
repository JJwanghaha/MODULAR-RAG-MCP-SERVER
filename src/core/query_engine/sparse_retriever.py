"""D3：BM25 命中 → 按 ID 取回正文 → 统一 RetrievalResult。"""

from typing import Any

from src.core.settings import Settings, resolve_path
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """每次查询重新加载磁盘索引；一份实例使用调用方注入的集合存储。"""

    def __init__(self, settings: Settings | None = None, bm25_indexer: BM25Indexer | None = None,
                 vector_store: BaseVectorStore | None = None, default_top_k: int = 10,
                 default_collection: str = "default"):
        self.bm25_indexer, self.vector_store = bm25_indexer, vector_store
        self.default_top_k = settings.retrieval.sparse_top_k if settings is not None else default_top_k
        self.default_collection = default_collection

    def retrieve(self, keywords: list[str], top_k: int | None = None, collection: str | None = None,
                 trace: Any = None) -> list[RetrievalResult]:
        """无索引返回空结果；损坏或读取失败明确抛错，供 D5 决定降级。"""
        if not isinstance(keywords, list) or not keywords or any(not isinstance(term, str) for term in keywords):
            raise ValueError("keywords must be a nonempty list of strings")
        limit = self.default_top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.bm25_indexer is None or self.vector_store is None:
            raise RuntimeError("SparseRetriever requires bm25_indexer and vector_store")
        selected = self.default_collection if collection is None else collection
        bound = getattr(self.vector_store, "collection_name", selected)
        if bound != selected:
            raise ValueError("BM25 collection must match the injected vector store collection")
        try:
            if not self.bm25_indexer.load(collection=selected, trace=trace):
                return []
            matches = self.bm25_indexer.query(query_terms=keywords, top_k=limit, trace=trace)
            if not matches:
                return []
            records = self.vector_store.get_by_ids([match["chunk_id"] for match in matches], trace=trace)
            # 按 ID 合并，而不是假设 adapter 会返回同样数量/顺序的记录。
            by_id = {str(record["id"]): record for record in records if record}
            return [RetrievalResult(match["chunk_id"], float(match["score"]), by_id[match["chunk_id"]].get("text", ""),
                                    dict(by_id[match["chunk_id"]].get("metadata") or {}))
                    for match in matches if match["chunk_id"] in by_id]
        except Exception as exc:
            raise RuntimeError(f"Sparse retrieval failed: {type(exc).__name__}") from None


def create_sparse_retriever(settings: Settings, bm25_indexer: BM25Indexer | None = None,
                            vector_store: BaseVectorStore | None = None, index_dir: str = "data/db/bm25",
                            *, collection: str | None = None) -> SparseRetriever:
    """默认目录对齐 C14：BM25 根目录/集合/集合_bm25.json。"""
    selected = collection if collection is not None else getattr(vector_store, "collection_name", settings.vector_store.collection_name)
    if not isinstance(selected, str) or not selected.strip() or selected in {".", ".."} or any(c in selected for c in "/\\"):
        raise ValueError("collection must be a single path component")
    if bm25_indexer is None:
        bm25_indexer = BM25Indexer(str(resolve_path(index_dir) / selected))
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings, collection_name=selected)
    return SparseRetriever(settings, bm25_indexer, vector_store, default_collection=selected)
