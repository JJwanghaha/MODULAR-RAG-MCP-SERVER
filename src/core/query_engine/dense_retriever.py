"""D2：问题编码与向量召回；不在此融合、重排或生成答案。"""

from typing import Any

from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """沿用上游注入已有模型/存储的 interface；构造本身不读 Key 或创建数据库。"""

    def __init__(self, settings: Settings | None = None, embedding_client: BaseEmbedding | None = None,
                 vector_store: BaseVectorStore | None = None, default_top_k: int = 10):
        self.embedding_client, self.vector_store = embedding_client, vector_store
        self.default_top_k = settings.retrieval.dense_top_k if settings is not None else default_top_k

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None,
                 trace: Any = None) -> list[RetrievalResult]:
        """显式使用 RETRIEVAL_QUERY，使 Gemini 问题编码与文档编码区分。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonblank string")
        limit = self.default_top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.embedding_client is None or self.vector_store is None:
            raise RuntimeError("DenseRetriever requires embedding_client and vector_store")
        try:
            vectors = self.embedding_client.embed([query], trace=trace, task_type="RETRIEVAL_QUERY")
            self.embedding_client.validate_vectors(vectors, 1)
        except Exception as exc:
            raise RuntimeError(f"Query embedding failed: {type(exc).__name__}") from None
        try:
            raw = self.vector_store.query(vector=vectors[0], top_k=limit, filters=filters, trace=trace)
            return [RetrievalResult(str(item["id"]), float(item["score"]), item.get("text", ""),
                                    dict(item.get("metadata") or {})) for item in raw]
        except Exception as exc:
            raise RuntimeError(f"Dense retrieval failed: {type(exc).__name__}") from None


def create_dense_retriever(settings: Settings, embedding_client: BaseEmbedding | None = None,
                           vector_store: BaseVectorStore | None = None) -> DenseRetriever:
    """显式构造运行资源；未提供的依赖才由现有 Factory 创建。"""
    if embedding_client is None:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        embedding_client = EmbeddingFactory.create(settings)
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    return DenseRetriever(settings, embedding_client, vector_store)
