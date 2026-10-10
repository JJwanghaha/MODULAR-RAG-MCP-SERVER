"""CLI 与 MCP 共用的依赖构造；检索和重排算法仍由 D5/D6 实现。"""

from dataclasses import dataclass
from pathlib import Path

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import HybridSearch, create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import CoreReranker, create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore


@dataclass
class QueryComponents:
    """保存可复用入口及资源所有权；不关闭调用方注入的存储或模型。"""

    hybrid_search: HybridSearch
    reranker: CoreReranker
    vector_store: BaseVectorStore
    owns_store: bool = True

    def close(self) -> None:
        if self.owns_store:
            client = getattr(self.vector_store, "client", None)
            if client is not None:
                client.close()
            self.owns_store = False


def create_query_components(settings, collection: str, index_dir: str | Path = "data/db/bm25",
                            *, embedding_client=None, vector_store=None) -> QueryComponents:
    """打开已有集合后才创建模型；失败释放本函数创建的存储。"""
    owns_store = vector_store is None
    store = vector_store if vector_store is not None else VectorStoreFactory.create(
        settings, collection_name=collection, create_if_missing=False)
    try:
        embedding = embedding_client if embedding_client is not None else EmbeddingFactory.create(settings)
        dense = create_dense_retriever(settings, embedding_client=embedding, vector_store=store)
        sparse = create_sparse_retriever(settings, vector_store=store, index_dir=str(index_dir), collection=collection)
        hybrid = create_hybrid_search(settings, QueryProcessor(), dense, sparse, collection=collection)
        return QueryComponents(hybrid, create_core_reranker(settings), store, owns_store)
    except Exception:
        if owns_store:
            client = getattr(store, "client", None)
            if client is not None:
                client.close()
        raise
