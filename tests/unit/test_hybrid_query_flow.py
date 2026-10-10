"""D5-D6：验证公开混合检索与 CoreReranker 的离线契约。"""

from copy import deepcopy
import socket

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
)
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import (
    CoreReranker,
    RerankConfig,
    RerankError,
)
from src.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from src.core.types import ProcessedQuery, RetrievalResult
from src.core.trace import TraceContext
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """本文件只运行离线 adapter，不允许建立网络连接。"""

    def reject_network(*args, **kwargs):
        raise AssertionError("D5-D6 测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def _settings(tmp_path, collection="docs", *, rerank_enabled=False, rerank_provider="none"):
    """直接构造非敏感设置，避免读取仓库配置或任何凭据。"""
    return Settings(
        llm=LLMSettings("offline", "offline", 0.0, 32),
        embedding=EmbeddingSettings("offline", "offline", 2),
        vector_store=VectorStoreSettings("chroma", str(tmp_path / "chroma"), collection),
        retrieval=RetrievalSettings(4, 5, 2, 17),
        rerank=RerankSettings(rerank_enabled, rerank_provider, "offline", 2),
        evaluation=EvaluationSettings(False, "offline", []),
        observability=ObservabilitySettings("INFO", False, str(tmp_path / "trace.jsonl"), False),
        ingestion=IngestionSettings(1000, 0, "recursive", 8, {"use_llm": False}, {"use_llm": False}),
    )


class FakeEmbedding(BaseEmbedding):
    """记录真实 DenseRetriever seam 的输入并返回固定二维向量。"""

    def __init__(self):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.append((list(texts), trace, kwargs.copy()))
        return [[1.0, 0.0] for _ in texts]

    def get_dimension(self):
        return 2


class FakeVectorStore(BaseVectorStore):
    """给真实 Retriever 提供记录可检查的向量存储 seam。"""

    def __init__(self, collection_name, dense_records=(), sparse_records=None):
        self.collection_name = collection_name
        self.dense_records = list(dense_records)
        self.sparse_records = sparse_records or {}
        self.query_calls = []
        self.get_calls = []

    def upsert(self, records, trace=None, **kwargs):
        self.validate_records(records)

    def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
        self.query_calls.append((vector, top_k, filters, trace))
        return list(self.dense_records)

    def get_by_ids(self, ids, trace=None, **kwargs):
        self.get_calls.append((list(ids), trace))
        return [self.sparse_records.get(identifier, {}) for identifier in ids]


class FakeBM25:
    """记录集合和关键词，返回预设倒排命中。"""

    def __init__(self, rows):
        self.rows = list(rows)
        self.loaded = []
        self.queried = []

    def load(self, collection, trace=None):
        self.loaded.append((collection, trace))
        return True

    def query(self, query_terms, top_k, trace=None):
        self.queried.append((list(query_terms), top_k, trace))
        return self.rows[:top_k]


class FakeRoute:
    """直接控制路线结果或异常，检查 HybridSearch 的降级边界。"""

    def __init__(self, collection="docs", results=None, error=None):
        self.vector_store = type("StoreBinding", (), {"collection_name": collection})()
        self.results = [] if results is None else results
        self.error = error
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return list(self.results)


class FixedProcessor:
    """隔离路线错误测试中的关键词输入。"""

    def __init__(self, keywords=("quartz",), filters=None):
        self.keywords = list(keywords)
        self.filters = {} if filters is None else dict(filters)

    def process(self, query):
        return ProcessedQuery(query, list(self.keywords), dict(self.filters))


def _result(identifier, score, text=None, **metadata):
    return RetrievalResult(identifier, score, text or f"正文 {identifier}", metadata)


def test_hybrid_search_preserves_query_and_applies_merged_filters_before_rrf(tmp_path):
    """真实 Dense/Sparse adapter 按绑定集合检索，后置条件命中过滤后才融合。"""
    settings = _settings(tmp_path)
    trace = TraceContext(trace_type="query")
    query = "quartz collection:docs source:guide tag:red kind:query-value"
    explicit = {"source_path": "guide/handbook", "tags": ["blue"], "kind": "explicit"}
    original_filters = deepcopy(explicit)
    dense_rows = [
        {"id": "d1", "score": 0.91, "text": "dense 命中", "metadata": {
            "source_path": "docs/guide/handbook.md", "tags": "red, blue", "kind": "explicit"}},
        {"id": "d2", "score": 0.88, "text": "标签不符", "metadata": {
            "source_path": "docs/guide/handbook.md", "tags": "red", "kind": "explicit"}},
        {"id": "d3", "score": 0.82, "text": "metadata 不符", "metadata": {
            "source_path": "docs/guide/handbook.md", "tags": ["blue"], "kind": "query-value"}},
    ]
    sparse_rows = {
        "s1": {"id": "s1", "text": "sparse 命中", "metadata": {
            "source_path": "archive/guide/handbook-copy.md", "tags": ["blue", "green"], "kind": "explicit"}},
        "d1": {"id": "d1", "text": "sparse 中的 dense 命中", "metadata": {
            "source_path": "docs/guide/handbook.md", "tags": ["blue"], "kind": "explicit"}},
        "d2": {"id": "d2", "text": "标签不符", "metadata": {
            "source_path": "docs/guide/handbook.md", "tags": "red", "kind": "explicit"}},
    }
    store = FakeVectorStore("docs", dense_rows, sparse_rows)
    embedding = FakeEmbedding()
    bm25 = FakeBM25([
        {"chunk_id": "s1", "score": 3.0},
        {"chunk_id": "d1", "score": 2.0},
        {"chunk_id": "d2", "score": 1.0},
    ])
    hybrid = HybridSearch(
        settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(settings, embedding, store),
        sparse_retriever=SparseRetriever(settings, bm25, store, default_collection="docs"),
        collection="docs",
    )

    details = hybrid.search(query, top_k=2, filters=explicit, trace=trace, return_details=True)

    assert [item.chunk_id for item in details.dense_results] == ["d1", "d2", "d3"]
    assert [item.chunk_id for item in details.sparse_results] == ["s1", "d1", "d2"]
    assert [item.chunk_id for item in details.results] == ["d1", "s1"]
    assert details.results[0].score == pytest.approx(1 / 18 + 1 / 19)
    assert details.results[1].score == pytest.approx(1 / 18)
    assert details.used_fallback is False
    assert details.processed_query.original_query == query
    assert embedding.calls == [([query], trace, {"task_type": "RETRIEVAL_QUERY"})]
    assert store.query_calls == [([1.0, 0.0], 4, {"kind": "explicit"}, trace)]
    assert bm25.loaded == [("docs", trace)]
    assert "quartz" in bm25.queried[0][0]
    assert bm25.queried[0][1:] == (5, trace)
    assert explicit == original_filters
    assert hybrid.config.dense_top_k == 4
    assert hybrid.config.sparse_top_k == 5
    assert hybrid.config.fusion_top_k == 2
    assert hybrid.fusion.k == 17


def test_hybrid_search_rejects_collection_rebinding_and_silent_filter_omission(tmp_path):
    """集合由注入实例绑定；关闭后置过滤时，带条件的查询必须明确失败。"""
    settings = _settings(tmp_path)
    store = FakeVectorStore("docs")
    dense = DenseRetriever(settings, FakeEmbedding(), store)

    with pytest.raises(ValueError, match="same collection"):
        HybridSearch(settings, dense_retriever=dense, collection="another")

    hybrid = HybridSearch(settings, query_processor=QueryProcessor(), dense_retriever=dense, collection="docs")
    with pytest.raises(ValueError, match="bound retrievers"):
        hybrid.search("quartz collection:another")

    no_post_filter = HybridSearch(
        settings,
        dense_retriever=dense,
        config=HybridSearchConfig(metadata_filter_post=False),
        collection="docs",
    )
    with pytest.raises(ValueError, match="Metadata filtering"):
        no_post_filter.search("quartz", filters={"kind": "guide"})


def test_hybrid_search_keeps_single_successful_route_score_when_other_route_is_empty(tmp_path):
    """正常空命中不是失败，单路结果保留本路原始分数。"""
    expected = _result("dense-only", 0.73)
    dense = FakeRoute(results=[expected])
    sparse = FakeRoute(results=[])
    hybrid = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(), dense_retriever=dense,
        sparse_retriever=sparse, collection="docs",
    )

    results = hybrid.search("quartz")

    assert results == [expected]
    assert results[0].score == 0.73


def test_hybrid_search_marks_failed_route_even_when_other_route_succeeds_empty(tmp_path):
    """一路失败、另一路正常空结果时返回空并标记降级，错误不带原文。"""
    dense = FakeRoute(error=RuntimeError("private dense detail"))
    sparse = FakeRoute(results=[])
    hybrid = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(), dense_retriever=dense,
        sparse_retriever=sparse, collection="docs",
    )

    details = hybrid.search("quartz", return_details=True)

    assert details.results == []
    assert details.used_fallback is True
    assert details.dense_error == "dense retrieval failed: RuntimeError"
    assert details.sparse_error is None
    assert "private dense detail" not in repr(details)


def test_hybrid_search_uses_remaining_route_and_errors_only_when_all_active_routes_fail(tmp_path):
    """失败路线由另一条命中接管；所有已执行路线都失败才抛出不含异常正文的错误。"""
    expected = _result("sparse-only", 1.25)
    hybrid = HybridSearch(
        _settings(tmp_path),
        query_processor=FixedProcessor(),
        dense_retriever=FakeRoute(error=TimeoutError("private dense timeout")),
        sparse_retriever=FakeRoute(results=[expected]),
        collection="docs",
    )

    details = hybrid.search("quartz", return_details=True)

    assert details.used_fallback is True
    assert details.results == [expected]
    assert details.results[0].score == 1.25

    all_failed = HybridSearch(
        _settings(tmp_path),
        query_processor=FixedProcessor(),
        dense_retriever=FakeRoute(error=RuntimeError("private dense")),
        sparse_retriever=FakeRoute(error=ValueError("private sparse")),
        collection="docs",
    )
    with pytest.raises(RuntimeError) as error:
        all_failed.search("quartz")
    assert "private dense" not in str(error.value)
    assert "private sparse" not in str(error.value)
    assert "RuntimeError" in str(error.value)
    assert "ValueError" in str(error.value)


def test_hybrid_search_skips_sparse_for_empty_keywords_and_rejects_no_routes(tmp_path):
    """零关键词时不调用 Sparse；缺少任何可运行路线时才报告配置错误。"""
    sparse = FakeRoute()
    configured_sparse = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(keywords=()),
        sparse_retriever=sparse, collection="docs",
    )

    details = configured_sparse.search("quartz", return_details=True)

    assert details.results == []
    assert details.used_fallback is False
    assert sparse.calls == []

    no_routes = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(keywords=()),
        config=HybridSearchConfig(enable_dense=False), collection="docs",
    )
    with pytest.raises(RuntimeError, match="No retrieval route"):
        no_routes.search("quartz")


class FixedLLM(BaseLLM):
    """提供离线 JSON 评分，验证已存在的 LLMReranker adapter。"""

    def __init__(self, response):
        self.response = response
        self.messages = []

    def chat(self, messages, trace=None, **kwargs):
        self.validate_messages(messages)
        self.messages.extend(messages)
        return ChatResponse(self.response, "offline-test")


class FixedReranker(BaseReranker):
    """用可控后端响应验证 Core 的结果校验。"""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def rerank(self, query, candidates, trace=None, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def _candidates():
    return [
        _result("a", 0.9, "原文甲", source_path="a.md", nested={"kept": True}),
        _result("b", 0.6, "原文乙", source_path="b.md"),
        _result("c", 0.3, "原文丙", source_path="c.md"),
    ]


def test_core_reranker_uses_existing_llm_adapter_and_preserves_candidates(tmp_path):
    """真实 LLMReranker 协议接收全部候选，Core 排序、改分并在最后裁切。"""
    settings = _settings(tmp_path, rerank_enabled=True, rerank_provider="llm")
    prompt = tmp_path / "rerank-prompt.txt"
    prompt.write_text("请按相关性给每段打分。", encoding="utf-8")
    fake_llm = FixedLLM(
        '[{"passage_id":"a","score":1},'
        '{"passage_id":"b","score":3},{"passage_id":"c","score":2}]'
    )
    adapter = RerankerFactory.create(settings, llm=fake_llm, prompt_path=prompt)
    core = CoreReranker(settings, adapter, RerankConfig(enabled=True, top_k=2))
    source = _candidates()
    before = deepcopy(source)

    result = core.rerank("哪个方案更相关？", source)

    assert result.reranker_type == "llm"
    assert result.used_fallback is False
    assert [item.chunk_id for item in result.results] == ["b", "c"]
    assert [item.score for item in result.results] == [3.0, 2.0]
    assert result.results[0].text == "原文乙"
    assert result.results[0].metadata == {
        "source_path": "b.md", "original_score": 0.6, "rerank_score": 3.0, "reranked": True,
    }
    assert result.results[1].metadata["source_path"] == "c.md"
    assert "Passage ID: a" in fake_llm.messages[0].content
    assert "Passage ID: b" in fake_llm.messages[0].content
    assert "Passage ID: c" in fake_llm.messages[0].content
    assert source == before


def test_core_reranker_is_lazy_when_disabled_or_candidate_count_is_below_two(tmp_path, monkeypatch):
    """关闭、空候选和单候选都不触发重排 Factory。"""
    def forbidden_create(cls, settings, **kwargs):
        pytest.fail("无需重排时不应创建评分后端")

    monkeypatch.setattr(RerankerFactory, "create", classmethod(forbidden_create))
    disabled = _settings(tmp_path, rerank_enabled=False, rerank_provider="llm")
    disabled_core = CoreReranker(disabled)
    source = _candidates()
    assert disabled_core.rerank("问题", source).results == source[:2]

    enabled = _settings(tmp_path, rerank_enabled=True, rerank_provider="llm")
    enabled_core = CoreReranker(enabled)
    assert enabled_core.rerank("问题", []).results == []
    single = source[:1]
    assert enabled_core.rerank("问题", single).results == single


@pytest.mark.parametrize(
    "response",
    [
        [{"id": "unknown", "rerank_score": 2.0}, {"id": "b", "rerank_score": 1.0}],
        [{"id": "a", "rerank_score": 2.0}, {"id": "a", "rerank_score": 1.0}],
        [{"id": "a", "rerank_score": 2.0}],
        [{"id": "a", "rerank_score": float("nan")}, {"id": "b", "rerank_score": 1.0}],
    ],
    ids=["未知ID", "重复ID", "漏项", "非有限分数"],
)
def test_core_reranker_rejects_invalid_backend_results_and_falls_back_in_input_order(tmp_path, response):
    """未知、重复、遗漏候选或 NaN 分数都回退到原顺序和原检索分。"""
    settings = _settings(tmp_path, rerank_enabled=True, rerank_provider="custom")
    source = _candidates()[:2]
    before = deepcopy(source)
    core = CoreReranker(settings, FixedReranker(response), RerankConfig(enabled=True, top_k=2))

    result = core.rerank("问题", source)

    assert result.used_fallback is True
    assert [item.chunk_id for item in result.results] == ["a", "b"]
    assert [item.score for item in result.results] == [0.9, 0.6]
    assert all(item.metadata["reranked"] is False for item in result.results)
    assert all(item.metadata["rerank_fallback"] is True for item in result.results)
    assert source == before


def test_core_reranker_handles_backend_timeout_and_can_disable_fallback(tmp_path):
    """已抛出的 TimeoutError 可降级；显式关闭回退时抛出有限类型错误。"""
    settings = _settings(tmp_path, rerank_enabled=True, rerank_provider="custom")
    source = _candidates()[:2]
    backend = FixedReranker(error=TimeoutError("private backend detail"))
    fallback = CoreReranker(settings, backend, RerankConfig(enabled=True, top_k=2))

    result = fallback.rerank("问题", source)

    assert result.used_fallback is True
    assert result.fallback_reason == "Reranking failed: TimeoutError"
    assert [item.score for item in result.results] == [0.9, 0.6]
    assert "private backend detail" not in result.fallback_reason

    no_fallback = CoreReranker(settings, backend, RerankConfig(enabled=True, top_k=2, fallback_on_error=False))
    with pytest.raises(RerankError, match="TimeoutError") as error:
        no_fallback.rerank("问题", source)
    assert "private backend detail" not in str(error.value)


def test_core_reranker_caches_backend_initialization_failure(tmp_path, monkeypatch):
    """后端初始化失败后保留有限错误类型，不重复尝试创建。"""
    settings = _settings(tmp_path, rerank_enabled=True, rerank_provider="custom")
    calls = []

    def fail_create(cls, selected_settings, **kwargs):
        calls.append(selected_settings)
        raise ValueError("private initialization detail")

    monkeypatch.setattr(RerankerFactory, "create", classmethod(fail_create))
    core = CoreReranker(settings)

    first = core.rerank("问题", _candidates()[:2])
    second = core.rerank("问题", _candidates()[:2])

    assert len(calls) == 1
    assert first.used_fallback is True
    assert second.used_fallback is True
    assert "private initialization detail" not in first.fallback_reason
    assert "private initialization detail" not in second.fallback_reason
    assert "ValueError" in first.fallback_reason
