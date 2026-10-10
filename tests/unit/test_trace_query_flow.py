"""F3：通过查询公开入口验证 Trace 的候选、计时和降级记录。"""

from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from src.core.query_engine import hybrid_search as hybrid_search_module
from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.query_engine.reranker import CoreReranker, RerankConfig
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
from src.core.trace import TraceContext
from src.core.trace import trace_context as trace_context_module
from src.core.types import ProcessedQuery, RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker

pytestmark = pytest.mark.unit


def _settings(tmp_path, *, rerank_enabled=True, rerank_provider="custom"):
    return Settings(
        llm=LLMSettings("offline", "offline", 0.0, 32),
        embedding=EmbeddingSettings("offline", "offline", 2),
        vector_store=VectorStoreSettings("chroma", str(tmp_path / "chroma"), "docs"),
        retrieval=RetrievalSettings(4, 5, 3, 10),
        rerank=RerankSettings(rerank_enabled, rerank_provider, "offline", 3),
        evaluation=EvaluationSettings(False, "offline", []),
        observability=ObservabilitySettings("INFO", False, str(tmp_path / "traces.jsonl"), False),
        ingestion=IngestionSettings(1000, 0, "recursive", 8, {"use_llm": False}, {"use_llm": False}),
    )


def _result(chunk_id, score, text, source):
    return RetrievalResult(chunk_id, score, text, {"source_path": source, "title": f"title-{chunk_id}"})


class FixedProcessor:
    def __init__(self, keywords=("quartz", "archive")):
        self.keywords = list(keywords)

    def process(self, query):
        return ProcessedQuery(query, list(self.keywords), {"collection": "docs"})


class FixedRoute:
    def __init__(self, results=(), error=None, barrier=None):
        self.vector_store = SimpleNamespace(collection_name="docs")
        self.results = list(results)
        self.error = error
        self.barrier = barrier
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.error is not None:
            raise self.error
        return list(self.results)


class FixedReranker(BaseReranker):
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, candidates, trace=None, **kwargs):
        self.calls.append((query, candidates, trace))
        return [
            {"id": candidate["id"], "rerank_score": self.scores[candidate["id"]]}
            for candidate in candidates
        ]


def _counting_clock():
    lock = Lock()
    state = {"value": 0.0}

    def monotonic():
        with lock:
            state["value"] += 0.001
            return state["value"]

    return monotonic


def test_hybrid_and_core_reranker_record_full_query_candidates_order_scores_and_parallel_timings(
    tmp_path, monkeypatch
):
    """真实 TraceContext 汇总全链路快照；并行路线耗时不相加成总耗时。"""
    query_clock = _counting_clock()
    trace_clock = _counting_clock()
    monkeypatch.setattr(hybrid_search_module, "time", SimpleNamespace(monotonic=query_clock))
    monkeypatch.setattr(trace_context_module, "time", SimpleNamespace(monotonic=trace_clock))
    barrier = Barrier(2)
    dense = FixedRoute(
        [_result("dense-a", 0.91, "Dense candidate full text.", "dense/a.md"),
         _result("shared", 0.82, "Shared candidate full text.", "shared.md")],
        barrier=barrier,
    )
    sparse = FixedRoute(
        [_result("shared", 4.0, "Shared candidate full text.", "shared.md"),
         _result("sparse-c", 2.5, "Sparse candidate full text.", "sparse/c.md")],
        barrier=barrier,
    )
    query = "Which archived quartz result should be preferred? collection:docs"
    trace = TraceContext(trace_type="query")
    hybrid = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(), dense_retriever=dense,
        sparse_retriever=sparse, collection="docs",
    )
    details = hybrid.search(query, top_k=3, trace=trace, return_details=True)
    backend = FixedReranker({"shared": 3.2, "dense-a": 2.1, "sparse-c": 1.1})
    reranked = CoreReranker(
        _settings(tmp_path), backend, RerankConfig(enabled=True, top_k=3),
    ).rerank(query, details.results, top_k=3, trace=trace)

    assert details.processed_query.original_query == query
    assert [item.chunk_id for item in details.results] == ["shared", "dense-a", "sparse-c"]
    assert [item.chunk_id for item in reranked.results] == ["shared", "dense-a", "sparse-c"]
    assert [item.score for item in reranked.results] == [3.2, 2.1, 1.1]
    assert backend.calls[0][0] == query
    assert [item["id"] for item in backend.calls[0][1]] == ["shared", "dense-a", "sparse-c"]
    assert backend.calls[0][2] is trace
    assert len(dense.calls) == len(sparse.calls) == 1

    processing = trace.get_stage_data("query_processing")
    assert processing == {
        "method": "query_processor", "status": "success", "original_query": query,
        "keywords": ["quartz", "archive"], "filters": {"collection": "docs"},
    }
    expected_stages = {
        "query_processing": ("query_processor", "success"),
        "dense_retrieval": ("dense", "success"),
        "sparse_retrieval": ("bm25", "success"),
        "fusion": ("rrf", "success"),
        "rerank": ("fixedreranker", "success"),
    }
    for name, (method, status) in expected_stages.items():
        entry = next(stage for stage in trace.stages if stage["stage"] == name)
        assert entry["data"]["method"] == method
        assert entry["data"]["status"] == status
        assert entry["elapsed_ms"] >= 0

    fusion = trace.get_stage_data("fusion")
    assert [(row["chunk_id"], row["score"], row["text"]) for row in fusion["chunks"]] == [
        ("shared", pytest.approx(1 / 12 + 1 / 11), "Shared candidate full text."),
        ("dense-a", pytest.approx(1 / 11), "Dense candidate full text."),
        ("sparse-c", pytest.approx(1 / 12), "Sparse candidate full text."),
    ]
    rerank = trace.get_stage_data("rerank")
    assert [row["chunk_id"] for row in rerank["input_chunks"]] == ["shared", "dense-a", "sparse-c"]
    assert [row["score"] for row in rerank["chunks"]] == [3.2, 2.1, 1.1]
    route_elapsed = (
        next(stage["elapsed_ms"] for stage in trace.stages if stage["stage"] == "dense_retrieval")
        + next(stage["elapsed_ms"] for stage in trace.stages if stage["stage"] == "sparse_retrieval")
    )
    assert trace.elapsed_ms() < route_elapsed
    assert trace.finished_at is None
    assert not (tmp_path / "traces.jsonl").exists()


def test_hybrid_trace_distinguishes_single_route_fallback_from_total_failure(tmp_path):
    """单路故障记录降级；所有实际路线故障时仍记录两路后再报错。"""
    fallback_trace = TraceContext(trace_type="query")
    fallback = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(),
        dense_retriever=FixedRoute(error=TimeoutError("private timeout")),
        sparse_retriever=FixedRoute([_result("sparse-only", 1.25, "Sparse result.", "s.md")]),
        collection="docs",
    )

    details = fallback.search("quartz archive", trace=fallback_trace, return_details=True)

    assert details.used_fallback is True
    assert [item.chunk_id for item in details.results] == ["sparse-only"]
    assert fallback_trace.get_stage_data("dense_retrieval")["status"] == "error"
    assert fallback_trace.get_stage_data("dense_retrieval")["error"] == "dense retrieval failed: TimeoutError"
    assert fallback_trace.get_stage_data("sparse_retrieval")["status"] == "success"
    assert fallback_trace.get_stage_data("fusion")["status"] == "fallback"

    failed_trace = TraceContext(trace_type="query")
    failed = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(),
        dense_retriever=FixedRoute(error=RuntimeError("private dense")),
        sparse_retriever=FixedRoute(error=ValueError("private sparse")),
        collection="docs",
    )
    with pytest.raises(RuntimeError, match="All active retrieval routes failed"):
        failed.search("quartz archive", trace=failed_trace)

    assert failed_trace.get_stage_data("dense_retrieval")["status"] == "error"
    assert failed_trace.get_stage_data("sparse_retrieval")["status"] == "error"
    assert "private dense" not in repr(failed_trace.to_dict())
    assert "private sparse" not in repr(failed_trace.to_dict())
    assert not any(stage["stage"] == "fusion" for stage in failed_trace.stages)


def test_hybrid_trace_marks_disabled_and_no_keyword_routes_and_rerank_skips(tmp_path):
    """零关键词/关闭路线与未执行重排均明确标记为 skipped。"""
    trace = TraceContext(trace_type="query")
    dense = FixedRoute([
        _result("one", 0.7, "One result.", "one.md"),
        _result("two", 0.6, "Second result.", "two.md"),
    ])
    sparse = FixedRoute([_result("ignored", 0.4, "Ignored.", "ignored.md")])
    hybrid = HybridSearch(
        _settings(tmp_path, rerank_enabled=False, rerank_provider="none"),
        query_processor=FixedProcessor(keywords=()), dense_retriever=dense,
        sparse_retriever=sparse, collection="docs",
    )

    details = hybrid.search("query without lexical terms", trace=trace, return_details=True)
    reranked = CoreReranker(
        _settings(tmp_path, rerank_enabled=False, rerank_provider="none"),
        config=RerankConfig(enabled=False, top_k=3),
    ).rerank("query without lexical terms", details.results, trace=trace)

    assert [item.chunk_id for item in reranked.results] == ["one", "two"]
    sparse_stage = trace.get_stage_data("sparse_retrieval")
    assert sparse_stage["method"] == "bm25"
    assert sparse_stage["status"] == "skipped"
    assert sparse_stage["reason"] == "no_keywords"
    assert sparse.calls == []
    rerank_stage = trace.get_stage_data("rerank")
    assert rerank_stage["method"] == "none"
    assert rerank_stage["status"] == "skipped"
    assert rerank_stage["reason"] == "disabled"

    disabled_trace = TraceContext(trace_type="query")
    disabled_dense = FixedRoute([_result("ignored-dense", 0.8, "Ignored.", "d.md")])
    disabled_hybrid = HybridSearch(
        _settings(tmp_path), query_processor=FixedProcessor(), dense_retriever=disabled_dense,
        sparse_retriever=FixedRoute(), config=HybridSearchConfig(enable_dense=False), collection="docs",
    )
    disabled_hybrid.search("query with a sparse route", trace=disabled_trace, return_details=True)
    dense_stage = disabled_trace.get_stage_data("dense_retrieval")
    assert dense_stage["status"] == "skipped"
    assert dense_stage["reason"] == "disabled_or_unconfigured"
    assert disabled_dense.calls == []
