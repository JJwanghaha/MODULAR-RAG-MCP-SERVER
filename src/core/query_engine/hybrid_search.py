"""D5：按上游组合查询处理、双路检索、融合和失败降级。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
import time
from typing import Any

from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.query_processor import QueryProcessor
from src.core.settings import Settings
from src.core.types import ProcessedQuery, RetrievalResult
from src.core.trace.snapshots import snapshot_results


@dataclass
class HybridSearchConfig:
    """只控制已有检索策略，不创建新的模型配置。"""

    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True


@dataclass
class HybridSearchResult:
    """上游详细结果字段；错误只存类别，区分空命中与失败。"""

    results: list[RetrievalResult] = field(default_factory=list)
    dense_results: list[RetrievalResult] | None = None
    sparse_results: list[RetrievalResult] | None = None
    dense_error: str | None = None
    sparse_error: str | None = None
    used_fallback: bool = False
    processed_query: ProcessedQuery | None = None


class HybridSearch:
    """实例绑定已注入的集合，不把 collection 当成 metadata 条件或访问权限。"""

    def __init__(self, settings: Settings | None = None, query_processor=None, dense_retriever=None,
                 sparse_retriever=None, fusion=None, config: HybridSearchConfig | None = None,
                 *, collection: str | None = None):
        retrieval = settings.retrieval if settings is not None else None
        self.config = config if config is not None else HybridSearchConfig(
            dense_top_k=retrieval.dense_top_k if retrieval else 20,
            sparse_top_k=retrieval.sparse_top_k if retrieval else 20,
            fusion_top_k=retrieval.fusion_top_k if retrieval else 10,
        )
        for value in (self.config.dense_top_k, self.config.sparse_top_k, self.config.fusion_top_k):
            self._validate_top_k(value)
        self.query_processor = query_processor
        self.dense_retriever, self.sparse_retriever = dense_retriever, sparse_retriever
        self.fusion = fusion if fusion is not None else RRFFusion(retrieval.rrf_k if retrieval else 60)
        stores = [getattr(retriever, "vector_store", None) for retriever in (dense_retriever, sparse_retriever)]
        names = {store.collection_name for store in stores if getattr(store, "collection_name", None)}
        if collection is not None:
            names.add(collection)
        if len(names) > 1:
            raise ValueError("Retrievers must use the same collection")
        self.collection = next(iter(names), None)

    @staticmethod
    def _validate_top_k(value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("top_k must be a positive integer")

    @staticmethod
    def _validate_filters(filters: dict[str, Any]) -> None:
        """本批支持平面相等条件、来源子串、标签交集，不实现 Chroma 操作符语言。"""
        for key, value in filters.items():
            if not isinstance(key, str) or key.startswith("$"):
                raise ValueError("Filter keys must be plain metadata names")
            if key == "tags":
                if isinstance(value, str):
                    continue
                if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
                    continue
                raise ValueError("tags filter must be a string or a list of strings")
            if key in {"collection", "source_path"} and not isinstance(value, str):
                raise ValueError("collection and source_path filters must be strings")
            if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not isfinite(value):
                raise ValueError("Filter values must be finite scalars")

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "tags":
                tags = metadata.get("tags", [])
                if isinstance(tags, str):
                    tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
                expected = [value] if isinstance(value, str) else value
                if not isinstance(tags, (list, tuple)) or not set(tags).intersection(expected):
                    return False
            elif key == "source_path":
                if value not in str(metadata.get("source_path", "")):
                    return False
            elif metadata.get(key) != value:
                return False
        return True

    def search(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None,
               trace: Any = None, return_details: bool = False) -> list[RetrievalResult] | HybridSearchResult:
        """成功的空结果不算失败；所有实际执行的路线都失败则明确报错。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonblank string")
        limit = self.config.fusion_top_k if top_k is None else top_k
        self._validate_top_k(limit)
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")
        started = time.monotonic()
        processed = (self.query_processor.process(query) if self.query_processor is not None
                     else ProcessedQuery(query, query.split()))
        if trace is not None:
            trace.record_stage("query_processing", {"method": "query_processor", "status": "success",
                "original_query": query, "keywords": list(processed.keywords),
                "filters": dict(processed.filters)}, elapsed_ms=(time.monotonic() - started) * 1000)
        merged = {**processed.filters, **(filters or {})}
        self._validate_filters(merged)
        selected = merged.pop("collection", self.collection)
        if self.collection is not None and selected != self.collection:
            raise ValueError("Requested collection differs from the bound retrievers; create matching retrievers")
        if merged and not self.config.metadata_filter_post:
            raise ValueError("Metadata filtering must be enabled when filters are requested")
        # 能准确下推的相等条件送入 Chroma；来源子串和 CSV 标签保留给召回后过滤。
        dense_filters = {key: value for key, value in merged.items() if key not in {"source_path", "tags"}}
        jobs = {}
        if self.config.enable_dense and self.dense_retriever is not None:
            jobs["dense"] = lambda: self.dense_retriever.retrieve(
                query=processed.original_query, top_k=self.config.dense_top_k,
                filters=dense_filters or None, trace=trace)
        sparse_configured = self.config.enable_sparse and self.sparse_retriever is not None
        if sparse_configured and processed.keywords:
            jobs["sparse"] = lambda: self.sparse_retriever.retrieve(
                keywords=processed.keywords, top_k=self.config.sparse_top_k, collection=selected, trace=trace)
        if not jobs and not sparse_configured:
            raise RuntimeError("No retrieval route configured and enabled")
        outputs, errors, timings = {}, {}, {}

        def run_job(name, callback):
            started = time.monotonic()
            try:
                result = callback()
                if not isinstance(result, list) or any(not isinstance(item, RetrievalResult) for item in result):
                    raise TypeError("Retriever output must be list[RetrievalResult]")
                return result, None, (time.monotonic() - started) * 1000
            except Exception as exc:
                return None, f"{name} retrieval failed: {type(exc).__name__}", (time.monotonic() - started) * 1000

        if self.config.parallel_retrieval and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name: pool.submit(run_job, name, callback) for name, callback in jobs.items()}
                for name, future in futures.items():
                    outputs[name], errors[name], timings[name] = future.result()
        else:
            for name, callback in jobs.items():
                outputs[name], errors[name], timings[name] = run_job(name, callback)
        # 由编排线程写 Trace，两个计时区间可以重叠；不把事件顺序当执行时间线。
        if trace is not None:
            for name, retriever in (("dense", self.dense_retriever), ("sparse", self.sparse_retriever)):
                result = outputs.get(name) or []
                reason = None if name in jobs else "no_keywords" if name == "sparse" and sparse_configured else "disabled_or_unconfigured"
                trace.record_stage(f"{name}_retrieval", {
                    "method": "dense" if name == "dense" else "bm25",
                    "provider": type(retriever).__name__ if retriever is not None else "none",
                    "status": "skipped" if name not in jobs else "error" if errors.get(name) else "success",
                    "reason": reason, "error": errors.get(name),
                    "top_k": self.config.dense_top_k if name == "dense" else self.config.sparse_top_k,
                    "result_count": len(result), "chunks": snapshot_results(result),
                }, elapsed_ms=timings.get(name, 0.0))
        if jobs and all(errors[name] is not None for name in jobs):
            raise RuntimeError("All active retrieval routes failed: " + "; ".join(errors.values()))
        details = HybridSearchResult(dense_results=outputs.get("dense"), sparse_results=outputs.get("sparse"),
                                     dense_error=errors.get("dense"), sparse_error=errors.get("sparse"),
                                     used_fallback=any(errors.values()), processed_query=processed)
        # 先在各路已召回的候选中应用条件，保持其顺序，再融合/截取；不宣称无限补召回。
        started = time.monotonic()
        rankings = [[item for item in result if self._matches_filters(item.metadata, merged)]
                    for result in outputs.values() if result is not None]
        nonempty = [ranking for ranking in rankings if ranking]
        if len(nonempty) > 1 and not details.used_fallback:
            details.results = self.fusion.fuse(nonempty, top_k=limit, trace=trace)
        else:
            details.results = nonempty[0][:limit] if nonempty else []
        if trace is not None:
            trace.record_stage("fusion", {"method": "rrf" if len(nonempty) > 1 and not details.used_fallback else "single_route_or_empty",
                "status": "fallback" if details.used_fallback else "success", "rrf_k": self.fusion.k if isinstance(self.fusion, RRFFusion) else None,
                "filters": dict(merged), "input_lists": len(rankings), "top_k": limit,
                "result_count": len(details.results), "chunks": snapshot_results(details.results),
                "dense_error": details.dense_error, "sparse_error": details.sparse_error},
                elapsed_ms=(time.monotonic() - started) * 1000)
        return details if return_details else details.results


def create_hybrid_search(settings: Settings | None = None, query_processor=None, dense_retriever=None,
                         sparse_retriever=None, fusion=None, *, collection: str | None = None) -> HybridSearch:
    """只连接注入的检索器；不在此自动创建外部模型或数据库。"""
    return HybridSearch(settings, query_processor, dense_retriever, sparse_retriever, fusion, collection=collection)
