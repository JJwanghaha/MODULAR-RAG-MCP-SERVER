"""D6：已有评分后端的结果转换、截取与错误回退，不新增评分算法。"""

from dataclasses import dataclass, field
from typing import Any
import time

from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.core.trace.snapshots import snapshot_results
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory


class RerankError(RuntimeError):
    """重排失败且调用方明确禁用回退。"""


@dataclass
class RerankConfig:
    """沿用上游开关/数量/回退；不暴露没有实现硬截止的 timeout 字段。"""

    enabled: bool = True
    top_k: int = 5
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """区分正常关闭与实际失败；original_order 保留输入次序用于诊断。"""

    results: list[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str | None = None
    reranker_type: str = "none"
    original_order: list[RetrievalResult] | None = None


class CoreReranker:
    """真正需要多候选重排时才创建后端；调用失败不切换供应商。"""

    def __init__(self, settings: Settings, reranker: BaseReranker | None = None, config: RerankConfig | None = None):
        self.settings = settings
        self.config = config if config is not None else RerankConfig(settings.rerank.enabled, settings.rerank.top_k)
        self._reranker = reranker
        self._initialization_error = None

    @property
    def is_enabled(self) -> bool:
        """只读取开关，不触发模型创建。"""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker) and (
            self._reranker is not None or self.settings.rerank.provider != "none")

    @property
    def reranker_type(self) -> str:
        """标识当前后端，未启用时为 none。"""
        if not self.is_enabled:
            return "none"
        if self._reranker is None:
            return self.settings.rerank.provider
        name = type(self._reranker).__name__
        return "llm" if "LLM" in name else "cross_encoder" if "CrossEncoder" in name else name.lower()

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None,
               trace: Any = None, **kwargs: Any) -> RerankResult:
        """在原重排行为外记录输入顺序、输出分数、关闭/回退及失败耗时。"""
        started = time.monotonic()
        before = (snapshot_results(results) if trace is not None and isinstance(results, list)
                  and all(isinstance(item, RetrievalResult) for item in results) else None)
        try:
            output = self._rerank(query, results, top_k, trace, **kwargs)
        except Exception as exc:
            if trace is not None:
                trace.record_stage("rerank", {"method": self.reranker_type, "status": "error",
                    "error_type": type(exc).__name__, "input_chunks": before},
                    elapsed_ms=(time.monotonic() - started) * 1000)
            raise
        if trace is not None:
            skipped = len(results) < 2 or not self.is_enabled
            trace.record_stage("rerank", {"method": output.reranker_type, "provider": output.reranker_type,
                "status": "fallback" if output.used_fallback else "skipped" if skipped else "success",
                "reason": output.fallback_reason or ("too_few_candidates" if len(results) < 2 else "disabled" if skipped else None),
                "input_count": len(results), "output_count": len(output.results), "top_k": top_k or self.config.top_k,
                "input_chunks": before, "chunks": snapshot_results(output.results)},
                elapsed_ms=(time.monotonic() - started) * 1000)
        return output

    def _rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None,
                trace: Any = None, **kwargs: Any) -> RerankResult:
        """使用模型分数排序；保留原分数、正文和 metadata，不接受伪造候选。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonblank string")
        limit = self.config.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(results, list) or any(not isinstance(item, RetrievalResult) for item in results):
            raise ValueError("results must be list[RetrievalResult]")
        output = RerankResult(results=list(results[:limit]), reranker_type=self.reranker_type, original_order=list(results))
        if len(results) < 2 or not self.is_enabled:
            return output
        try:
            if self._initialization_error is not None:
                raise RerankError(self._initialization_error)
            if self._reranker is None:
                try:
                    self._reranker = RerankerFactory.create(self.settings)
                except Exception as exc:
                    self._initialization_error = type(exc).__name__
                    raise
            if isinstance(self._reranker, NoneReranker):
                output.reranker_type = "none"
                return output
            originals = {result.chunk_id: result for result in results}
            if len(originals) != len(results):
                raise ValueError("Candidate IDs must be unique")
            candidates = [{"id": result.chunk_id, "text": result.text, "score": result.score,
                           "metadata": dict(result.metadata)} for result in results]
            ranked = self._reranker.rerank(query=query, candidates=candidates, trace=trace, **kwargs)
            if not isinstance(ranked, list) or len(ranked) != len(results):
                raise ValueError("Reranker must return every candidate exactly once")
            converted, seen = [], set()
            for candidate in ranked:
                cid = candidate["id"]
                if cid not in originals or cid in seen or "rerank_score" not in candidate:
                    raise ValueError("Unknown, duplicate or unscored rerank candidate")
                seen.add(cid)
                original = originals[cid]
                converted.append(RetrievalResult(cid, candidate["rerank_score"], original.text,
                    {**original.metadata, "original_score": original.score, "rerank_score": candidate["rerank_score"], "reranked": True}))
            converted.sort(key=lambda item: item.score, reverse=True)
            output.results, output.reranker_type = converted[:limit], self.reranker_type
            return output
        except Exception as exc:
            reason = f"Reranking failed: {type(exc).__name__}"
            if not self.config.fallback_on_error:
                raise RerankError(reason) from None
            output.results = [RetrievalResult(result.chunk_id, result.score, result.text,
                              {**result.metadata, "reranked": False, "rerank_fallback": True}) for result in results[:limit]]
            output.used_fallback, output.fallback_reason = True, reason
            return output


def create_core_reranker(settings: Settings, reranker: BaseReranker | None = None) -> CoreReranker:
    """便捷入口；创建 Core 不会因此初始化评分模型。"""
    return CoreReranker(settings, reranker)
