"""用问题/候选文本对预测相关性，不生成 JSON 或改写候选内容。"""

from math import isfinite
from numbers import Real
from typing import Any

from src.libs.reranker.base_reranker import BaseReranker


class CrossEncoderRerankError(RuntimeError):
    """推理、模型加载或分数契约失败，由 Core 决定回退。"""


class CrossEncoderReranker(BaseReranker):
    """注入 predict 评分器可离线使用；真实权重只从本地加载。"""

    def __init__(self, settings: Any, model=None, **kwargs: Any) -> None:
        self.settings = settings
        if model is not None:
            self.model = model
        else:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise CrossEncoderRerankError(
                    "sentence-transformers is not installed; install .[rerankers] separately"
                ) from exc
            try:
                self.model = CrossEncoder(
                    settings.rerank.model,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise CrossEncoderRerankError(
                    f"Unable to load local Cross-Encoder: {type(exc).__name__}"
                ) from exc

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], trace=None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """同步预测后排序；默认返回全部候选，显式 top_k 可截断。"""
        self.validate_query(query)
        self.validate_candidates(candidates)
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        pairs = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Candidate text must be a non-empty string")
            pairs.append((query, text))
        try:
            scores = self.model.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        except Exception as exc:
            raise CrossEncoderRerankError(
                f"Cross-Encoder prediction failed: {type(exc).__name__}"
            ) from None
        if not isinstance(scores, (list, tuple)) or len(scores) != len(candidates):
            raise CrossEncoderRerankError("Cross-Encoder must return one score per candidate")
        if not all(
            isinstance(score, Real) and not isinstance(score, bool) and isfinite(score)
            for score in scores
        ):
            raise CrossEncoderRerankError("Cross-Encoder scores must be finite numbers")
        result = [
            dict(candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        result.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
        return result[:top_k]
