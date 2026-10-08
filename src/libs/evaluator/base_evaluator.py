"""检索与生成质量评估的统一接口。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """通过同一接口选择轻量指标或后续模型评估。"""

    @abstractmethod
    def evaluate(
        self, query: str, retrieved_chunks: list[Any], generated_answer=None,
        ground_truth=None, trace=None, **kwargs,
    ) -> dict[str, float]:
        """返回指标名称到分数的映射。"""
        raise NotImplementedError

    def validate_query(self, query: str) -> None:
        """验证评估问题。"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: list[Any]) -> None:
        """沿用上游 B6 的非空列表约定。"""
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """评估关闭时返回空指标，沿用相同输入接口。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def evaluate(
        self, query, retrieved_chunks, generated_answer=None, ground_truth=None,
        trace=None, **kwargs,
    ) -> dict[str, float]:
        """跳过指标计算。"""
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
