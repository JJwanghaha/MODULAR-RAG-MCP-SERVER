"""重排接口与保持原顺序的默认实现。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """根据问题重新排列候选，保留候选内容。"""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict[str, Any]], trace=None, **kwargs):
        """返回排序后的候选列表。"""
        raise NotImplementedError

    def validate_query(self, query: str) -> None:
        """接受非空字符串问题。"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """遵循上游非空字典列表契约。"""
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list of dicts")
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"Candidate at index {index} is not a dict")


class NoneReranker(BaseReranker):
    """重排关闭时返回浅拷贝，候选顺序保持不变。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def rerank(self, query: str, candidates: list[dict[str, Any]], trace=None, **kwargs):
        """使用同一接口完成无重排路径。"""
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
