"""向量存储接口；正式数据库实现留给 B7.6。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """使用字典记录承载上游 B4 的 ID、向量、文本和元数据。"""

    @abstractmethod
    def upsert(self, records: list[dict[str, Any]], trace=None, **kwargs) -> None:
        """按 ID 插入或更新；重复写入应得到相同状态。"""
        raise NotImplementedError

    @abstractmethod
    def query(
        self, vector: list[float], top_k: int = 10, filters=None, trace=None, **kwargs
    ) -> list[dict[str, Any]]:
        """返回按相关度降序排列的记录，至少包含 id、score、metadata。"""
        raise NotImplementedError

    def validate_records(self, records: list[dict[str, Any]]) -> None:
        """沿用上游 B4 的最小记录校验，维度规则由具体存储负责。"""
        if not records:
            raise ValueError("Records list cannot be empty")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {index} is not a dict")
            for key in ("id", "vector"):
                if key not in record:
                    raise ValueError(f"Record at index {index} is missing required field: '{key}'")
            if not isinstance(record["vector"], (list, tuple)):
                raise ValueError(f"Record at index {index} has invalid vector type")
            if not record["vector"]:
                raise ValueError(f"Record at index {index} has empty vector")

    def validate_query_vector(self, vector: list[float], top_k: int) -> None:
        """验证查询向量形状和结果数量。"""
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Query vector must be a list or tuple")
        if not vector:
            raise ValueError("Query vector cannot be empty")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
