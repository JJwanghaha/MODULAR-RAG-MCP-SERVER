"""所有 Embedding adapter 共同遵循的 interface。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """批量文本向量化 adapter 的抽象基类。"""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """把 N 段文本编码为 N 个向量。"""
        raise NotImplementedError

    def validate_texts(self, texts: list[str]) -> None:
        """验证所有 Embedding adapter 共享的输入约束。"""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {index} is not a string "
                    f"(type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(f"Text at index {index} is empty or whitespace-only")

    def get_dimension(self) -> int:
        """返回 adapter 生成的向量维度。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension() method"
        )
