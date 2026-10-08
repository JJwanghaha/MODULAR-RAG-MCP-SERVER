"""文本切分策略的统一接口；文档元数据由 C4 负责。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """把一段文本切成有序字符串列表。"""

    @abstractmethod
    def split_text(
        self, text: str, trace: Any | None = None, **kwargs: Any
    ) -> list[str]:
        """具体策略负责切分，返回的片段保持原文顺序。"""
        raise NotImplementedError

    def validate_text(self, text: str) -> None:
        """遵循上游约定，只接受非空字符串。"""
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")

    def validate_chunks(self, chunks: list[str]) -> None:
        """验证策略返回的文本片段类型和内容。"""
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list of strings")
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(f"Chunk at index {index} is not a string")
            if not chunk.strip():
                raise ValueError(f"Chunk at index {index} is empty or whitespace-only")
