"""按上游配置封装 LangChain 递归字符切分，不提供 Markdown 结构保护。"""

from typing import Any

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """先按自然边界拆分，再合并小片段；长度单位固定为字符。"""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        settings: Any,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters is not installed; install .[splitters]"
            ) from exc

        self.settings = settings
        try:
            ingestion = settings.ingestion
            self.chunk_size = (
                chunk_size if chunk_size is not None else ingestion.chunk_size
            )
            self.chunk_overlap = (
                chunk_overlap if chunk_overlap is not None else ingestion.chunk_overlap
            )
        except AttributeError as exc:
            raise ValueError("Missing ingestion configuration: chunk_size and chunk_overlap") from exc
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )

    def split_text(self, text: str, trace=None, **kwargs: Any) -> list[str]:
        """返回有序文本片段；ID、标题和来源元数据由 C 阶段处理。"""
        self.validate_text(text)
        try:
            chunks = self._splitter.split_text(text)
            if not chunks:
                chunks = [text]
            self.validate_chunks(chunks)
            return chunks
        except Exception as exc:
            raise RuntimeError(
                f"RecursiveSplitter failed: {type(exc).__name__}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from exc
