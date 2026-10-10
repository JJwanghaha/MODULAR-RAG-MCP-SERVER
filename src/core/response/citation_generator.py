"""E3/E6：把检索命中转换为可回查的引用，不生成回答。"""

from dataclasses import dataclass, field
from typing import Any

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """页码仅沿用已有元数据提示，不保证精确文档定位。"""

    index: int
    chunk_id: str
    source: str
    score: float
    text_snippet: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {"index": self.index, "chunk_id": self.chunk_id, "source": self.source,
                  "score": self.score, "text_snippet": self.text_snippet, "metadata": dict(self.metadata)}
        if self.page is not None:
            result["page"] = self.page
        return result


class CitationGenerator:
    def __init__(self, snippet_max_length: int = 200, include_metadata_fields: list[str] | None = None):
        if snippet_max_length < 1:
            raise ValueError("snippet_max_length must be positive")
        self.snippet_max_length = snippet_max_length
        self.include_metadata_fields = include_metadata_fields if include_metadata_fields is not None else [
            "title", "section", "chunk_index", "doc_type", "source_ref", "doc_hash"]

    def generate(self, results: list[RetrievalResult]) -> list[Citation]:
        citations = []
        for index, result in enumerate(results, 1):
            metadata = result.metadata
            page = metadata.get("page") if metadata.get("page") is not None else metadata.get("page_num")
            try:
                page = int(page) if page is not None and not isinstance(page, bool) else None
            except (ValueError, TypeError):
                page = None
            text = " ".join(result.text.split())
            snippet = text[:self.snippet_max_length] + ("..." if len(text) > self.snippet_max_length else "")
            citations.append(Citation(index, result.chunk_id, str(metadata.get("source_path", "unknown")),
                                      result.score, snippet, page,
                                      {key: metadata[key] for key in self.include_metadata_fields if key in metadata}))
        return citations

    def format_citation_marker(self, index: int) -> str:
        return f"[{index}]"
