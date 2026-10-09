"""C1 共用数据类型；对象不负责解析、编码或数据库写入。"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """解析后的整篇文档；正文可包含图片占位符，metadata 至少有来源。"""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """沿用上游最小约束，不把解析或切分规则放进数据对象。"""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        """返回独立的字典结构；调用方负责传入 JSON 可序列化的 metadata。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """按已约定字段恢复文档对象。"""
        return cls(**data)


@dataclass
class Chunk:
    """切分后的内容；偏移和父文档引用是可选字段，不自动写入 metadata。"""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int | None = None
    end_offset: int | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        """与上游一致，检查来源字段存在。"""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        """偏移、引用及 metadata 一并序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """恢复切片对象，不重新执行切分。"""
        return cls(**data)


@dataclass
class ChunkRecord:
    """内容与可选向量的共用载体，不等同于 Chroma 的实际记录格式。"""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """保持上游的来源约束。"""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        """将可选向量和正文转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """恢复载体；不验证或生成模型向量。"""
        return cls(**data)

    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: list[float] | None = None,
                   sparse_vector: dict[str, float] | None = None) -> "ChunkRecord":
        """复制顶层 metadata；上游不自动搬运独立偏移字段。"""
        return cls(chunk.id, chunk.text, chunk.metadata.copy(), dense_vector, sparse_vector)
