"""C8：把 Chunk 正文按批交给已有 Embedding 接口。"""

from typing import Any

from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """只编排编码，不选择模型、不写数据库。"""

    def __init__(self, embedding: BaseEmbedding, batch_size: int = 100):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.embedding = embedding
        self.batch_size = batch_size

    def encode(self, chunks: list[Chunk], trace: Any = None) -> list[list[float]]:
        """结果与输入逐项对应；任意批次失败就停止，不返回部分向量。"""
        texts = [chunk.text for chunk in chunks]
        self.embedding.validate_texts(texts)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            try:
                encoded = self.embedding.embed(batch, trace=trace)
                self.embedding.validate_vectors(encoded, len(batch))
            except Exception as exc:
                raise RuntimeError(f"Dense encoding failed for batch starting at {start}: {type(exc).__name__}") from exc
            vectors.extend(encoded)
        return vectors

    def get_batch_count(self, num_chunks: int) -> int:
        """返回需要的模型调用批数。"""
        return max(0, (num_chunks + self.batch_size - 1) // self.batch_size)
