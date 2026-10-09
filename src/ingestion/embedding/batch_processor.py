"""C10：按原始顺序编排稠密和稀疏编码。"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


@dataclass
class BatchResult:
    """保留上游结果字段；失败抛异常，所以成功返回时 failed_chunks 为零。"""

    dense_vectors: list[list[float]]
    sparse_stats: list[dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """两种编码都完成才接收该批，避免静默错位。"""

    def __init__(self, dense_encoder: DenseEncoder, sparse_encoder: SparseEncoder, batch_size: int = 100):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(self, chunks: list[Chunk], trace: Any = None) -> BatchResult:
        """不做隐式重试或模型切换，也不写入任何存储。"""
        if not chunks:
            raise ValueError("Cannot process empty chunks list")
        started = perf_counter()
        vectors, stats = [], []
        batches = self._create_batches(chunks)
        for number, batch in enumerate(batches):
            try:
                dense = self.dense_encoder.encode(batch, trace=trace)
                sparse = self.sparse_encoder.encode(batch, trace=trace)
                if len(dense) != len(batch) or len(sparse) != len(batch):
                    raise ValueError("Encoder output count does not match batch")
                if [stat["chunk_id"] for stat in sparse] != [chunk.id for chunk in batch]:
                    raise ValueError("Sparse output order does not match batch")
            except Exception as exc:
                raise RuntimeError(f"Encoding batch {number} failed: {type(exc).__name__}") from exc
            vectors.extend(dense)
            stats.extend(sparse)
        return BatchResult(vectors, stats, len(batches), perf_counter() - started, len(chunks), 0)

    def _create_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [chunks[start:start + self.batch_size] for start in range(0, len(chunks), self.batch_size)]

    def get_batch_count(self, total_chunks: int) -> int:
        """外层处理批数，可能不同于 DenseEncoder 的模型调用批数。"""
        return max(0, (total_chunks + self.batch_size - 1) // self.batch_size)
