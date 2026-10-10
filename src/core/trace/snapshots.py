"""查询 Trace 的候选快照：保留全文和分数，不持有可变结果对象。"""

from src.core.types import RetrievalResult


def snapshot_results(results: list[RetrievalResult]) -> list[dict]:
    """只复制展示所需的标量字段，不复制模型、向量或图片载荷。"""
    return [{"chunk_id": item.chunk_id, "score": item.score, "text": item.text,
             "source": item.metadata.get("source_path", item.metadata.get("source", "")),
             "title": item.metadata.get("title", ""),
             "original_score": item.metadata.get("original_score", item.score)} for item in results]
