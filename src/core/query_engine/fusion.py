"""D4：沿用上游 RRF，只融合排名，不比较两路原始分数。"""

from math import isfinite
from typing import Any

from src.core.types import RetrievalResult


class RRFFusion:
    """每路应包含唯一 Chunk ID；输出 score 替换成 RRF 分数。"""

    DEFAULT_K = 60

    def __init__(self, k: int = DEFAULT_K):
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def fuse(self, ranking_lists: list[list[RetrievalResult]], top_k: int | None = None,
             trace: Any = None) -> list[RetrievalResult]:
        """普通融合等价于各路权重 1；全为空时返回空列表。"""
        return self.fuse_with_weights(ranking_lists, top_k=top_k, trace=trace)

    def fuse_with_weights(self, ranking_lists: list[list[RetrievalResult]], weights: list[float] | None = None,
                          top_k: int | None = None, trace: Any = None) -> list[RetrievalResult]:
        """保留首次命中的正文和顶层 metadata 副本，同分按 chunk_id 确定性排序。"""
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        weights = [1.0] * len(ranking_lists) if weights is None else weights
        if len(weights) != len(ranking_lists) or any(not isinstance(w, (int, float)) or isinstance(w, bool)
                                                   or not isfinite(w) or w < 0 for w in weights):
            raise ValueError("weights must be finite nonnegative numbers matching ranking_lists")
        scores, originals = {}, {}
        for ranking, weight in zip(ranking_lists, weights):
            for rank, result in enumerate(ranking, 1):
                originals.setdefault(result.chunk_id, result)
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (self.k + rank)
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
        if top_k is not None:
            ordered = ordered[:top_k]
        return [RetrievalResult(cid, scores[cid], originals[cid].text, dict(originals[cid].metadata)) for cid in ordered]


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """返回单路某一名次的贡献；k 不是返回数量。"""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    return 1.0 / (RRFFusion(k).k + rank)
