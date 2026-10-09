"""C9：jieba 分词、词频和长度统计；这里不构建索引。"""

import re
from collections import Counter
from typing import Any

from src.core.types import Chunk


class SparseEncoder:
    """沿用上游中英文分词、转小写和最小词长规则。"""

    def __init__(self, min_term_length: int = 2, lowercase: bool = True):
        if min_term_length < 1:
            raise ValueError("min_term_length must be at least 1")
        try:
            import jieba
        except ImportError as exc:
            raise ImportError("SparseEncoder requires the 'sparse' extra") from exc
        self._jieba = jieba
        self.min_term_length = min_term_length
        self.lowercase = lowercase

    def encode(self, chunks: list[Chunk], trace: Any = None) -> list[dict[str, Any]]:
        """返回 chunk_id、term_frequencies、doc_length、unique_terms。"""
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        results = []
        for chunk in chunks:
            if not chunk.text.strip():
                raise ValueError("Cannot encode blank chunk text")
            terms = self._tokenize(chunk.text)
            counts = Counter(terms)
            results.append({"chunk_id": chunk.id, "term_frequencies": dict(counts),
                            "doc_length": len(terms), "unique_terms": len(counts)})
        return results

    def _tokenize(self, text: str) -> list[str]:
        """按上游规则过滤空白、纯标点和过短词；不额外去停用词。"""
        terms = []
        for token in self._jieba.lcut(text):
            token = token.strip()
            if not token or re.fullmatch(r"[\s\W]+", token):
                continue
            if self.lowercase:
                token = token.lower()
            if len(token) >= self.min_term_length:
                terms.append(token)
        return terms

    def get_corpus_stats(self, encoded_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """文档频率按块计数，不是词出现总次数。"""
        df = Counter(term for stat in encoded_chunks for term in stat["term_frequencies"])
        return {"num_docs": len(encoded_chunks),
                "avg_doc_length": sum(stat["doc_length"] for stat in encoded_chunks) / len(encoded_chunks) if encoded_chunks else 0.0,
                "document_frequency": dict(df)}
