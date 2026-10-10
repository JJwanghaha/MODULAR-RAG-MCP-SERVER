"""D1：复现上游 jieba、停用词、关键词去重和 key:value 过滤语法。"""

import re
from dataclasses import dataclass, field

from src.core.types import ProcessedQuery

CHINESE_STOPWORDS = set("""如何 怎么 怎样 什么 哪个 哪些 为什么 为何 谁 多少 几 是否 能否 可否
的 地 得 了 着 过 吗 呢 吧 啊 呀 在 于 和 与 或 及 并 而 但 但是 因为 所以 如果 那么 虽然 然而
我 你 他 她 它 我们 你们 他们 这 那 这个 那个 这些 那些 这里 那里 很 非常 特别 更 最 都 也 还 又 再
已 已经 正在 将 会 能 可以 应该 必须 是 有 做 进行 使用 通过 个 种 类 ？ 。 ！ ， 、""".split())
ENGLISH_STOPWORDS = set("""a an the in on at to for of with by from as into about through between after before
and or but if then because while although i you he she it we they this that these those what which who whom whose
is am are was were be been being have has had do does did will would could should may might must can
get use make how why when where not no yes so very just also too""".split())
DEFAULT_STOPWORDS = CHINESE_STOPWORDS | ENGLISH_STOPWORDS
FILTER_PATTERN = re.compile(r"(\w+):([^\s]+)")


@dataclass
class QueryProcessorConfig:
    """沿用上游默认；查询最小词长 1，摄取端默认 2，两者不等同。"""

    stopwords: set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """不调用 LLM、不改写同义词；过滤语法只解析，不代表已应用或授权。"""

    def __init__(self, config: QueryProcessorConfig | None = None):
        self.config = config if config is not None else QueryProcessorConfig()
        if self.config.min_keyword_length < 1 or self.config.max_keywords < 1:
            raise ValueError("Keyword length and count limits must be positive")
        try:
            import jieba
        except ImportError as exc:
            raise ImportError("QueryProcessor requires the 'sparse' extra") from exc
        self._jieba = jieba

    def process(self, query: str) -> ProcessedQuery:
        """原查询不变；仅在关键词提取时规范空白并移除过滤表达式。"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        filters = {}
        text = " ".join(query.split())
        if self.config.enable_filter_parsing:
            for key, value in FILTER_PATTERN.findall(text):
                normalized = key.lower()
                if normalized in {"collection", "col", "c"}:
                    filters["collection"] = value
                elif normalized in {"type", "doc_type", "t"}:
                    filters["doc_type"] = value
                elif normalized in {"source", "src", "s"}:
                    filters["source_path"] = value
                elif normalized in {"tag", "tags"}:
                    filters.setdefault("tags", []).extend(value.split(","))
                else:
                    filters[key] = value
            text = FILTER_PATTERN.sub("", text).strip()
        keywords, seen = [], set()
        for token in self._jieba.lcut(text):
            token = token.strip()
            lowered = token.lower()
            if not token or re.fullmatch(r"[\s\W]+", token) or lowered in seen:
                continue
            if token in self.config.stopwords or lowered in self.config.stopwords or len(token) < self.config.min_keyword_length:
                continue
            seen.add(lowered)
            keywords.append(token)
            if len(keywords) == self.config.max_keywords:
                break
        return ProcessedQuery(query, keywords, filters)

    def add_stopwords(self, words: set[str]) -> None:
        """仅调整当前 processor 的停用词集合。"""
        self.config.stopwords.update(words)

    def remove_stopwords(self, words: set[str]) -> None:
        """允许实验保留被默认规则过滤的词。"""
        self.config.stopwords.difference_update(words)


def create_query_processor(stopwords: set[str] | None = None, min_keyword_length: int = 1,
                           max_keywords: int = 20, enable_filter_parsing: bool = True) -> QueryProcessor:
    """上游便捷构造入口，不新增 Registry。"""
    return QueryProcessor(QueryProcessorConfig(DEFAULT_STOPWORDS.copy() if stopwords is None else set(stopwords),
                          min_keyword_length, max_keywords, enable_filter_parsing))
