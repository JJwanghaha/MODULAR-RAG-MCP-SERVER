"""D1-D4 检索基线的离线公开契约测试。"""

from contextlib import contextmanager
from dataclasses import replace
import math
import socket

import pytest

from src.core.settings import load_settings
from src.core.types import ProcessedQuery, RetrievalResult
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.fusion import RRFFusion, rrf_score
from src.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
)
from src.core.query_engine.sparse_retriever import (
    SparseRetriever,
    create_sparse_retriever,
)
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def offline_process(monkeypatch):
    """临时屏蔽模型 Key 并拒绝测试进程发起网络连接。"""
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("检索基线测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def _settings_for(tmp_path, collection="notes"):
    """使用项目默认配置结构，但把持久化目录放进当前测试临时目录。"""
    settings = load_settings()
    vector_store = replace(
        settings.vector_store,
        persist_directory=str(tmp_path / "chroma"),
        collection_name=collection,
    )
    return replace(settings, vector_store=vector_store)


def _offline_ingestion_settings(tmp_path):
    """关闭摄取阶段的可选文本模型，并限制向量维度供离线 fake 使用。"""
    settings = _settings_for(tmp_path)
    ingestion = replace(
        settings.ingestion,
        chunk_refiner={"use_llm": False},
        metadata_enricher={"use_llm": False},
    )
    return replace(
        settings,
        ingestion=ingestion,
        embedding=replace(settings.embedding, dimensions=2),
    )


@contextmanager
def _open_chroma(settings, persist_directory, collection_name="notes"):
    """只在调用方提供的临时目录打开真实 Chroma。"""
    store = ChromaStore(
        settings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    try:
        yield store
    finally:
        store.client.close()


class FakeEmbedding(BaseEmbedding):
    """可记录调用并返回指定输出的离线 Embedding adapter。"""

    def __init__(self, dimension=2, vectors=None, error=None):
        self.dimension = dimension
        self.vectors = [[1.0, 0.0]] if vectors is None else vectors
        self.error = error
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.append((list(texts), trace, kwargs.copy()))
        if self.error is not None:
            raise self.error
        return self.vectors

    def get_dimension(self):
        return self.dimension


class FakeVectorStore(BaseVectorStore):
    """记录公开 query 参数并返回测试指定的结果。"""

    def __init__(self, results=None, error=None, collection_name="notes"):
        self.results = [] if results is None else results
        self.error = error
        self.collection_name = collection_name
        self.query_calls = []
        self.get_calls = []

    def upsert(self, records, trace=None, **kwargs):
        self.validate_records(records)

    def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
        self.query_calls.append((vector, top_k, filters, trace))
        if self.error is not None:
            raise self.error
        return self.results

    def get_by_ids(self, ids, trace=None, **kwargs):
        self.get_calls.append((list(ids), trace))
        return []


def _bm25_stats(*rows):
    """构造 BM25Indexer 接受的确定词频记录。"""
    return [
        {
            "chunk_id": chunk_id,
            "term_frequencies": dict(frequencies),
            "doc_length": sum(frequencies.values()),
        }
        for chunk_id, frequencies in rows
    ]


def _chroma_records(store, *rows):
    """把 ID、正文和元数据写入测试专属 Chroma 集合。"""
    store.upsert(
        [
            {
                "id": chunk_id,
                "vector": vector,
                "metadata": {"text": text, **metadata},
            }
            for chunk_id, vector, text, metadata in rows
        ]
    )


# D1：中文和英文查询解析


def test_query_processor_uses_jieba_stopwords_and_case_insensitive_deduplication():
    """真实 jieba 解析中英文，停用词移除且大小写变体只保留首次出现。"""
    processor = create_query_processor()

    result = processor.process("如何 Explain RAG rag 使用 Retriever RETRIEVER retrieval 的方案")

    assert isinstance(result, ProcessedQuery)
    assert result.original_query == "如何 Explain RAG rag 使用 Retriever RETRIEVER retrieval 的方案"
    assert "Explain" in result.keywords
    assert "RAG" in result.keywords
    assert "Retriever" in result.keywords
    assert "retrieval" in result.keywords
    assert "如何" not in result.keywords
    assert "使用" not in result.keywords
    assert "的" not in result.keywords
    assert [word.lower() for word in result.keywords].count("rag") == 1
    assert [word.lower() for word in result.keywords].count("retriever") == 1


def test_query_processor_applies_minimum_word_length_and_keyword_limit():
    """最小词长与最大关键词数分别约束提取结果。"""
    from src.ingestion.embedding.sparse_encoder import SparseEncoder

    one_character = QueryProcessor(
        QueryProcessorConfig(stopwords=set(), min_keyword_length=1, max_keywords=2)
    ).process("x Alpha Beta")
    two_characters = QueryProcessor(
        QueryProcessorConfig(stopwords=set(), min_keyword_length=2, max_keywords=20)
    ).process("x Alpha Beta")

    assert one_character.keywords == ["x", "Alpha"]
    assert two_characters.keywords == ["Alpha", "Beta"]
    assert QueryProcessorConfig().min_keyword_length == 1
    assert SparseEncoder().min_term_length == 2


def test_query_processor_parses_filter_aliases_tags_and_generic_filters():
    """过滤别名归一化，标签拆分，未知过滤键保留原名。"""
    result = create_query_processor().process(
        "type:pdf doc_type:note t:wiki source:a.md src:b.md s:c.md "
        "collection:first col:second c:third tag:red,blue tags:green,yellow custom:value"
    )

    assert result.filters == {
        "doc_type": "wiki",
        "source_path": "c.md",
        "collection": "third",
        "tags": ["red", "blue", "green", "yellow"],
        "custom": "value",
    }
    assert "type:pdf" not in result.keywords


def test_query_processor_preserves_original_query_and_can_disable_filter_parsing():
    """原问题逐字保留，关闭过滤解析后不生成结构化 filters。"""
    original = "  type:pdf\n RAG  "
    processor = QueryProcessor(QueryProcessorConfig(enable_filter_parsing=False))

    result = processor.process(original)

    assert result.original_query == original
    assert result.filters == {}


@pytest.mark.parametrize("query", ["", " \t\n "])
def test_query_processor_returns_zero_keywords_for_blank_query(query):
    """空白问题保留原值并返回零关键词。"""
    result = create_query_processor().process(query)

    assert result.original_query == query
    assert result.keywords == []
    assert result.filters == {}


@pytest.mark.parametrize("query", [None, 1, ["RAG"]])
def test_query_processor_rejects_non_string_input(query):
    """非字符串输入在 jieba 调用前被拒绝。"""
    with pytest.raises(ValueError, match="Query must be a string"):
        create_query_processor().process(query)


def test_custom_query_stopwords_are_copied_and_isolated():
    """定制停用词及后续增删只影响各自 processor。"""
    supplied = {"customword"}
    first = create_query_processor(stopwords=supplied)
    second = create_query_processor(stopwords=supplied)
    first.add_stopwords({"specificword"})

    assert "customword" not in first.process("customword specificword").keywords
    assert "customword" not in second.process("customword specificword").keywords
    assert "specificword" in second.process("specificword").keywords
    assert supplied == {"customword"}

    first.remove_stopwords({"customword"})
    assert "customword" in first.process("customword").keywords
    assert "customword" not in second.process("customword").keywords


# D2：DenseRetriever 与 Embedding seam


def test_dense_retriever_uses_retrieval_query_trace_settings_and_explicit_top_k(tmp_path):
    """问题编码任务类型、trace、默认 TopK、显式覆盖和统一结果均可观测。"""
    settings = _settings_for(tmp_path)
    trace = object()
    embedding = FakeEmbedding()
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "score": 0.8,
                "text": "召回正文",
                "metadata": {"source_path": "note.md", "text": "元数据正文"},
            }
        ]
    )
    retriever = DenseRetriever(settings, embedding, store)

    default_result = retriever.retrieve("RAG", trace=trace)
    explicit_result = retriever.retrieve("RAG", top_k=3, filters={"kind": "note"}, trace=trace)

    assert retriever.default_top_k == settings.retrieval.dense_top_k
    assert embedding.calls[0] == (["RAG"], trace, {"task_type": "RETRIEVAL_QUERY"})
    assert store.query_calls[0][1:] == (settings.retrieval.dense_top_k, None, trace)
    assert store.query_calls[1][1:] == (3, {"kind": "note"}, trace)
    assert default_result == explicit_result == [
        RetrievalResult("chunk-1", 0.8, "召回正文", {"source_path": "note.md", "text": "元数据正文"})
    ]


@pytest.mark.parametrize("query", ["", " \t\n "])
def test_dense_retriever_rejects_blank_query_without_calling_embedding(query):
    """空白查询在任何模型或向量存储调用前被拒绝。"""
    embedding = FakeEmbedding()
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="nonblank string"):
        DenseRetriever(embedding_client=embedding, vector_store=store).retrieve(query)

    assert embedding.calls == []
    assert store.query_calls == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2"])
def test_dense_retriever_rejects_invalid_top_k(top_k):
    """TopK 必须是正整数且不能以 bool 冒充整数。"""
    retriever = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("RAG", top_k=top_k)


@pytest.mark.parametrize(
    "vectors,error_type",
    [
        ([], "ValueError"),
        ([[1.0, 2.0, 3.0]], "ValueError"),
        ([[math.nan, 0.0]], "ValueError"),
    ],
)
def test_dense_retriever_validates_embedding_count_dimension_and_finite_values(vectors, error_type):
    """向量数量、维度与有限数值不符时以稳定错误类型报告。"""
    embedding = FakeEmbedding(vectors=vectors)
    store = FakeVectorStore()

    with pytest.raises(RuntimeError, match=f"Query embedding failed: {error_type}"):
        DenseRetriever(embedding_client=embedding, vector_store=store).retrieve("RAG")

    assert store.query_calls == []


def test_dense_retriever_reports_missing_dependencies():
    """缺少模型或存储依赖时明确指出构造不完整。"""
    with pytest.raises(RuntimeError, match="requires embedding_client and vector_store"):
        DenseRetriever().retrieve("RAG")

    with pytest.raises(RuntimeError, match="requires embedding_client and vector_store"):
        DenseRetriever(embedding_client=FakeEmbedding()).retrieve("RAG")


def test_dense_retriever_does_not_echo_embedding_or_storage_error_messages():
    """底层敏感错误文本不进入公开异常，只保留失败阶段和类型。"""
    secret = "SENSITIVE-RAW-ERROR"
    embedding_failure = DenseRetriever(
        embedding_client=FakeEmbedding(error=RuntimeError(secret)),
        vector_store=FakeVectorStore(),
    )
    with pytest.raises(RuntimeError, match="Query embedding failed: RuntimeError") as error:
        embedding_failure.retrieve("RAG")
    assert secret not in str(error.value)

    storage_failure = DenseRetriever(
        embedding_client=FakeEmbedding(),
        vector_store=FakeVectorStore(error=ValueError(secret)),
    )
    with pytest.raises(RuntimeError, match="Dense retrieval failed: ValueError") as error:
        storage_failure.retrieve("RAG")
    assert secret not in str(error.value)


def test_dense_retriever_maps_missing_text_and_metadata_to_safe_defaults():
    """存储返回缺省正文或 metadata 时统一成空值容器。"""
    store = FakeVectorStore([{"id": "bare", "score": 0.2}])

    result = DenseRetriever(embedding_client=FakeEmbedding(), vector_store=store).retrieve("RAG")

    assert result == [RetrievalResult("bare", 0.2, "", {})]


# D3：真实 Chroma、BM25 JSON 与稀疏召回


def test_base_vector_store_get_by_ids_remains_optional_for_dense_only_adapters():
    """只实现旧 dense 接口的 adapter 仍可构造，默认按 ID 读取明确未实现。"""
    class DenseOnlyStore(BaseVectorStore):
        def upsert(self, records, trace=None, **kwargs):
            pass

        def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
            return []

    store = DenseOnlyStore()

    with pytest.raises(NotImplementedError, match="does not support get_by_ids"):
        store.get_by_ids(["chunk-1"])


def test_chroma_get_by_ids_preserves_order_missing_slots_and_duplicate_ids(tmp_path):
    """真实 Chroma 按输入顺序取回，缺失 ID 占位，重复 ID 可读取。"""
    settings = _settings_for(tmp_path)
    with _open_chroma(settings, tmp_path / "chroma") as store:
        _chroma_records(
            store,
            ("a", [1.0, 0.0], "正文 A", {"source_path": "a.md", "kind": "a"}),
            ("b", [0.0, 1.0], "正文 B", {"source_path": "b.md", "kind": "b"}),
        )

        result = store.get_by_ids(["b", "missing", "a", "b"])

    assert [record.get("id") for record in result] == ["b", None, "a", "b"]
    assert result[0] == {
        "id": "b",
        "text": "正文 B",
        "metadata": {"text": "正文 B", "source_path": "b.md", "kind": "b"},
    }
    assert result[1] == {}
    assert result[2]["text"] == "正文 A"
    assert result[3] == result[0]


def test_chroma_get_by_ids_rejects_empty_ids(tmp_path):
    """空 ID 列表在调用 Chroma SDK 前被拒绝。"""
    settings = _settings_for(tmp_path)
    with _open_chroma(settings, tmp_path / "chroma") as store:
        with pytest.raises(ValueError, match="IDs list cannot be empty"):
            store.get_by_ids([])


def test_chroma_get_by_ids_reads_persisted_text_and_metadata_after_reopen(tmp_path):
    """关闭并重开本地数据库后可读正文 metadata，返回结构不含向量。"""
    settings = _settings_for(tmp_path)
    with _open_chroma(settings, tmp_path / "chroma") as writer:
        _chroma_records(
            writer,
            ("persisted", [1.0, 0.0], "重开后的正文", {"source_path": "persisted.md"}),
        )

    with _open_chroma(settings, tmp_path / "chroma") as reader:
        result = reader.get_by_ids(["persisted"])

    assert result == [
        {
            "id": "persisted",
            "text": "重开后的正文",
            "metadata": {"text": "重开后的正文", "source_path": "persisted.md"},
        }
    ]
    assert "vector" not in result[0]
    assert "embedding" not in result[0]


def test_sparse_retriever_keeps_remaining_hits_when_bm25_id_is_missing(tmp_path):
    """BM25 命中包含不存在 ID 时跳过该项，后续正文仍按 ID 正确关联。"""
    settings = _settings_for(tmp_path)
    index_dir = tmp_path / "bm25"
    collection = "notes"
    with _open_chroma(settings, tmp_path / "chroma", collection) as store:
        _chroma_records(
            store,
            ("a", [1.0, 0.0], "正文 A", {"source_path": "a.md"}),
            ("b", [0.0, 1.0], "正文 B", {"source_path": "b.md"}),
        )
        BM25Indexer(str(index_dir / collection)).build(
            _bm25_stats(
                ("a", {"needle": 1}),
                ("missing", {"needle": 1}),
                ("b", {"needle": 1}),
            ),
            collection,
        )
        retriever = create_sparse_retriever(
            settings,
            vector_store=store,
            index_dir=str(index_dir),
        )

        results = retriever.retrieve(["needle"], collection=collection)

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert [result.text for result in results] == ["正文 A", "正文 B"]
    assert results[0].metadata["source_path"] == "a.md"
    assert results[1].metadata["source_path"] == "b.md"
    assert retriever.bm25_indexer.index_dir == index_dir / collection


def test_sparse_retriever_maps_unordered_id_reads_without_zip_assumptions(tmp_path):
    """存储以不同顺序返回记录也按 chunk_id 关联文本和 metadata。"""
    class ReorderedStore(FakeVectorStore):
        def get_by_ids(self, ids, trace=None, **kwargs):
            self.get_calls.append((list(ids), trace))
            return [
                {"id": "b", "text": "正文 B", "metadata": {"source_path": "b.md"}},
                {"id": "a", "text": "正文 A", "metadata": {"source_path": "a.md"}},
            ]

    settings = _settings_for(tmp_path)
    store = ReorderedStore(collection_name="notes")
    indexer = BM25Indexer(str(tmp_path / "bm25" / "notes"))
    indexer.build(
        _bm25_stats(("a", {"needle": 1}), ("b", {"needle": 1})),
        "notes",
    )
    retriever = SparseRetriever(
        settings,
        indexer,
        store,
        default_collection="notes",
    )

    results = retriever.retrieve(["needle"], collection="notes")

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert [result.text for result in results] == ["正文 A", "正文 B"]
    assert [result.metadata["source_path"] for result in results] == ["a.md", "b.md"]


def test_sparse_retriever_preserves_real_c9_bm25_ranking_and_chroma_content(tmp_path):
    """真实 C9 词频写入 BM25 JSON 后按 BM25 排名取回 Chroma 正文与元数据。"""
    from src.core.types import Chunk
    from src.ingestion.embedding.sparse_encoder import SparseEncoder

    settings = _settings_for(tmp_path)
    collection = "notes"
    rows = [
        ("a", "needle needle needle filler"),
        ("b", "needle filler filler filler"),
        ("c", "filler filler filler filler"),
        ("d", "filler filler filler filler"),
        ("e", "filler filler filler filler"),
    ]
    chunks = [Chunk(chunk_id, text, {"source_path": f"{chunk_id}.md"}) for chunk_id, text in rows]
    stats = SparseEncoder().encode(chunks)
    indexer = BM25Indexer(str(tmp_path / "bm25" / collection))
    indexer.build(stats, collection)

    with _open_chroma(settings, tmp_path / "chroma", collection) as store:
        _chroma_records(
            store,
            *[
                (chunk_id, [1.0, 0.0], text, {"source_path": f"{chunk_id}.md"})
                for chunk_id, text in rows
            ],
        )
        retriever = create_sparse_retriever(
            settings,
            vector_store=store,
            index_dir=str(tmp_path / "bm25"),
        )
        results = retriever.retrieve(["needle"], collection=collection, top_k=5)

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert results[0].score > results[1].score > 0
    assert [result.text for result in results] == [rows[0][1], rows[1][1]]
    assert [result.metadata["source_path"] for result in results] == ["a.md", "b.md"]


def test_sparse_retriever_loads_latest_index_and_clears_missing_index(tmp_path):
    """每次查询读取最新 BM25 JSON，索引文件消失后不复用旧内存状态。"""
    settings = _settings_for(tmp_path)
    collection = "notes"
    indexer = BM25Indexer(str(tmp_path / "bm25" / collection))
    index_path = indexer._get_index_path(collection)
    indexer.build(_bm25_stats(("old", {"alpha": 1})), collection)

    class MutableStore(FakeVectorStore):
        def __init__(self):
            super().__init__(collection_name=collection)
            self.records = {
                "old": {"id": "old", "text": "旧正文", "metadata": {}},
                "new": {"id": "new", "text": "新正文", "metadata": {}},
            }

        def get_by_ids(self, ids, trace=None, **kwargs):
            return [self.records.get(identifier, {}) for identifier in ids]

    store = MutableStore()
    retriever = SparseRetriever(settings, indexer, store, default_collection=collection)

    assert [item.chunk_id for item in retriever.retrieve(["alpha"])] == ["old"]
    BM25Indexer(str(tmp_path / "bm25" / collection)).build(
        _bm25_stats(("new", {"beta": 1})), collection
    )
    assert [item.chunk_id for item in retriever.retrieve(["beta"])] == ["new"]
    index_path.unlink()
    assert retriever.retrieve(["beta"]) == []


def test_sparse_retriever_reports_corrupt_index_without_echoing_file_content(tmp_path):
    """损坏的索引抛出稳定 RuntimeError，文件内容不会进入异常。"""
    settings = _settings_for(tmp_path)
    collection = "notes"
    indexer = BM25Indexer(str(tmp_path / "bm25" / collection))
    index_path = indexer._get_index_path(collection)
    index_path.parent.mkdir(parents=True)
    index_path.write_text("SENSITIVE-INDEX-CONTENT", encoding="utf-8")
    retriever = SparseRetriever(
        settings,
        indexer,
        FakeVectorStore(collection_name=collection),
        default_collection=collection,
    )

    with pytest.raises(RuntimeError, match="Sparse retrieval failed: ValueError") as error:
        retriever.retrieve(["needle"])

    assert "SENSITIVE-INDEX-CONTENT" not in str(error.value)


def test_sparse_retriever_requires_collection_to_match_injected_store(tmp_path):
    """请求集合必须和注入的真实 Chroma 集合一致。"""
    settings = _settings_for(tmp_path, collection="notes")
    with _open_chroma(settings, tmp_path / "chroma", "notes") as store:
        retriever = create_sparse_retriever(
            settings,
            vector_store=store,
            index_dir=str(tmp_path / "bm25"),
            collection="other",
        )

        with pytest.raises(ValueError, match="collection must match"):
            retriever.retrieve(["needle"])


@pytest.mark.parametrize("keywords", [[], ["needle", 1], "needle", None])
def test_sparse_retriever_rejects_invalid_keywords(keywords, tmp_path):
    """SparseRetriever 只接受非空字符串列表。"""
    retriever = SparseRetriever(
        _settings_for(tmp_path),
        BM25Indexer(str(tmp_path / "bm25")),
        FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="keywords must be a nonempty list of strings"):
        retriever.retrieve(keywords)


def test_sparse_retriever_rejects_invalid_top_k(tmp_path):
    """稀疏召回的 TopK 与 DenseRetriever 使用相同正整数约束。"""
    retriever = SparseRetriever(
        _settings_for(tmp_path),
        BM25Indexer(str(tmp_path / "bm25")),
        FakeVectorStore(),
    )

    for top_k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            retriever.retrieve(["needle"], top_k=top_k)


def test_ingestion_dense_and_sparse_retrievers_share_real_chroma_chunk_ids(tmp_path):
    """临时 Markdown 经真实 C9/BM25/Chroma 摄取后可由两路按同一 ID 召回。"""
    from src.core.query_engine.query_processor import create_query_processor
    from src.ingestion.pipeline import IngestionPipeline

    settings = _offline_ingestion_settings(tmp_path)
    document = tmp_path / "retrieval-note.md"
    document.write_text(
        "# Retrieval Notes\n\n"
        "RAG retrieval combines dense vectors and sparse BM25 ranking.\n"
        "中文知识检索使用向量索引，也可以通过关键词定位相关内容。\n",
        encoding="utf-8",
    )
    embedding = FakeEmbedding()
    data_dir = tmp_path / "pipeline-data"
    pipeline = IngestionPipeline(
        settings,
        "notes",
        embedding=embedding,
        data_dir=data_dir,
        source_root=tmp_path,
    )

    try:
        ingested = pipeline.run(document)
        assert ingested.success, ingested.error
        assert len(ingested.vector_ids) == 1
        store = pipeline.vector_upserter.vector_store
        query = "retrieval"

        dense = DenseRetriever(settings, embedding, store).retrieve(query, top_k=5)
        sparse_query = create_query_processor().process(query)
        sparse = create_sparse_retriever(
            settings,
            vector_store=store,
            index_dir=str(data_dir / "db" / "bm25"),
        )
        sparse_results = sparse.retrieve(sparse_query.keywords, collection="notes", top_k=5)

        assert sparse.bm25_indexer.index_dir == data_dir / "db" / "bm25" / "notes"
        assert [result.chunk_id for result in dense] == ingested.vector_ids
        assert [result.chunk_id for result in sparse_results] == ingested.vector_ids
        assert dense[0].text.strip()
        assert dense[0].metadata["source_path"] == str(document)
        assert sparse_results[0].text == dense[0].text
        assert sparse_results[0].metadata["source_path"] == dense[0].metadata["source_path"]
        assert embedding.calls[-1][2] == {"task_type": "RETRIEVAL_QUERY"}
    finally:
        pipeline.close()


# D4：RRF 与结果类型


def test_rrf_fuses_dense_and_sparse_rankings_without_comparing_raw_scores():
    """RRF 示例只按名次累计，忽略各路输入分数尺度。"""
    dense = [
        RetrievalResult("A", 0.001, "正文 A"),
        RetrievalResult("B", 0.999, "正文 B"),
        RetrievalResult("C", 0.5, "正文 C"),
    ]
    sparse = [
        RetrievalResult("B", -999.0, "稀疏正文 B"),
        RetrievalResult("D", 9000.0, "正文 D"),
        RetrievalResult("A", 0.0, "稀疏正文 A"),
    ]

    results = RRFFusion(k=60).fuse([dense, sparse])

    assert [result.chunk_id for result in results] == ["B", "A", "D", "C"]
    assert [result.score for result in results] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 62, 1 / 63]
    )
    assert results[0].text == "正文 B"


def test_rrf_preserves_first_body_and_shallow_metadata_copy_without_mutation():
    """跨路重复 ID 使用首次正文，复制 metadata 顶层且不改写输入对象。"""
    nested = ["original"]
    first = RetrievalResult("shared", 100.0, "首次正文", {"nested": nested, "source": "first"})
    later = RetrievalResult("shared", -100.0, "后续正文", {"source": "later"})
    original_first = first.to_dict()
    original_later = later.to_dict()

    result = RRFFusion().fuse([[first], [later]])[0]

    assert result.text == "首次正文"
    assert result.metadata == {"nested": nested, "source": "first"}
    assert result.metadata is not first.metadata
    assert result.metadata["nested"] is first.metadata["nested"]
    assert first.to_dict() == original_first
    assert later.to_dict() == original_later
    result.metadata["new"] = "只写入输出"
    assert "new" not in first.metadata


def test_rrf_ties_are_sorted_by_chunk_id():
    """相同 RRF 分数按 chunk_id 升序稳定排序。"""
    results = RRFFusion().fuse(
        [[RetrievalResult("z", 100.0, "Z"), RetrievalResult("a", -100.0, "A")]]
    )

    assert [result.chunk_id for result in results] == ["z", "a"]
    tied = RRFFusion().fuse(
        [[RetrievalResult("z", 100.0, "Z")], [RetrievalResult("a", -100.0, "A")]]
    )
    assert [result.chunk_id for result in tied] == ["a", "z"]


def test_rrf_handles_single_empty_and_all_empty_rankings():
    """单路保持名次贡献，空路和全空路返回空列表。"""
    one = RetrievalResult("one", 3.0, "正文")

    assert RRFFusion().fuse([[one]]) == [RetrievalResult("one", 1 / 61, "正文", {})]
    assert RRFFusion().fuse([[]]) == []
    assert RRFFusion().fuse([[], []]) == []
    assert RRFFusion().fuse_with_weights([[], []], weights=[0.0, 1.0]) == []


def test_rrf_single_ranking_duplicate_id_contributes_at_each_rank():
    """沿上游每路 ID 唯一的前提，重复 ID 仍按其出现名次分别累计。"""
    duplicate = RetrievalResult("same", 2.0, "首次正文")
    later = RetrievalResult("same", 1.0, "重复正文")

    results = RRFFusion().fuse([[duplicate, later]])

    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[0].text == "首次正文"


def test_rrf_supports_weights_and_top_k():
    """非负有限权重调整各路贡献，TopK 只截断最终排序。"""
    first = [RetrievalResult("A", 1.0, "正文 A"), RetrievalResult("B", 0.0, "正文 B")]
    second = [RetrievalResult("B", -1.0, "稀疏正文 B"), RetrievalResult("C", 1.0, "正文 C")]

    results = RRFFusion(k=10).fuse_with_weights(
        [first, second], weights=[0.0, 2.0], top_k=1
    )

    assert [result.chunk_id for result in results] == ["B"]
    assert results[0].score == pytest.approx(2 / 11)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, -0.1],
        [1.0, math.nan],
        [1.0, math.inf],
        [1.0, True],
    ],
)
def test_rrf_rejects_invalid_weights(weights):
    """权重数量必须匹配且每项为有限非负数。"""
    with pytest.raises(ValueError, match="weights must be finite nonnegative"):
        RRFFusion().fuse_with_weights([[], []], weights=weights)


@pytest.mark.parametrize("top_k", [0, -1, 1.2, True])
def test_rrf_rejects_invalid_top_k(top_k):
    """融合 TopK 只能是正整数或 None。"""
    with pytest.raises(ValueError, match="positive integer or None"):
        RRFFusion().fuse([[]], top_k=top_k)


def test_rrf_rejects_empty_ranking_list_collection():
    """至少需要提供一路排名，空集合与一路空排名区分处理。"""
    with pytest.raises(ValueError, match="ranking_lists cannot be empty"):
        RRFFusion().fuse([])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_rrf_rejects_invalid_k(k):
    """平滑常数必须为正整数且不能接受 bool。"""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        RRFFusion(k=k)


@pytest.mark.parametrize("rank,k", [(1, 60), (3, 10), (2, 3)])
def test_rrf_score_helper_returns_one_rank_contribution(rank, k):
    """辅助函数只计算单一路径的名次贡献。"""
    assert rrf_score(rank, k) == pytest.approx(1 / (k + rank))


@pytest.mark.parametrize("rank,k", [(0, 60), (-1, 60), (True, 60), (1, 0), (1, True)])
def test_rrf_score_helper_rejects_invalid_rank_or_k(rank, k):
    """辅助函数拒绝非法名次或非法平滑常数。"""
    with pytest.raises(ValueError):
        rrf_score(rank, k)


@pytest.mark.parametrize("score", [-10.0, -0.25, 0.0, 1000.0])
def test_retrieval_result_accepts_finite_scores_including_negative(score):
    """统一结果允许有限负分，score 不被误认为概率。"""
    assert RetrievalResult("chunk", score, "正文").score == score


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, True])
def test_retrieval_result_rejects_nonfinite_or_boolean_score(score):
    """统一结果拒绝 NaN、无穷和 bool 分数。"""
    with pytest.raises(ValueError, match="finite number"):
        RetrievalResult("chunk", score, "正文")


@pytest.mark.parametrize("value", [
    ProcessedQuery("RAG", ["RAG"], {"doc_type": "note"}),
    RetrievalResult("chunk", -0.5, "正文", {"source_path": "note.md"}),
])
def test_retrieval_types_roundtrip_through_dict(value):
    """D1 和 D2-D4 类型均支持公开字典序列化往返。"""
    assert type(value).from_dict(value.to_dict()) == value
