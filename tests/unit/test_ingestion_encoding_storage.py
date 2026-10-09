"""C8-C13 摄取编码与存储契约测试；所有持久化数据均限定在临时目录。"""

import json
import math
import socket
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore


pytestmark = pytest.mark.unit


def make_chunk(identifier: str, text: str, source_path: str = "notes.md", index: int = 0) -> Chunk:
    """建立带有稳定来源字段的最小测试块。"""
    return Chunk(identifier, text, {"source_path": source_path, "chunk_index": index})


class FakeEmbedding(BaseEmbedding):
    """记录批次和 trace 的离线 Embedding 替身。"""

    def __init__(self, dimension: int = 2, failure_text: str | None = None, malformed: str | None = None):
        self.dimension = dimension
        self.failure_text = failure_text
        self.malformed = malformed
        self.calls: list[tuple[list[str], object]] = []

    def embed(self, texts: list[str], trace=None, **kwargs):
        self.calls.append((list(texts), trace))
        if self.failure_text is not None and self.failure_text in texts:
            raise OSError("synthetic offline failure")
        vectors = [[float(len(text)), float(sum(map(ord, text)))] for text in texts]
        if self.malformed == "count":
            return vectors[:-1]
        if self.malformed == "dimension":
            return [[0.0] for _ in texts]
        return vectors

    def get_dimension(self) -> int:
        return self.dimension


def test_dense_encoder_preserves_order_batches_and_trace():
    """稠密向量严格按输入顺序分批，且每批收到同一个 trace 对象。"""
    chunks = [make_chunk(f"c{i}", text) for i, text in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])]
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding, batch_size=2)
    trace = {"request": "trace-1"}

    vectors = encoder.encode(chunks, trace=trace)

    assert embedding.calls == [(["alpha", "beta"], trace), (["gamma", "delta"], trace), (["epsilon"], trace)]
    assert vectors == [[float(len(chunk.text)), float(sum(map(ord, chunk.text)))] for chunk in chunks]
    assert encoder.get_batch_count(5) == 3
    assert encoder.get_batch_count(0) == 0


@pytest.mark.parametrize("malformed", ["count", "dimension"])
def test_dense_encoder_rejects_invalid_model_output(malformed):
    """模型返回数量或维度不符时，编码器报告失败而不接收结果。"""
    encoder = DenseEncoder(FakeEmbedding(malformed=malformed), batch_size=2)

    with pytest.raises(RuntimeError, match="Dense encoding failed for batch starting at 0: ValueError"):
        encoder.encode([make_chunk("a", "alpha"), make_chunk("b", "beta")])


def test_dense_encoder_validates_input_and_stops_after_failed_batch():
    """空白输入在调用前拒绝，后续批次失败时不再调用模型。"""
    empty_embedding = FakeEmbedding()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        DenseEncoder(empty_embedding).encode([make_chunk("blank", " \n")])
    assert empty_embedding.calls == []

    failing_embedding = FakeEmbedding(failure_text="gamma")
    with pytest.raises(RuntimeError, match="batch starting at 2: OSError"):
        DenseEncoder(failing_embedding, batch_size=2).encode(
            [make_chunk("a", "alpha"), make_chunk("b", "beta"), make_chunk("c", "gamma"), make_chunk("d", "delta")]
        )
    assert [call[0] for call in failing_embedding.calls] == [["alpha", "beta"], ["gamma", "delta"]]


@pytest.fixture
def sparse_encoder(tmp_path, monkeypatch):
    """把 jieba 的词典缓存也放进本测试临时目录。"""
    import jieba

    monkeypatch.setattr(jieba.dt, "tmp_dir", str(tmp_path))
    return SparseEncoder(min_term_length=2, lowercase=True)


def test_sparse_encoder_uses_real_jieba_filters_and_counts_document_frequency(sparse_encoder):
    """真实 jieba 分词保留中英文重复词，词频与跨块文档频率分开统计。"""
    encoded = sparse_encoder.encode([
        make_chunk("first", "RAG rag, 中文 中文！ AI x"),
        make_chunk("second", "rag, 中文"),
    ])

    assert encoded[0]["term_frequencies"] == {"rag": 2, "中文": 2, "ai": 1}
    assert encoded[0]["doc_length"] == 5
    assert encoded[0]["unique_terms"] == 3
    assert encoded[1]["term_frequencies"] == {"rag": 1, "中文": 1}
    assert sparse_encoder.get_corpus_stats(encoded) == {
        "num_docs": 2,
        "avg_doc_length": 3.5,
        "document_frequency": {"rag": 2, "中文": 2, "ai": 1},
    }


def test_sparse_encoder_handles_zero_terms_and_rejects_empty_or_blank(sparse_encoder):
    """纯单字符与标点形成零词项块；空列表和空白块仍按接口拒绝。"""
    zero_terms = sparse_encoder.encode([make_chunk("zero", "a ! x")])[0]
    assert zero_terms == {"chunk_id": "zero", "term_frequencies": {}, "doc_length": 0, "unique_terms": 0}
    assert sparse_encoder.get_corpus_stats([zero_terms]) == {
        "num_docs": 1, "avg_doc_length": 0.0, "document_frequency": {},
    }
    with pytest.raises(ValueError, match="empty chunks list"):
        sparse_encoder.encode([])
    with pytest.raises(ValueError, match="blank chunk text"):
        sparse_encoder.encode([make_chunk("blank", "  ")])


def test_batch_processor_keeps_dense_and_sparse_results_aligned(sparse_encoder):
    """组合编码按原顺序对齐块 ID、稠密向量和稀疏统计。"""
    chunks = [make_chunk("c0", "alpha 中文"), make_chunk("c1", "beta 中文"), make_chunk("c2", "gamma 中文")]
    embedding = FakeEmbedding()
    processor = BatchProcessor(DenseEncoder(embedding, batch_size=10), sparse_encoder, batch_size=2)
    trace = object()

    result = processor.process(chunks, trace=trace)

    assert result.dense_vectors == [[float(len(chunk.text)), float(sum(map(ord, chunk.text)))] for chunk in chunks]
    assert [stat["chunk_id"] for stat in result.sparse_stats] == [chunk.id for chunk in chunks]
    assert result.batch_count == 2
    assert result.successful_chunks == 3
    assert result.failed_chunks == 0
    assert [call[0] for call in embedding.calls] == [["alpha 中文", "beta 中文"], ["gamma 中文"]]
    assert all(call[1] is trace for call in embedding.calls)


def test_batch_processor_raises_on_failed_or_misaligned_batch():
    """第二批失败或稀疏结果乱序时整次调用抛错，不返回首批的部分结果。"""
    chunks = [make_chunk(f"c{i}", text) for i, text in enumerate(["alpha", "beta", "gamma", "delta"])]

    class SparseFailure:
        def __init__(self):
            self.calls = []

        def encode(self, batch, trace=None):
            self.calls.append([chunk.id for chunk in batch])
            if batch[0].id == "c2":
                raise LookupError("synthetic sparse failure")
            return [{"chunk_id": chunk.id} for chunk in batch]

    failure = SparseFailure()
    processor = BatchProcessor(DenseEncoder(FakeEmbedding(), batch_size=8), failure, batch_size=2)
    with pytest.raises(RuntimeError, match="Encoding batch 1 failed: LookupError"):
        processor.process(chunks)
    assert failure.calls == [["c0", "c1"], ["c2", "c3"]]

    class ReorderedSparse:
        def encode(self, batch, trace=None):
            return [{"chunk_id": chunk.id} for chunk in reversed(batch)]

    processor = BatchProcessor(DenseEncoder(FakeEmbedding(), batch_size=8), ReorderedSparse(), batch_size=2)
    with pytest.raises(RuntimeError, match="Encoding batch 0 failed: ValueError"):
        processor.process(chunks[:2])


def bm25_stat(chunk_id: str, frequencies: dict[str, int], doc_length: int) -> dict:
    """构造与实际分词器输出相同形状的 BM25 统计。"""
    return {"chunk_id": chunk_id, "term_frequencies": frequencies, "doc_length": doc_length}


def test_bm25_json_roundtrip_preserves_formula_order_and_zero_term_chunks(tmp_path):
    """BM25 写入真实 JSON 后可重载；负 IDF、TopK 顺序和零词项块均保留。"""
    stats = [
        bm25_stat("doc-a-0", {"common": 3, "rare": 1}, 4),
        bm25_stat("doc-b-0", {"common": 1}, 1),
        bm25_stat("doc-c-0", {}, 0),
    ]
    writer = BM25Indexer(str(tmp_path / "bm25"))
    writer.build(stats, collection="notes")

    index_path = tmp_path / "bm25" / "notes_bm25.json"
    saved = json.loads(index_path.read_text(encoding="utf-8"))
    assert saved["metadata"]["num_docs"] == 3
    assert "doc-c-0" in saved["documents"]
    assert saved["index"]["common"]["df"] == 2
    assert saved["index"]["common"]["idf"] == pytest.approx(math.log((3 - 2 + 0.5) / (2 + 0.5)))
    assert saved["index"]["common"]["idf"] < 0

    reader = BM25Indexer(str(tmp_path / "bm25"))
    assert reader.load("notes") is True
    results = reader.query(["common"], top_k=2)
    assert [item["chunk_id"] for item in results] == ["doc-b-0", "doc-a-0"]
    assert results[0]["score"] < 0
    assert results[1]["score"] < results[0]["score"]
    assert "doc-c-0" in reader._documents


def test_bm25_add_documents_overwrites_ids_and_replaces_document_prefix(tmp_path):
    """同 ID 加入覆盖旧统计，doc_id 前缀模式移除该文档原有的其余块。"""
    indexer = BM25Indexer(str(tmp_path / "bm25"))
    indexer.build([
        bm25_stat("doc-a-0", {"old": 1}, 1),
        bm25_stat("doc-a-1", {"remove": 1}, 1),
        bm25_stat("doc-b-0", {"keep": 1}, 1),
    ], collection="replace")

    indexer.add_documents([bm25_stat("doc-a-0", {"updated": 2}, 2)], collection="replace")
    assert set(indexer._documents) == {"doc-a-0", "doc-a-1", "doc-b-0"}
    assert indexer._documents["doc-a-0"]["term_frequencies"] == {"updated": 2}

    indexer.add_documents([bm25_stat("doc-a-0", {"final": 1}, 1)], collection="replace", doc_id="doc-a")
    saved = json.loads((tmp_path / "bm25" / "replace_bm25.json").read_text(encoding="utf-8"))
    assert set(saved["documents"]) == {"doc-a-0", "doc-b-0"}
    assert saved["documents"]["doc-a-0"]["term_frequencies"] == {"final": 1}
    assert indexer.query(["remove"], top_k=5) == []


def test_bm25_collection_switch_clears_state_and_keeps_zero_idf_result(tmp_path):
    """加载缺失集合不沿用上个集合；零 IDF 命中仍返回零分结果。"""
    indexer = BM25Indexer(str(tmp_path / "bm25"))
    indexer.build([
        bm25_stat("hit", {"balanced": 1}, 1),
        bm25_stat("empty", {}, 0),
    ], collection="present")
    assert indexer.query(["balanced"])[0]["score"] == pytest.approx(0.0)

    assert indexer.load("missing") is False
    with pytest.raises(ValueError, match=r"Call load\(\) or build\(\)"):
        indexer.query(["balanced"])


class RecordingVectorStore(BaseVectorStore):
    """只记录调用的本地存储替身，用于验证批次前置校验。"""

    def __init__(self):
        self.upsert_calls = []

    def upsert(self, records, trace=None, **kwargs):
        self.upsert_calls.append(records)

    def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
        return []


def test_vector_upserter_stable_ids_idempotence_and_chroma_restart(tmp_path, monkeypatch):
    """真实临时 Chroma 写入、查询、重开后保持稳定 ID 与元数据。"""
    def deny_network(*args, **kwargs):
        raise AssertionError("Chroma test must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    settings = load_settings()
    settings = replace(
        settings,
        vector_store=replace(settings.vector_store, persist_directory=str(tmp_path / "chroma")),
    )
    chunk = make_chunk("temporary-source-id", "persistent local text", "fixture.md", 7)
    expected_id = (
        sha256(b"fixture.md").hexdigest()[:8]
        + "_0007_"
        + sha256(b"persistent local text").hexdigest()[:8]
    )

    first = VectorUpserter(settings, collection_name="ingestion_contract")
    first_ids = first.upsert([chunk], [[1.0, 0.0]])
    assert first.upsert([chunk], [[1.0, 0.0]]) == first_ids == [expected_id]
    result = first.vector_store.query([1.0, 0.0], top_k=2)
    assert len(result) == 1
    assert result[0]["id"] == expected_id
    assert result[0]["text"] == chunk.text
    assert result[0]["metadata"]["text"] == chunk.text
    assert result[0]["metadata"]["chunk_id"] == expected_id
    first.vector_store.client.close()

    reopened = VectorUpserter(settings, collection_name="ingestion_contract")
    reloaded = reopened.vector_store.query([1.0, 0.0], top_k=2)
    assert len(reloaded) == 1
    assert reloaded[0]["id"] == expected_id
    assert reloaded[0]["metadata"]["chunk_id"] == expected_id
    reopened.vector_store.client.close()


def test_vector_upserter_checks_each_batch_before_flattening():
    """单批多向量与另一批少向量即使总数相等也必须在写入前拒绝。"""
    store = RecordingVectorStore()
    upserter = VectorUpserter(load_settings(), vector_store=store)
    batches = [
        ([make_chunk("a", "alpha")], [[1.0, 0.0], [0.0, 1.0]]),
        ([make_chunk("b", "beta")], []),
    ]

    with pytest.raises(ValueError, match="Each batch must have matching"):
        upserter.upsert_batch(batches)
    assert store.upsert_calls == []


def test_image_storage_copies_raw_bytes_and_files_and_survives_reopen(tmp_path):
    """SQLite 登记、原样字节保存、路径复制、筛选和重开查询均可用临时数据完成。"""
    db_path = tmp_path / "image-index.sqlite"
    images_root = tmp_path / "images"
    storage = ImageStorage(str(db_path), str(images_root))
    raw = b"not-transcoded-image-bytes\x00\xff"
    byte_path = storage.save_image("image-bytes", raw, "collection-a", "doc-a", 3, "png")

    source = tmp_path / "source-image.dat"
    source.write_bytes(b"copied source bytes\x00\xfe")
    copied_path = storage.save_image("image-copy", source, "collection-a", "doc-a", 4, "dat")
    other_path = storage.save_image("image-other", b"other", "collection-b", "doc-b", extension="bin")

    assert Path(byte_path).read_bytes() == raw
    assert Path(copied_path).read_bytes() == source.read_bytes()
    assert Path(other_path).read_bytes() == b"other"
    assert {row["image_id"] for row in storage.list_images(collection="collection-a", doc_hash="doc-a")} == {
        "image-bytes", "image-copy",
    }
    assert [row["image_id"] for row in storage.list_images(collection="collection-b")] == ["image-other"]

    reopened = ImageStorage(str(db_path), str(images_root))
    assert reopened.image_exists("image-bytes") is True
    assert reopened.get_image_path("image-copy") == copied_path
    assert reopened.list_images(doc_hash="doc-b")[0]["collection"] == "collection-b"
    Path(byte_path).unlink()
    assert reopened.image_exists("image-bytes") is True


@pytest.mark.parametrize("image_id,collection,extension", [
    ("../escape", "collection", "png"),
    ("image", "../escape", "png"),
    ("image", "collection", "../png"),
])
def test_image_storage_rejects_path_components(tmp_path, image_id, collection, extension):
    """图片 ID、集合名和后缀都不能越过图片根目录。"""
    storage = ImageStorage(str(tmp_path / "index.sqlite"), str(tmp_path / "images"))

    with pytest.raises(ValueError, match="single names"):
        storage.save_image(image_id, b"fixture", collection=collection, extension=extension)

    assert not (tmp_path / "images").exists()
