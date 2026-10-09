"""C14：通过真实解析、切块与临时存储验证摄取流水线。"""

import hashlib
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.loader.file_integrity import SQLiteIntegrityChecker

pytestmark = pytest.mark.integration


class OfflineEmbedding(BaseEmbedding):
    """返回固定二维向量，不连接模型服务。"""

    def __init__(self, dimension=2, fail_once=False):
        self.dimension = dimension
        self.fail_once = fail_once
        self.calls = 0

    def get_dimension(self):
        return self.dimension

    def embed(self, texts, trace=None, **kwargs):
        self.calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("offline embedding failure")
        return [[float(len(text) % 7 + 1), 1.0] for text in texts]


@pytest.fixture
def offline_settings():
    """关闭可选文本与视觉模型，并保留真实切分器配置。"""
    settings = load_settings()
    ingestion = replace(
        settings.ingestion,
        chunk_refiner={"use_llm": False},
        metadata_enricher={"use_llm": False},
    )
    embedding = replace(settings.embedding, dimensions=2)
    return replace(settings, ingestion=ingestion, embedding=embedding)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """摄取测试的文件、图片和索引都来自本地临时目录。"""
    def fail_connect(*args, **kwargs):
        raise AssertionError("摄取测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)


def _write_markdown(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _bm25_posting_ids(index_path: Path) -> set[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        posting["chunk_id"]
        for entry in data["index"].values()
        for posting in entry["postings"]
    }


def test_real_markdown_image_and_two_documents_commit_after_all_indexes(tmp_path, offline_settings, monkeypatch):
    """真实 Markdown、jieba、Chroma、BM25 与图片登记在 success 前全部可见。"""
    from PIL import Image
    from src.ingestion.pipeline import IngestionPipeline

    documents = tmp_path / "documents"
    assets = documents / "assets"
    assets.mkdir(parents=True)
    image_source = assets / "diagram.png"
    with Image.new("RGB", (24, 16), "navy") as image:
        image.save(image_source)

    first_path = _write_markdown(
        documents / "first.md",
        "# Hybrid Retrieval\n\nDense retrieval and sparse ranking combine evidence.\n\n"
        "![流程图](assets/diagram.png)\n",
    )
    second_path = _write_markdown(
        documents / "second.md",
        "# Query Evidence\n\nSearch combines query evidence with retrieval results.\n",
    )
    data_dir = tmp_path / "pipeline-data"
    pipeline = IngestionPipeline(
        offline_settings,
        "notes",
        embedding=OfflineEmbedding(offline_settings.embedding.dimensions),
        data_dir=data_dir,
        source_root=documents,
    )

    original_mark_success = SQLiteIntegrityChecker.mark_success
    success_snapshots = []

    def check_stores_before_success(checker, file_hash, file_path, collection=None):
        assert checker.should_skip(file_hash) is False
        stored_ids = set(pipeline.vector_upserter.vector_store.collection.get(include=[])["ids"])
        posting_ids = _bm25_posting_ids(data_dir / "db" / "bm25" / "notes" / "notes_bm25.json")
        assert stored_ids == posting_ids
        rows = pipeline.image_storage.list_images(collection="notes")
        assert all(Path(row["file_path"]).is_file() for row in rows)
        success_snapshots.append((file_path, stored_ids, {row["image_id"] for row in rows}))
        return original_mark_success(checker, file_hash, file_path, collection)

    monkeypatch.setattr(SQLiteIntegrityChecker, "mark_success", check_stores_before_success)
    progress = []

    def broken_progress_callback(name, step, total):
        progress.append((name, step, total))
        raise RuntimeError("显示进度失败不应影响摄取")

    first = pipeline.run(first_path, on_progress=broken_progress_callback)
    second = pipeline.run(second_path)

    assert first.success and first.image_count == 1
    assert second.success and second.image_count == 0
    assert [step for _, step, _ in progress] == [1, 2, 3, 4, 5, 6]
    assert all(total == 6 for _, _, total in progress)
    assert [name for name, _, _ in progress] == [
        "integrity", "load", "split", "transform", "embed", "upsert",
    ]
    assert len(success_snapshots) == 2
    assert success_snapshots[0][2]
    assert len(success_snapshots[1][1]) > len(success_snapshots[0][1])

    chroma_ids = set(pipeline.vector_upserter.vector_store.collection.get(include=[])["ids"])
    bm25_ids = _bm25_posting_ids(data_dir / "db" / "bm25" / "notes" / "notes_bm25.json")
    assert chroma_ids == bm25_ids == set(first.vector_ids + second.vector_ids)
    copied_images = list((data_dir / "images" / "notes").rglob("*.png"))
    rows = pipeline.image_storage.list_images(collection="notes")
    assert len(copied_images) == len(rows) == 1
    assert copied_images[0].read_bytes() == image_source.read_bytes()
    assert Path(rows[0]["file_path"]) == copied_images[0]

    duplicate = pipeline.run(first_path)
    assert duplicate.success and duplicate.stages["integrity"]["skipped"] is True
    assert len(list((data_dir / "images" / "notes").rglob("*.png"))) == 1

    client = pipeline.vector_upserter.vector_store.client
    real_close = client.close
    close_calls = []

    def record_real_close():
        close_calls.append(True)
        return real_close()

    monkeypatch.setattr(client, "close", record_real_close)
    pipeline.close()
    pipeline.close()
    assert close_calls == [True]


def test_duplicate_skips_factory_force_reprocesses_and_hash_skip_is_global(
    tmp_path, offline_settings, monkeypatch
):
    """成功内容先跳过；force 绕过跳过规则，跨集合仍沿用全局内容哈希。"""
    from src.ingestion.pipeline import IngestionPipeline

    path = _write_markdown(tmp_path / "same.md", "# Stable note\n\nRepeated ingestion should be skipped.\n")
    data_dir = tmp_path / "data"
    first = IngestionPipeline(
        offline_settings, "collection_a", embedding=OfflineEmbedding(), data_dir=data_dir
    )
    assert first.run(path).success
    first.close()

    def model_must_not_be_created(cls, settings, **kwargs):
        raise AssertionError("未跳过成功记录时不应创建 embedding")

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(model_must_not_be_created))
    duplicate = IngestionPipeline(offline_settings, "collection_a", data_dir=data_dir)
    skipped = duplicate.run(path)
    assert skipped.success and skipped.stages["integrity"]["skipped"] is True
    duplicate.close()

    factory_calls = []

    def create_offline(cls, settings, **kwargs):
        factory_calls.append(settings.embedding.provider)
        return OfflineEmbedding(settings.embedding.dimensions)

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(create_offline))
    forced = IngestionPipeline(offline_settings, "collection_a", force=True, data_dir=data_dir)
    forced_result = forced.run(path)
    assert forced_result.success and forced_result.stages["integrity"]["skipped"] is False
    assert len(factory_calls) == 1
    forced.close()

    other_collection = IngestionPipeline(offline_settings, "collection_b", data_dir=data_dir)
    globally_skipped = other_collection.run(path)
    assert globally_skipped.success and globally_skipped.stages["integrity"]["skipped"] is True
    assert len(factory_calls) == 1
    other_collection.close()


def test_changed_source_replaces_bm25_blocks_but_keeps_old_chroma_vectors(
    tmp_path, offline_settings
):
    """来源更新替换该路径的 BM25 posting；旧 Chroma ID 保留是既定限制。"""
    from src.ingestion.pipeline import IngestionPipeline

    path = _write_markdown(tmp_path / "changing.md", "# Alpha Topic\n\nOriginal searchable material.\n")
    data_dir = tmp_path / "data"
    pipeline = IngestionPipeline(
        offline_settings, "changes", embedding=OfflineEmbedding(), data_dir=data_dir
    )
    original = pipeline.run(path)
    assert original.success

    _write_markdown(path, "# Beta Topic\n\nReplacement searchable material.\n")
    updated = pipeline.run(path)
    assert updated.success
    assert set(original.vector_ids).isdisjoint(updated.vector_ids)

    posting_ids = _bm25_posting_ids(data_dir / "db" / "bm25" / "changes" / "changes_bm25.json")
    assert posting_ids == set(updated.vector_ids)
    assert set(original.vector_ids).isdisjoint(posting_ids)
    chroma_ids = set(pipeline.vector_upserter.vector_store.collection.get(include=[])["ids"])
    assert set(original.vector_ids + updated.vector_ids) <= chroma_ids
    pipeline.close()


@pytest.mark.parametrize(
    "suffix, expected_type",
    [(".MD", "markdown"), (".markdown", "markdown"), (".PDF", "pdf")],
)
def test_pipeline_routes_supported_extensions_case_insensitively(
    tmp_path, offline_settings, suffix, expected_type
):
    """真实 Loader 按大小写无关后缀选择 Markdown 或 PDF 解析。"""
    from src.ingestion.pipeline import IngestionPipeline

    path = tmp_path / f"sample{suffix}"
    if expected_type == "pdf":
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(path))
        pdf.drawString(72, 740, "Uppercase PDF loader route evidence")
        pdf.save()
    else:
        _write_markdown(path, f"# {suffix} route\n\nCase insensitive document routing.\n")

    pipeline = IngestionPipeline(
        offline_settings,
        f"route_{expected_type}_{suffix[1:].lower()}",
        embedding=OfflineEmbedding(),
        data_dir=tmp_path / "data",
    )
    result = pipeline.run(path)
    assert result.success, result.error
    assert result.stages["loading"]["doc_type"] == expected_type
    pipeline.close()


def test_encoding_failure_is_retryable_for_same_content(tmp_path, offline_settings):
    """Embedding 的一次性失败记录为 failed，同一内容重跑可成功。"""
    from src.ingestion.pipeline import IngestionPipeline

    path = _write_markdown(tmp_path / "retry.md", "# Retry\n\nThis document can be retried.\n")
    embedding = OfflineEmbedding(fail_once=True)
    pipeline = IngestionPipeline(
        offline_settings, "retry", embedding=embedding, data_dir=tmp_path / "data"
    )

    failed = pipeline.run(path)
    assert failed.success is False
    assert failed.stages["failure"]["stage"] == "encoding"
    assert pipeline.integrity_checker.should_skip(failed.doc_id) is False

    retried = pipeline.run(path)
    assert retried.success is True
    assert pipeline.integrity_checker.should_skip(retried.doc_id) is True
    pipeline.close()


def test_parse_and_storage_failures_return_results_and_remain_retryable(
    tmp_path, offline_settings, monkeypatch
):
    """解析失败和 BM25 存储边界失败都不写 success，存储失败后可重试。"""
    from src.ingestion.pipeline import IngestionPipeline

    bad_path = tmp_path / "bad.md"
    bad_path.write_bytes(b"\xff\xfe invalid utf-8")
    parse_pipeline = IngestionPipeline(
        offline_settings, "parse", embedding=OfflineEmbedding(), data_dir=tmp_path / "parse-data"
    )
    parse_result = parse_pipeline.run(bad_path)
    assert parse_result.success is False
    assert parse_result.stages["failure"]["stage"] == "loading"
    assert parse_pipeline.integrity_checker.should_skip(parse_result.doc_id) is False
    parse_pipeline.close()

    path = _write_markdown(tmp_path / "storage.md", "# Store\n\nStorage failure can be retried.\n")
    pipeline = IngestionPipeline(
        offline_settings, "storage", embedding=OfflineEmbedding(), data_dir=tmp_path / "storage-data"
    )

    def fail_bm25_write(*args, **kwargs):
        raise OSError("injected local index write failure")

    with monkeypatch.context() as patch:
        patch.setattr(BM25Indexer, "add_documents", fail_bm25_write)
        failed = pipeline.run(path)
    assert failed.success is False
    assert failed.stages["failure"]["stage"] == "storage"
    assert pipeline.integrity_checker.should_skip(failed.doc_id) is False

    retried = pipeline.run(path)
    assert retried.success is True
    assert pipeline.integrity_checker.should_skip(retried.doc_id) is True
    pipeline.close()


def test_failure_before_hash_and_failure_history_write_do_not_escape_as_exceptions(
    tmp_path, offline_settings, monkeypatch
):
    """初始哈希前及 mark_failed 自身故障均返回有界失败结果。"""
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        offline_settings,
        "failure-history",
        embedding=OfflineEmbedding(fail_once=True),
        data_dir=tmp_path / "data",
    )
    missing = pipeline.run(tmp_path / "missing.md")
    assert missing.success is False
    assert missing.doc_id is None
    assert missing.stages["failure"]["stage"] == "integrity"
    assert "UnboundLocalError" not in (missing.error or "")

    path = _write_markdown(tmp_path / "embedding.md", "# Encode\n\nTrigger one encoding failure.\n")

    def fail_history_write(*args, **kwargs):
        raise RuntimeError("history storage unavailable")

    monkeypatch.setattr(pipeline.integrity_checker, "mark_failed", fail_history_write)
    failed = pipeline.run(path)
    assert failed.success is False
    assert failed.stages["failure"]["history_error_type"] == "RuntimeError"
    pipeline.close()
