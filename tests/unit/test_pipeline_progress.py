"""F4：通过 IngestionPipeline.run 验证 Trace 所有权和进度通知。"""

import json
from dataclasses import replace
from pathlib import Path
import socket

import pytest
from PIL import Image

from src.core.settings import load_settings
from src.core.trace import TraceContext
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.loader.file_integrity import SQLiteIntegrityChecker

pytestmark = pytest.mark.unit


class OfflineEmbedding(BaseEmbedding):
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.extend(texts)
        if self.fail:
            raise RuntimeError("offline embedding failure")
        return [[float(len(text) % 11 + 1), 1.0] for text in texts]

    def get_dimension(self):
        return 2


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Pipeline Trace tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def _settings(tmp_path):
    settings = load_settings()
    ingestion = replace(
        settings.ingestion,
        chunk_size=1800,
        chunk_overlap=0,
        chunk_refiner={"use_llm": False},
        metadata_enricher={"use_llm": False},
    )
    embedding = replace(settings.embedding, dimensions=2)
    observability = replace(
        settings.observability, trace_enabled=True,
        trace_file=str(tmp_path / "caller-owned-traces.jsonl"),
    )
    return replace(settings, ingestion=ingestion, embedding=embedding, observability=observability)


def _write_source(tmp_path, name="handbook.md"):
    source_root = tmp_path / "documents"
    assets = source_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (18, 12), "navy") as image:
        image.save(assets / "diagram.png")
    body = (
        "Quartz quartz quartz evidence evidence records the offline retrieval handbook.\n\n"
        "The final passage keeps every sentence available in the trace snapshot.\n"
    )
    path = source_root / name
    path.write_text(
        "---\ntitle: Traceable Handbook\ntags: [trace, offline]\n---\n"
        "# Traceable Handbook\n\n" + body + "\n![Diagram](assets/diagram.png)\n",
        encoding="utf-8",
    )
    return path, source_root, body


def _pipeline(tmp_path, source_root, embedding=None):
    return IngestionPipeline(
        _settings(tmp_path), "notes", embedding=embedding or OfflineEmbedding(),
        data_dir=tmp_path / "pipeline-data", source_root=source_root,
    )


def test_pipeline_records_full_offline_ingestion_trace_without_finishing_or_collecting(tmp_path):
    """成功 Trace 包含正文、变换前后、存储映射和词频，但 Pipeline 不替调用方落盘。"""
    path, source_root, body = _write_source(tmp_path)
    pipeline = _pipeline(tmp_path, source_root)
    trace = TraceContext(trace_type="ingestion")
    progress = []

    def failing_display_callback(stage, step, total):
        progress.append((stage, step, total))
        raise RuntimeError("display callback failure")

    try:
        result = pipeline.run(path, trace=trace, on_progress=failing_display_callback)

        assert result.success is True
        assert result.image_count == 1
        assert trace.finished_at is None
        assert trace.metadata["status"] == "success"
        assert trace.metadata["source_path"] == str(path.resolve())
        assert trace.metadata["collection"] == "notes"
        assert trace.metadata["doc_id"] == result.doc_id
        assert trace.metadata["chunk_count"] == result.chunk_count
        assert trace.metadata["image_count"] == 1
        assert [step for _, step, _ in progress] == [1, 2, 3, 4, 5, 6]
        assert [name for name, _, _ in progress] == [
            "integrity", "load", "split", "transform", "embed", "upsert",
        ]
        assert all(total == 6 for _, _, total in progress)

        load_stage = trace.get_stage_data("load")
        assert load_stage["method"] == "markdown"
        assert load_stage["status"] == "success"
        assert load_stage["doc_type"] == "markdown"
        assert "Quartz quartz quartz evidence evidence" in load_stage["text_preview"]
        assert "The final passage keeps every sentence available in the trace snapshot." in load_stage["text_preview"]
        assert load_stage["image_count"] == 1

        split_stage = trace.get_stage_data("split")
        transform_stage = trace.get_stage_data("transform")
        before_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in split_stage["chunks"]}
        transformed_by_id = {chunk["chunk_id"]: chunk for chunk in transform_stage["chunks"]}
        assert set(transformed_by_id) == set(before_by_id)
        assert body.strip() in "\n".join(before_by_id.values())
        for chunk_id, chunk in transformed_by_id.items():
            assert chunk["text_before"] == before_by_id[chunk_id]
            assert chunk["text_after"] == before_by_id[chunk_id]
        assert transform_stage["status"] == "success"

        embed_stage = trace.get_stage_data("embed")
        assert embed_stage["status"] == "success"
        assert embed_stage["dense_vector_count"] == result.chunk_count
        assert embed_stage["sparse_doc_count"] == result.chunk_count
        top_terms = embed_stage["chunks"][0]["top_terms"]
        assert top_terms
        assert all(set(term) == {"term", "freq"} for term in top_terms)
        quartz = next(term for term in top_terms if term["term"] == "quartz")
        assert quartz["freq"] == 3
        assert all("vector" not in chunk and "dense_vector" not in chunk for chunk in embed_stage["chunks"])

        upsert_stage = trace.get_stage_data("upsert")
        assert upsert_stage["status"] == "success"
        mapping = upsert_stage["chunk_mapping"]
        assert [row["vector_id"] for row in mapping] == result.vector_ids
        assert len(mapping) == result.chunk_count
        assert all(row["chunk_id"] and row["vector_id"] for row in mapping)
        assert upsert_stage["vector_count"] == upsert_stage["bm25_docs"] == result.chunk_count
        assert len(upsert_stage["images"]) == 1
        assert trace.get_stage_data("completion")["status"] == "success"

        stored = pipeline.vector_upserter.vector_store.collection.get(
            ids=result.vector_ids, include=["documents", "metadatas"],
        )
        assert stored["ids"] == result.vector_ids
        assert all("Quartz quartz quartz evidence evidence" in text for text in stored["documents"])
        assert all(metadata["source_path"] == str(path.resolve()) for metadata in stored["metadatas"])
        assert all(metadata["title"] == "Traceable Handbook" for metadata in stored["metadatas"])

        serialized = json.dumps(trace.to_dict(), ensure_ascii=False)
        assert '"dense_vector":' not in serialized
        assert '"vector":' not in serialized
        assert not Path(_settings(tmp_path).observability.trace_file).exists()
    finally:
        pipeline.close()


def test_pipeline_skip_notifies_only_integrity_and_trace_none_never_collects(tmp_path):
    """重复文件只产生一次跳过通知；无传入 Trace 时不创建追踪文件。"""
    path, source_root, _ = _write_source(tmp_path)
    pipeline = _pipeline(tmp_path, source_root)
    try:
        assert pipeline.run(path).success is True
        trace = TraceContext(trace_type="ingestion")
        progress = []
        duplicate = pipeline.run(path, trace=trace, on_progress=lambda *args: progress.append(args))
        untraced = pipeline.run(path)

        assert duplicate.success is True
        assert duplicate.stages["integrity"]["skipped"] is True
        assert progress == [("integrity", 1, 6)]
        assert trace.metadata["status"] == "skipped"
        assert [stage["stage"] for stage in trace.stages] == ["integrity"]
        assert trace.finished_at is None
        assert untraced.success is True
        assert not Path(_settings(tmp_path).observability.trace_file).exists()
    finally:
        pipeline.close()


def test_pipeline_failure_never_notifies_six_of_six_when_success_record_fails(tmp_path, monkeypatch):
    """mark_success 写入失败记为 completion error，进度不会显示 6/6。"""
    path, source_root, _ = _write_source(tmp_path)
    pipeline = _pipeline(tmp_path, source_root)
    trace = TraceContext(trace_type="ingestion")
    progress = []

    def fail_mark_success(*args, **kwargs):
        raise OSError("history store unavailable")

    monkeypatch.setattr(SQLiteIntegrityChecker, "mark_success", fail_mark_success)
    try:
        result = pipeline.run(path, trace=trace, on_progress=lambda *args: progress.append(args))

        assert result.success is False
        assert result.error == "completion failed: OSError"
        assert [step for _, step, _ in progress] == [1, 2, 3, 4, 5]
        assert all(step != 6 for _, step, _ in progress)
        assert trace.metadata["status"] == "error"
        assert trace.metadata["failed_stage"] == "completion"
        assert trace.get_stage_data("upsert")["status"] == "success"
        assert trace.get_stage_data("error")["method"] == "completion"
        assert trace.get_stage_data("error")["error_type"] == "OSError"
        assert not any(stage["stage"] == "completion" for stage in trace.stages)
        assert trace.finished_at is None
    finally:
        pipeline.close()
