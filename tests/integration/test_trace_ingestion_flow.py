"""F5：通过 CLI、MCP stdio 与真实临时存储验证 Trace 生命周期。"""

import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import textwrap

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import pytest
import yaml

from scripts import ingest as ingest_cli
from scripts import query as query_cli
from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.chroma_store import ChromaStore

pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY_NAMES = {
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
}
SAFE_INHERITED_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")


class OfflineEmbedding(BaseEmbedding):
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.extend(texts)
        if self.fail:
            raise RuntimeError("offline embedding failure")
        return [[1.0, 0.0] for _ in texts]

    def get_dimension(self):
        return 2


@pytest.fixture(autouse=True)
def deny_network_and_keys(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Trace integration tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    for name in MODEL_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _settings_file(tmp_path: Path, trace_file: Path, *, trace_enabled=True) -> Path:
    assert trace_file.is_absolute()
    config = {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "offline", "dimensions": 2},
        "vector_store": {
            "provider": "chroma", "persist_directory": str(tmp_path / "unused-default-store"),
            "collection_name": "notes",
        },
        "retrieval": {"dense_top_k": 5, "sparse_top_k": 5, "fusion_top_k": 5, "rrf_k": 10},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 3},
        "evaluation": {"enabled": False, "provider": "offline", "metrics": []},
        "observability": {
            "log_level": "ERROR", "trace_enabled": trace_enabled,
            "trace_file": str(trace_file), "structured_logging": False,
        },
        "ingestion": {
            "chunk_size": 1800, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
            "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False},
        },
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 1024},
    }
    path = tmp_path / f"settings-{trace_file.stem}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_document(root: Path, name="trace-notes.md") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "diagram.png"
    from PIL import Image

    with Image.new("RGB", (20, 14), "teal") as image:
        image.save(image_path)
    source = root / name
    source.write_text(
        "---\ntitle: Trace Integration Notes\ntags: [trace, offline]\n---\n"
        "# Trace Integration Notes\n\n"
        "Quartz needle evidence verifies the CLI and MCP trace lifecycle.\n\n"
        "A second paragraph preserves the complete searchable local record.\n\n"
        "![Diagram](diagram.png)\n",
        encoding="utf-8",
    )
    return source


def _read_trace_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_completed_trace(row: dict, trace_type: str, status: str) -> None:
    assert row["trace_type"] == trace_type
    assert row["finished_at"] is not None
    assert row["total_elapsed_ms"] >= 0
    assert row["metadata"]["status"] == status


def _install_embedding(monkeypatch, embedding: BaseEmbedding) -> None:
    monkeypatch.setattr(
        EmbeddingFactory, "create", classmethod(lambda cls, settings, **kwargs: embedding),
    )


def test_ingest_and_query_cli_write_one_completed_trace_per_outcome(tmp_path, monkeypatch, capsys):
    """CLI 擁有 Trace 收集；成功、失敗與空庫均各寫一條，dry-run 不落盤。"""
    source_root = tmp_path / "documents"
    source = _write_document(source_root)
    data_dir = tmp_path / "cli-data"

    ingest_success_path = tmp_path / "ingest-success.jsonl"
    ingest_config = _settings_file(tmp_path, ingest_success_path)

    def forbidden_embedding(cls, settings, **kwargs):
        pytest.fail("CLI help and dry-run must not initialize an embedding model")

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(forbidden_embedding))
    with pytest.raises(SystemExit) as help_exit:
        ingest_cli.main(["--help"])
    assert help_exit.value.code == 0
    assert ingest_cli.main([
        "--path", str(source_root), "--config", str(ingest_config), "--dry-run",
        "--data-dir", str(data_dir),
    ]) == 0
    assert not ingest_success_path.exists()
    assert not (tmp_path / "unused-default-store").exists()

    _install_embedding(monkeypatch, OfflineEmbedding())
    ingest_code = ingest_cli.main([
        "--path", str(source_root), "--collection", "notes", "--config", str(ingest_config),
        "--data-dir", str(data_dir),
    ])
    ingest_output = capsys.readouterr().out
    assert ingest_code == 0
    ingest_rows = _read_trace_rows(ingest_success_path)
    assert len(ingest_rows) == 1
    _assert_completed_trace(ingest_rows[0], "ingestion", "success")
    assert "Trace：" not in ingest_output
    assert ingest_rows[0]["trace_id"] not in ingest_output

    ingest_error_path = tmp_path / "ingest-error.jsonl"
    ingest_error_config = _settings_file(tmp_path, ingest_error_path)
    failed_source = _write_document(tmp_path / "failed-documents", "will-fail.md")
    _install_embedding(monkeypatch, OfflineEmbedding(fail=True))
    assert ingest_cli.main([
        "--path", str(failed_source), "--collection", "notes", "--config", str(ingest_error_config),
        "--data-dir", str(tmp_path / "failed-ingest-data"),
    ]) == 2
    ingest_error_output = capsys.readouterr().out
    ingest_error_rows = _read_trace_rows(ingest_error_path)
    assert len(ingest_error_rows) == 1
    _assert_completed_trace(ingest_error_rows[0], "ingestion", "error")
    assert "Trace：" not in ingest_error_output

    _install_embedding(monkeypatch, OfflineEmbedding())
    query_success_path = tmp_path / "query-success.jsonl"
    query_success_config = _settings_file(tmp_path, query_success_path)
    query_code = query_cli.main([
        "--query", "quartz needle", "--collection", "notes", "--config", str(query_success_config),
        "--data-dir", str(data_dir),
    ])
    query_output = capsys.readouterr().out
    assert query_code == 0
    query_rows = _read_trace_rows(query_success_path)
    assert len(query_rows) == 1
    _assert_completed_trace(query_rows[0], "query", "success")
    assert "Quartz needle evidence" in query_output
    assert "Trace：" not in query_output
    assert query_rows[0]["trace_id"] not in query_output

    missing_path = tmp_path / "query-missing.jsonl"
    missing_config = _settings_file(tmp_path, missing_path)
    assert query_cli.main([
        "--query", "missing database", "--collection", "notes", "--config", str(missing_config),
        "--data-dir", str(tmp_path / "missing-query-data"),
    ]) == 0
    missing_output = capsys.readouterr().out
    missing_rows = _read_trace_rows(missing_path)
    assert len(missing_rows) == 1
    _assert_completed_trace(missing_rows[0], "query", "empty")
    assert "Trace：" not in missing_output

    bm25_path = data_dir / "db" / "bm25" / "notes" / "notes_bm25.json"
    bm25_path.write_text("not a BM25 index", encoding="utf-8")
    query_error_path = tmp_path / "query-error.jsonl"
    query_error_config = _settings_file(tmp_path, query_error_path)
    _install_embedding(monkeypatch, OfflineEmbedding(fail=True))
    assert query_cli.main([
        "--query", "quartz needle", "--collection", "notes", "--config", str(query_error_config),
        "--data-dir", str(data_dir),
    ]) == 1
    query_error_capture = capsys.readouterr()
    query_error_rows = _read_trace_rows(query_error_path)
    assert len(query_error_rows) == 1
    _assert_completed_trace(query_error_rows[0], "query", "error")
    assert "offline embedding failure" not in query_error_capture.out + query_error_capture.err
    assert "Trace：" not in query_error_capture.out

    disabled_path = tmp_path / "query-disabled.jsonl"
    disabled_config = _settings_file(tmp_path, disabled_path, trace_enabled=False)
    assert query_cli.main([
        "--query", "disabled tracing", "--collection", "notes", "--config", str(disabled_config),
        "--data-dir", str(tmp_path / "missing-disabled-data"),
    ]) == 0
    disabled_output = capsys.readouterr().out
    assert not disabled_path.exists()
    assert "Trace：" not in disabled_output
    assert not (tmp_path / "unused-default-store").exists()


def _write_mcp_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import socket

            def _deny_network(*args, **kwargs):
                raise AssertionError("network access is disabled in this child process")

            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            socket.create_connection = _deny_network
            """
        ).lstrip(),
        encoding="utf-8",
    )
    wrapper = tmp_path / "offline_trace_mcp.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            import sys
            from src.core.settings import load_settings
            from src.libs.embedding.base_embedding import BaseEmbedding
            from src.mcp_server.server import create_default_protocol_handler, run_stdio_server

            class OfflineEmbedding(BaseEmbedding):
                def embed(self, texts, trace=None, **kwargs):
                    if any("mcp forced error" in text for text in texts):
                        raise RuntimeError("offline embedding failure")
                    return [[1.0, 0.0] for _ in texts]

                def get_dimension(self):
                    return 2

            settings = load_settings(sys.argv[1])
            handler = create_default_protocol_handler(
                settings, data_dir=sys.argv[2], embedding_client=OfflineEmbedding(),
            )
            raise SystemExit(run_stdio_server(handler, log_level="ERROR"))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    environment = {key: os.environ[key] for key in SAFE_INHERITED_ENV if key in os.environ}
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    assert MODEL_KEY_NAMES.isdisjoint(environment)
    return wrapper, environment


async def _with_mcp_session(parameters, stderr_path: Path, exercise):
    with anyio.fail_after(45):
        with stderr_path.open("w", encoding="utf-8") as stderr_log:
            async with stdio_client(parameters, errlog=stderr_log) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await exercise(session)


def test_mcp_stdio_trace_id_matches_one_persisted_completed_trace_per_call(tmp_path):
    """真实 stdio 工具对成功、空集合、缺失集合和错误各收集一条 Trace。"""
    trace_path = tmp_path / "mcp-traces.jsonl"
    config_path = _settings_file(tmp_path, trace_path)
    settings = load_settings(config_path)
    data_dir = tmp_path / "mcp-data"
    source = _write_document(tmp_path / "mcp-documents")
    pipeline = IngestionPipeline(
        settings, "notes", embedding=OfflineEmbedding(), data_dir=data_dir,
        source_root=source.parent,
    )
    try:
        ingested = pipeline.run(source)
    finally:
        pipeline.close()
    assert ingested.success is True
    assert not trace_path.exists(), "IngestionPipeline.run must not collect its caller-owned trace"

    empty_store = ChromaStore(
        settings, persist_directory=data_dir / "db" / "chroma", collection_name="empty",
    )
    try:
        assert empty_store.collection.count() == 0
    finally:
        empty_store.client.close()

    wrapper, environment = _write_mcp_runtime(tmp_path)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(wrapper), str(config_path), str(data_dir)],
        env=environment,
        cwd=tmp_path,
    )
    stderr_path = tmp_path / "mcp-server.stderr"
    bm25_path = data_dir / "db" / "bm25" / "notes" / "notes_bm25.json"

    async def exercise(session):
        success = await session.call_tool("query_knowledge_hub", {
            "query": "quartz needle evidence", "collection": "notes",
        })
        empty = await session.call_tool("query_knowledge_hub", {
            "query": "empty collection probe", "collection": "empty",
        })
        missing = await session.call_tool("query_knowledge_hub", {
            "query": "missing collection probe", "collection": "not-created",
        })
        bm25_path.write_text("not a BM25 index", encoding="utf-8")
        failed = await session.call_tool("query_knowledge_hub", {
            "query": "mcp forced error", "collection": "notes",
        })
        return success, empty, missing, failed

    success, empty, missing, failed = asyncio.run(_with_mcp_session(parameters, stderr_path, exercise))

    assert success.isError is False
    assert success.structuredContent["citations"]
    assert empty.isError is False and empty.structuredContent["isEmpty"] is True
    assert missing.isError is False and missing.structuredContent["isEmpty"] is True
    assert failed.isError is True
    assert "RuntimeError" in "\n".join(item.text for item in failed.content)
    trace_ids = {
        success.structuredContent["metadata"]["trace_id"],
        empty.structuredContent["metadata"]["trace_id"],
        missing.structuredContent["metadata"]["trace_id"],
    }
    rows = _read_trace_rows(trace_path)
    assert len(rows) == 4
    assert len({row["trace_id"] for row in rows}) == 4
    assert trace_ids.issubset({row["trace_id"] for row in rows})
    expected_status = {
        "quartz needle evidence": "success",
        "empty collection probe": "empty",
        "missing collection probe": "empty",
        "mcp forced error": "error",
    }
    assert {row["metadata"]["query"]: row["metadata"]["status"] for row in rows} == expected_status
    assert all(row["finished_at"] is not None and row["total_elapsed_ms"] >= 0 for row in rows)
    error_row = next(row for row in rows if row["metadata"]["query"] == "mcp forced error")
    assert error_row["metadata"]["failed_stage"] == "hybrid_search"

    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert "Traceback" not in stderr_text
    assert "network access is disabled" not in stderr_text
    assert "offline embedding failure" not in stderr_text
    assert "API_KEY" not in stderr_text

    disabled_trace_path = tmp_path / "mcp-traces-disabled.jsonl"
    disabled_config = _settings_file(tmp_path, disabled_trace_path, trace_enabled=False)
    disabled_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(wrapper), str(disabled_config), str(data_dir)],
        env=environment,
        cwd=tmp_path,
    )
    disabled_stderr = tmp_path / "mcp-server-disabled.stderr"

    async def exercise_disabled(session):
        return await session.call_tool("query_knowledge_hub", {
            "query": "quartz needle evidence", "collection": "notes",
        })

    disabled = asyncio.run(_with_mcp_session(disabled_parameters, disabled_stderr, exercise_disabled))
    assert disabled.isError is False
    assert "trace_id" not in disabled.structuredContent["metadata"]
    assert not disabled_trace_path.exists()
    assert not (tmp_path / "unused-default-store").exists()
