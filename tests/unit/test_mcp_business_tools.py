"""E3-E5：默认业务工具的无库、惰性与参数边界。"""

import asyncio
from pathlib import Path
import socket

import pytest

from src.core.settings import load_settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.mcp_server.server import create_default_protocol_handler


pytestmark = pytest.mark.unit


class ForbiddenEmbedding(BaseEmbedding):
    """在无需模型的业务路径中，任何编码调用都让测试失败。"""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, trace=None, **kwargs):
        self.calls += 1
        raise RuntimeError("MODEL_KEY_SENTINEL")

    def get_dimension(self):
        return 2


def _write_settings(tmp_path: Path) -> Path:
    """配置文件和空存储都限制在 pytest 临时目录。"""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """llm:
  provider: offline
  model: offline
  temperature: 0.0
  max_tokens: 32
embedding:
  provider: offline
  model: offline
  dimensions: 2
vector_store:
  provider: chroma
  persist_directory: unused-chroma
  collection_name: default
retrieval:
  dense_top_k: 4
  sparse_top_k: 4
  fusion_top_k: 4
  rrf_k: 20
rerank:
  enabled: false
  provider: none
  model: offline
  top_k: 2
evaluation:
  enabled: false
  provider: offline
  metrics: []
observability:
  log_level: ERROR
  trace_enabled: false
  trace_file: unused-trace.jsonl
  structured_logging: false
ingestion:
  chunk_size: 200
  chunk_overlap: 0
  splitter: recursive
  batch_size: 4
  chunk_refiner:
    use_llm: false
  metadata_enricher:
    use_llm: false
vision_llm:
  enabled: false
  provider: offline
  model: offline
  max_image_size: 1024
""",
        encoding="utf-8",
    )
    return config_path


def test_default_tools_handle_missing_storage_without_model_or_filesystem_side_effects(tmp_path, monkeypatch):
    """列库、空库查询和摘要缺失均走真实默认工具，且不创建数据库或 Embedding。"""
    monkeypatch.chdir(tmp_path)

    def reject_network(*args, **kwargs):
        raise AssertionError("业务工具单元测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    for name in (
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
        "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    data_dir = tmp_path / "not-created-data"
    embedding = ForbiddenEmbedding()
    handler = create_default_protocol_handler(
        load_settings(_write_settings(tmp_path)), data_dir=data_dir, embedding_client=embedding,
    )

    assert {tool.name for tool in handler.get_tool_schemas()} == {
        "query_knowledge_hub", "list_collections", "get_document_summary",
    }
    assert not data_dir.exists()

    async def exercise_tools():
        listed = await handler.execute_tool("list_collections", {})
        missing_database_query = await handler.execute_tool(
            "query_knowledge_hub", {"query": "quartz alpha", "collection": "notes"},
        )
        missing_document = await handler.execute_tool(
            "get_document_summary", {"doc_id": "doc_0000000000000000", "collection": "notes"},
        )
        schema_errors = [
            await handler.execute_tool("query_knowledge_hub", {"query": "valid", "top_k": top_k})
            for top_k in (0, 21, 1.5, True)
        ]
        blank_query = await handler.execute_tool("query_knowledge_hub", {"query": "   "})
        invalid_collection = await handler.execute_tool(
            "query_knowledge_hub", {"query": "valid", "collection": "../outside"},
        )
        unknown_tool = await handler.execute_tool("not_registered", {})
        recovered = await handler.execute_tool("list_collections", {"include_stats": False})
        return listed, missing_database_query, missing_document, schema_errors, blank_query, invalid_collection, unknown_tool, recovered

    listed, empty_query, missing_document, schema_errors, blank_query, invalid_collection, unknown_tool, recovered = asyncio.run(exercise_tools())

    assert listed.isError is False
    assert listed.structuredContent == {"collections": []}
    assert empty_query.isError is False
    assert empty_query.structuredContent["isEmpty"] is True
    assert empty_query.structuredContent["citations"] == []
    assert missing_document.isError is True
    assert all(result.isError for result in schema_errors)
    assert blank_query.isError is True
    assert invalid_collection.isError is True
    assert unknown_tool.isError is True
    assert recovered.isError is False
    assert recovered.structuredContent == {"collections": []}
    for result in (missing_document, *schema_errors, blank_query, invalid_collection, unknown_tool):
        assert "MODEL_KEY_SENTINEL" not in " ".join(item.text for item in result.content if hasattr(item, "text"))
    assert embedding.calls == 0
    assert not data_dir.exists()
