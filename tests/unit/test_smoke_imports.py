"""公开 package 的冒烟测试。"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """验证上游定义的关键 package 可以正常导入。"""

    def test_import_src_package(self) -> None:
        """`src` package 可以导入。"""
        import src

        assert src is not None

    def test_import_mcp_server(self) -> None:
        """`src.mcp_server` package 可以导入。"""
        from src import mcp_server

        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        """`src.mcp_server.tools` package 可以导入。"""
        from src.mcp_server import tools

        assert tools is not None

    def test_import_core(self) -> None:
        """`src.core` package 可以导入。"""
        from src import core

        assert core is not None

    def test_import_core_query_engine(self) -> None:
        """`src.core.query_engine` package 可以导入。"""
        from src.core import query_engine

        assert query_engine is not None

    def test_import_core_response(self) -> None:
        """`src.core.response` package 可以导入。"""
        from src.core import response

        assert response is not None

    def test_import_core_trace(self) -> None:
        """`src.core.trace` package 可以导入。"""
        from src.core import trace

        assert trace is not None

    def test_import_ingestion(self) -> None:
        """`src.ingestion` package 可以导入。"""
        from src import ingestion

        assert ingestion is not None

    def test_import_ingestion_embedding(self) -> None:
        """`src.ingestion.embedding` package 可以导入。"""
        from src.ingestion import embedding

        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        """`src.ingestion.storage` package 可以导入。"""
        from src.ingestion import storage

        assert storage is not None

    def test_import_ingestion_transform(self) -> None:
        """`src.ingestion.transform` package 可以导入。"""
        from src.ingestion import transform

        assert transform is not None

    def test_import_libs(self) -> None:
        """`src.libs` package 可以导入。"""
        from src import libs

        assert libs is not None

    def test_import_libs_embedding(self) -> None:
        """`src.libs.embedding` package 可以导入。"""
        from src.libs import embedding

        assert embedding is not None

    def test_import_libs_evaluator(self) -> None:
        """`src.libs.evaluator` package 可以导入。"""
        from src.libs import evaluator

        assert evaluator is not None

    def test_import_libs_llm(self) -> None:
        """`src.libs.llm` package 可以导入。"""
        from src.libs import llm

        assert llm is not None

    def test_import_libs_loader(self) -> None:
        """`src.libs.loader` package 可以导入。"""
        from src.libs import loader

        assert loader is not None

    def test_import_libs_reranker(self) -> None:
        """`src.libs.reranker` package 可以导入。"""
        from src.libs import reranker

        assert reranker is not None

    def test_import_libs_splitter(self) -> None:
        """`src.libs.splitter` package 可以导入。"""
        from src.libs import splitter

        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        """`src.libs.vector_store` package 可以导入。"""
        from src.libs import vector_store

        assert vector_store is not None

    def test_import_observability(self) -> None:
        """`src.observability` package 可以导入。"""
        from src import observability

        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        """`src.observability.dashboard` package 可以导入。"""
        from src.observability import dashboard

        assert dashboard is not None

    def test_import_observability_evaluation(self) -> None:
        """`src.observability.evaluation` package 可以导入。"""
        from src.observability import evaluation

        assert evaluation is not None
