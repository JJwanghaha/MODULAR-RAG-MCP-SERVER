"""D7：验证查询 CLI 参数、临时语料闭环和存储清理。"""

from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from scripts import query as query_cli
from src.core.query_engine.hybrid_search import HybridSearchResult
from src.core.settings import load_settings
from src.core.types import ProcessedQuery, RetrievalResult
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """摄取、查询和评分均只使用本地资源。"""

    def reject_network(*args, **kwargs):
        raise AssertionError("查询 CLI 测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def _settings_yaml(tmp_path: Path, *, fusion_top_k=2, rerank_enabled=True, rerank_top_k=1):
    """生成不含凭据的临时 YAML，路径和默认集合均隔离在测试目录。"""
    path = tmp_path / "settings.yaml"
    path.write_text(
        f"""llm:
  provider: offline
  model: offline
  temperature: 0.0
  max_tokens: 64
embedding:
  provider: offline
  model: offline
  dimensions: 2
vector_store:
  provider: chroma
  persist_directory: ./unused-default-chroma
  collection_name: default
retrieval:
  dense_top_k: 4
  sparse_top_k: 5
  fusion_top_k: {fusion_top_k}
  rrf_k: 17
rerank:
  enabled: {str(rerank_enabled).lower()}
  provider: llm
  model: offline
  top_k: {rerank_top_k}
evaluation:
  enabled: false
  provider: offline
  metrics: []
observability:
  log_level: INFO
  trace_enabled: false
  trace_file: ./unused-traces.jsonl
  structured_logging: false
ingestion:
  chunk_size: 1000
  chunk_overlap: 0
  splitter: recursive
  batch_size: 8
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
    return path


class OfflineEmbedding(BaseEmbedding):
    """以固定向量替代外部模型，保留所有摄取和查询输入供断言。"""

    def __init__(self):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def get_dimension(self):
        return 2


def _write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _ingest(settings_path: Path, data_dir: Path, source_path: Path, collection: str):
    """用真实 Pipeline、Chroma 与 BM25 摄取一份测试专属 Markdown。"""
    pipeline = IngestionPipeline(
        load_settings(settings_path),
        collection=collection,
        embedding=OfflineEmbedding(),
        data_dir=data_dir,
        source_root=source_path.parent,
    )
    try:
        result = pipeline.run(source_path)
    finally:
        pipeline.close()
    assert result.success is True
    assert result.chunk_count >= 1
    return result


def _persisted_ids(settings, persist_dir: Path, collection: str):
    """用真实 SDK 只读重开集合并检查已持久化记录。"""
    store = ChromaStore(
        settings,
        persist_directory=persist_dir,
        collection_name=collection,
        create_if_missing=False,
    )
    try:
        records = store.collection.get(include=["metadatas"])
        return store.collection.count(), tuple(sorted(records["ids"]))
    finally:
        store.client.close()


def test_parse_args_exposes_query_options_and_rejects_nonpositive_top_k():
    """公开参数覆盖查询、集合、配置、重排、详细输出和隔离数据目录。"""
    args = query_cli.parse_args([
        "-q", "quartz", "-c", "notes", "--top-k", "3", "--config", "settings.yaml",
        "--no-rerank", "--verbose", "--data-dir", "data-root",
    ])

    assert args.query == "quartz"
    assert args.collection == "notes"
    assert args.top_k == 3
    assert args.config == "settings.yaml"
    assert args.no_rerank is True
    assert args.verbose is True
    assert args.data_dir == "data-root"
    with pytest.raises(SystemExit) as error:
        query_cli.parse_args(["--query", "quartz", "--top-k", "0"])
    assert error.value.code == 2


def test_query_cli_ingests_then_queries_only_selected_temporary_collection(tmp_path, monkeypatch, capsys):
    """临时 Markdown 经真实双路摄取后按 CLI/问题集合选择，查询不改变索引。"""
    settings_path = _settings_yaml(tmp_path)
    original_yaml = settings_path.read_bytes()
    settings = load_settings(settings_path)
    data_dir = tmp_path / "isolated-data"
    docs_dir = tmp_path / "markdown"
    alpha_path = _write_markdown(docs_dir / "alpha.md", "AlphaQuartzEvidence: alpha collection record.")
    beta_body = "BetaQuartzEvidence: " + "B" * 240 + "BETA_TAIL_SENTINEL"
    beta_path = _write_markdown(docs_dir / "beta.md", beta_body)
    default_path = _write_markdown(docs_dir / "default.md", "DefaultQuartzEvidence: default collection record.")

    _ingest(settings_path, data_dir, alpha_path, "alpha")
    _ingest(settings_path, data_dir, beta_path, "beta")
    _ingest(settings_path, data_dir, default_path, "default")

    persist_dir = data_dir / "db" / "chroma"
    bm25_dir = data_dir / "db" / "bm25"
    before = {
        name: _persisted_ids(settings, persist_dir, name)
        for name in ("alpha", "beta", "default")
    }
    bm25_before = {
        name: (bm25_dir / name / f"{name}_bm25.json").read_bytes()
        for name in ("alpha", "beta", "default")
    }
    query_embedding = OfflineEmbedding()
    factory_calls = []

    def fake_embedding_create(cls, selected_settings, **kwargs):
        factory_calls.append(selected_settings.embedding.provider)
        return query_embedding

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(fake_embedding_create))
    real_vector_create = VectorStoreFactory.create
    opened_stores = []

    def require_existing_collection(cls, selected_settings, **kwargs):
        assert kwargs.get("create_if_missing") is False
        store = real_vector_create(selected_settings, **kwargs)
        opened_stores.append(store)
        return store

    monkeypatch.setattr(VectorStoreFactory, "create", classmethod(require_existing_collection))

    def forbidden_reranker_create(cls, selected_settings, **kwargs):
        pytest.fail("测试不得初始化真实或本地评分模型")

    monkeypatch.setattr(RerankerFactory, "create", classmethod(forbidden_reranker_create))

    common = ["--config", str(settings_path), "--data-dir", str(data_dir)]
    code = query_cli.main([
        "--query", "quartz collection:alpha", "--collection", "beta", "--no-rerank", "--verbose", *common,
    ])
    output = capsys.readouterr().out
    assert code == 0
    assert "BetaQuartzEvidence" in output
    assert "AlphaQuartzEvidence" not in output
    assert "BETA_TAIL_SENTINEL" not in output
    assert "Dense 召回" in output
    assert "BM25 召回" in output
    assert "融合/过滤结果" in output
    assert "重排后端：none" in output
    assert query_embedding.calls[-1] == "quartz collection:alpha"

    code = query_cli.main(["--query", "quartz collection:alpha", *common])
    alpha_output = capsys.readouterr().out
    assert code == 0
    assert "AlphaQuartzEvidence" in alpha_output
    assert "BetaQuartzEvidence" not in alpha_output

    code = query_cli.main(["--query", "quartz", *common])
    default_output = capsys.readouterr().out
    assert code == 0
    assert "DefaultQuartzEvidence" in default_output
    assert "AlphaQuartzEvidence" not in default_output

    after = {
        name: _persisted_ids(settings, persist_dir, name)
        for name in ("alpha", "beta", "default")
    }
    assert after == before
    assert {
        name: (bm25_dir / name / f"{name}_bm25.json").read_bytes()
        for name in ("alpha", "beta", "default")
    } == bm25_before
    assert settings_path.read_bytes() == original_yaml
    assert factory_calls == ["offline", "offline", "offline"]
    assert len(opened_stores) == 3


def test_help_missing_database_and_missing_collection_do_not_load_embedding_or_create_data(tmp_path, monkeypatch, capsys):
    """帮助和不存在的库/集合均在模型初始化前退出，且不会创建空数据。"""
    settings_path = _settings_yaml(tmp_path)
    missing_root = tmp_path / "not-created"

    def forbidden_embedding(cls, settings, **kwargs):
        pytest.fail("缺少 Chroma 库或集合时不应创建 Embedding")

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(forbidden_embedding))
    with pytest.raises(SystemExit) as help_exit:
        query_cli.main(["--help"])
    assert help_exit.value.code == 0
    assert not missing_root.exists()

    code = query_cli.main([
        "--query", "quartz", "--config", str(settings_path), "--data-dir", str(missing_root),
    ])
    assert code == 0
    assert not missing_root.exists()
    assert "未创建数据库" in capsys.readouterr().out

    settings = load_settings(settings_path)
    existing_root = tmp_path / "existing-data"
    store = ChromaStore(
        settings,
        persist_directory=existing_root / "db" / "chroma",
        collection_name="present",
    )
    store.client.close()
    code = query_cli.main([
        "--query", "quartz", "--collection", "absent", "--config", str(settings_path),
        "--data-dir", str(existing_root),
    ])
    assert code == 0
    assert "未创建数据库" in capsys.readouterr().out
    check = ChromaStore(
        settings,
        persist_directory=existing_root / "db" / "chroma",
        collection_name="present",
        create_if_missing=False,
    )
    try:
        assert check.collection.name == "present"
        collection_names = [item.name if hasattr(item, "name") else item for item in check.client.list_collections()]
        assert collection_names == [check.collection.name]
    finally:
        check.client.close()


class CloseSpy:
    """记录 CLI 对存储客户端的关闭责任。"""

    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class StoreSpy:
    """给 CLI 清理分支提供最小 client seam。"""

    def __init__(self):
        self.client = CloseSpy()


def test_build_components_closes_store_if_embedding_initialization_fails(tmp_path, monkeypatch):
    """向量库已打开后 Embedding 初始化失败，必须立即关闭已有 client。"""
    settings_path = _settings_yaml(tmp_path)
    settings = load_settings(settings_path)
    store = StoreSpy()

    def return_store(cls, selected_settings, **kwargs):
        return store

    def fail_embedding(cls, selected_settings, **kwargs):
        raise RuntimeError("private initialization detail")

    monkeypatch.setattr(VectorStoreFactory, "create", classmethod(return_store))
    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(fail_embedding))

    with pytest.raises(RuntimeError, match="private initialization detail"):
        query_cli._build_components(settings, "default", tmp_path / "bm25")

    assert store.client.closed == 1


def test_main_returns_configuration_error_and_closes_store_after_query_failure(tmp_path, monkeypatch, capsys):
    """配置错误返回 2；运行期查询错误返回 1 并在 finally 关闭 client。"""
    bad_config = tmp_path / "bad-settings.yaml"
    bad_config.write_text("embedding: {}\n", encoding="utf-8")
    assert query_cli.main(["--query", "quartz", "--config", str(bad_config)]) == 2

    settings_path = _settings_yaml(tmp_path)
    store = StoreSpy()

    class BrokenHybrid:
        config = SimpleNamespace(fusion_top_k=2)

        def search(self, *args, **kwargs):
            raise RuntimeError("private query detail")

    monkeypatch.setattr(
        query_cli,
        "_build_components",
        lambda settings, collection, index_dir: (BrokenHybrid(), object(), store),
    )
    code = query_cli.main(["--query", "quartz", "--config", str(settings_path)])

    assert code == 1
    assert store.client.closed == 1
    error_output = capsys.readouterr().err
    assert "RuntimeError" in error_output
    assert "private query detail" not in error_output


def test_main_preserves_candidate_count_and_no_rerank_override(tmp_path, monkeypatch):
    """候选数至少达到显式 TopK，默认重排时按配置裁切且 CLI 覆盖只在内存生效。"""
    settings_path = _settings_yaml(tmp_path, fusion_top_k=3, rerank_enabled=True, rerank_top_k=2)
    original_yaml = settings_path.read_bytes()
    candidates = [
        RetrievalResult(f"id-{index}", 1.0 - index / 10, f"候选正文 {index}", {"source_path": f"{index}.md"})
        for index in range(5)
    ]
    built = []

    class RecordingHybrid:
        def __init__(self):
            self.config = SimpleNamespace(fusion_top_k=3)
            self.calls = []

        def search(self, query, top_k, filters, return_details):
            self.calls.append((query, top_k, filters, return_details))
            return HybridSearchResult(
                results=list(candidates), dense_results=[], sparse_results=[],
                processed_query=ProcessedQuery(query, ["quartz"], {}),
            )

    class RecordingReranker:
        def __init__(self, enabled):
            self.config = SimpleNamespace(top_k=2)
            self.is_enabled = enabled
            self.calls = []

        def rerank(self, query, results, top_k):
            self.calls.append((query, list(results), top_k))
            return SimpleNamespace(
                results=list(results[:top_k]), used_fallback=False, reranker_type="test",
            )

    def build(settings, collection, index_dir):
        hybrid = RecordingHybrid()
        reranker = RecordingReranker(settings.rerank.enabled)
        store = StoreSpy()
        built.append((settings, hybrid, reranker, store))
        return hybrid, reranker, store

    monkeypatch.setattr(query_cli, "_build_components", build)

    assert query_cli.main([
        "--query", "quartz", "--collection", "notes", "--top-k", "5", "--config", str(settings_path),
    ]) == 0
    assert built[-1][1].calls[0][1:] == (5, {"collection": "notes"}, True)
    assert built[-1][2].calls[0][2] == 5

    assert query_cli.main(["--query", "quartz", "--config", str(settings_path)]) == 0
    assert built[-1][1].calls[0][1] == 3
    assert built[-1][2].calls[0][2] == 2

    assert query_cli.main([
        "--query", "quartz", "--config", str(settings_path), "--no-rerank",
    ]) == 0
    assert built[-1][0].rerank.enabled is False
    assert built[-1][0].rerank.provider == "none"
    assert built[-1][2].is_enabled is False
    assert built[-1][1].calls[0][1] == 3
    assert built[-1][2].calls[0][2] == 3
    assert settings_path.read_bytes() == original_yaml
    assert all(item[3].client.closed == 1 for item in built)
