"""Settings 配置加载的公开行为测试。"""

from pathlib import Path
import textwrap

import pytest

from src.core.settings import SettingsError, load_settings


def _write_yaml(path: Path, content: str) -> None:
    """把测试配置写入独立的临时文件。"""
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


@pytest.mark.unit
def test_load_settings_success(tmp_path: Path) -> None:
    """完整 YAML 可以加载为类型化的 Settings。"""
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
        - mrr
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    ingestion:
      chunk_size: 1000
      chunk_overlap: 200
      splitter: recursive
      batch_size: 100
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.llm.provider == "openai"
    assert settings.embedding.dimensions == 1536
    assert settings.vector_store.collection_name == "knowledge_hub"
    assert settings.retrieval.rrf_k == 60
    assert settings.rerank.provider == "none"
    assert settings.evaluation.metrics == ["hit_rate", "mrr"]
    assert settings.observability.log_level == "INFO"
    assert settings.ingestion is not None


@pytest.mark.unit
def test_missing_required_field_raises_readable_error(tmp_path: Path) -> None:
    """缺少必填字段时，错误信息包含完整字段路径。"""
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_path)


@pytest.mark.unit
def test_default_settings_file_loads() -> None:
    """仓库默认配置可以从任意调用入口加载。"""
    settings = load_settings()

    assert settings.vector_store.collection_name == "knowledge_hub"


def test_model_connection_options_survive_loading(tmp_path):
    """网关地址和 Azure 连接参数是可选非敏感字段，不在配置中保存密钥。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data["llm"].update(base_url="https://llm.example/v1", deployment_name="chat-deployment", api_version="test-version", timeout=15.0)
    data["embedding"].update(base_url="https://embedding.example/v1", deployment_name="embed-deployment", api_version="test-version", timeout=20.0)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_settings(path)
    assert settings.llm.base_url == "https://llm.example/v1"
    assert settings.llm.deployment_name == "chat-deployment"
    assert settings.llm.api_version == "test-version"
    assert settings.llm.timeout == 15.0
    assert settings.embedding.base_url == "https://embedding.example/v1"
    assert settings.embedding.deployment_name == "embed-deployment"
    assert settings.embedding.api_version == "test-version"
    assert settings.embedding.timeout == 20.0


@pytest.mark.parametrize("section,field,value", [
    ("llm", "timeout", 0), ("embedding", "timeout", -1),
    ("embedding", "dimensions", 0), ("embedding", "dimensions", True),
])
def test_model_connection_config_rejects_invalid_limits(tmp_path, section, field, value):
    """超时和维度必须具有实际运行意义。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data[section][field] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SettingsError, match=f"{section}.{field}"):
        load_settings(path)
