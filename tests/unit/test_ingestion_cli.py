"""C15：验证摄取 CLI 的公开入口、发现规则与离线执行链路。"""

import socket
from pathlib import Path

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory

from scripts import ingest

pytestmark = pytest.mark.integration


class OfflineEmbedding(BaseEmbedding):
    """CLI 集成链路使用的固定二维向量，不调用真实 provider。"""

    def __init__(self, dimension=2, fail_marker=None):
        self.dimension = dimension
        self.fail_marker = fail_marker

    def get_dimension(self):
        return self.dimension

    def embed(self, texts, trace=None, **kwargs):
        if self.fail_marker and any(self.fail_marker in text for text in texts):
            raise RuntimeError("offline embedding failure")
        return [[float(len(text) % 7 + 1), 1.0] for text in texts]


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """CLI 的模型与存储链路必须保持本地。"""
    def fail_connect(*args, **kwargs):
        raise AssertionError("CLI 测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)


def _settings_yaml(tmp_path: Path) -> Path:
    """生成只含非敏感字段且关闭 LLM/vision 的临时配置。"""
    path = tmp_path / "settings.yaml"
    path.write_text(
        """llm:
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
  dense_top_k: 10
  sparse_top_k: 10
  fusion_top_k: 10
  rrf_k: 60
rerank:
  enabled: false
  provider: none
  model: offline
  top_k: 5
evaluation:
  enabled: false
  provider: custom
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
  provider: none
  model: offline
  max_image_size: 1024
""",
        encoding="utf-8",
    )
    return path


def _write_markdown(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _install_offline_factory(monkeypatch, calls=None, fail_marker=None):
    calls = calls if calls is not None else []

    def create(cls, settings, **kwargs):
        calls.append(settings.embedding.provider)
        return OfflineEmbedding(settings.embedding.dimensions, fail_marker)

    monkeypatch.setattr(EmbeddingFactory, "create", classmethod(create))
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["--path", "--file"])
def test_parse_args_accepts_path_alias_and_force(flag):
    """保留 --path 主入口及 --file 兼容别名。"""
    args = ingest.parse_args([flag, "notes", "--force"])
    assert args.path == "notes"
    assert args.force is True
    assert args.dry_run is False


@pytest.mark.unit
def test_discover_files_sorts_deduplicates_and_matches_case_insensitively(tmp_path):
    """目录递归结果按绝对路径排序，大小写后缀受支持且符号链接不重复。"""
    root = tmp_path / "input"
    first = _write_markdown(root / "zeta.MD", "# zeta")
    second = _write_markdown(root / "nested" / "alpha.markdown", "# alpha")
    third = root / "nested" / "middle.PDF"
    third.write_bytes(b"local PDF placeholder")
    (root / "nested" / "alias.MD").symlink_to(second)
    (root / "ignore.txt").write_text("ignored", encoding="utf-8")

    found = ingest.discover_files(root)
    assert found == sorted({first.resolve(), second.resolve(), third.resolve()})


@pytest.mark.unit
def test_discover_files_empty_directory_and_unsupported_file(tmp_path):
    """空目录返回空列表，单个不支持格式给出入口错误。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not ingested", encoding="utf-8")
    assert ingest.discover_files(empty) == []
    with pytest.raises(ValueError, match="Supported formats"):
        ingest.discover_files(unsupported)


def test_help_and_dry_run_do_not_initialize_models_or_storage_from_other_cwd(
    tmp_path, monkeypatch, capsys
):
    """help/dry-run 在任意 cwd 可用，且不创建流水线、模型或默认数据库。"""
    source_dir = tmp_path / "source"
    source = _write_markdown(source_dir / "note.MD", "# Preview\n\nPreview only.\n")
    settings_path = _settings_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)

    def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("help 和 dry-run 不应创建流水线或存储")

    monkeypatch.setattr(ingest, "IngestionPipeline", forbidden_pipeline)
    factory_calls = _install_offline_factory(monkeypatch)

    with pytest.raises(SystemExit) as help_exit:
        ingest.main(["--help"])
    assert help_exit.value.code == 0
    assert "--dry-run" in capsys.readouterr().out

    code = ingest.main(["--path", "source", "--config", settings_path.name, "--dry-run"])
    output = capsys.readouterr().out
    assert code == 0
    assert str(source.resolve()) in output
    assert "预览完成" in output
    assert factory_calls == []
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "unused-default-chroma").exists()


def test_main_empty_supported_set_is_a_successful_noop(tmp_path, monkeypatch, capsys):
    """目录没有支持格式时成功 no-op，不初始化流水线。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    settings_path = _settings_yaml(tmp_path)

    def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("空目录无需初始化流水线")

    monkeypatch.setattr(ingest, "IngestionPipeline", forbidden_pipeline)
    code = ingest.main(["--path", str(empty), "--config", str(settings_path)])
    assert code == 0
    assert "没有支持的文档" in capsys.readouterr().out


def test_cli_runs_real_pipeline_skips_duplicates_and_force_reprocesses(
    tmp_path, monkeypatch, capsys
):
    """真实 CLI 摄取使用临时配置与 Chroma；汇总区分 skip，force 再次编码。"""
    source = _write_markdown(
        tmp_path / "input" / "note.md",
        "# CLI Retrieval\n\nCommand line ingestion stores searchable evidence.\n",
    )
    settings_path = _settings_yaml(tmp_path)
    data_dir = tmp_path / "cli-data"
    factory_calls = _install_offline_factory(monkeypatch)
    args = [
        "--path", str(source), "--config", str(settings_path), "--data-dir", str(data_dir),
        "--collection", "cli_notes",
    ]

    first_code = ingest.main(args)
    first_output = capsys.readouterr().out
    assert first_code == 0
    assert "成功 1，跳过 0，失败 0" in first_output
    assert len(factory_calls) == 1

    second_code = ingest.main(args)
    second_output = capsys.readouterr().out
    assert second_code == 0
    assert "成功 0，跳过 1，失败 0" in second_output
    assert "跳过：已有成功记录" in second_output
    assert len(factory_calls) == 1

    forced_code = ingest.main(args + ["--force"])
    forced_output = capsys.readouterr().out
    assert forced_code == 0
    assert "成功 1，跳过 0，失败 0" in forced_output
    assert len(factory_calls) == 2


def test_cli_returns_distinct_codes_for_partial_and_total_failure(
    tmp_path, monkeypatch, capsys
):
    """部分失败返回 1，全失败返回 2；失败文档通过离线 embedding 触发。"""
    settings_path = _settings_yaml(tmp_path)
    _install_offline_factory(monkeypatch, fail_marker="FAIL_EMBEDDING")

    partial_dir = tmp_path / "partial"
    _write_markdown(partial_dir / "a-good.md", "# Good\n\nThis document succeeds.\n")
    _write_markdown(partial_dir / "b-fail.md", "# Failure\n\nFAIL_EMBEDDING should fail.\n")
    partial_code = ingest.main([
        "--path", str(partial_dir), "--config", str(settings_path),
        "--data-dir", str(tmp_path / "partial-data"),
    ])
    partial_output = capsys.readouterr()
    assert partial_code == 1
    assert "成功 1，跳过 0，失败 1" in partial_output.out
    assert "失败：encoding failed" in partial_output.err

    failed_dir = tmp_path / "all-failed"
    _write_markdown(failed_dir / "a-fail.md", "# Failure A\n\nFAIL_EMBEDDING one.\n")
    _write_markdown(failed_dir / "b-fail.md", "# Failure B\n\nFAIL_EMBEDDING two.\n")
    failed_code = ingest.main([
        "--path", str(failed_dir), "--config", str(settings_path),
        "--data-dir", str(tmp_path / "failed-data"),
    ])
    failed_output = capsys.readouterr()
    assert failed_code == 2
    assert "成功 0，跳过 0，失败 2" in failed_output.out
    assert "失败：encoding failed" in failed_output.err
