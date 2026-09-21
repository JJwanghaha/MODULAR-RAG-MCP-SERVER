"""项目配置的数据结构与加载入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


class SettingsError(ValueError):
    """配置文件不存在或内容不合法。"""


def resolve_path(relative: str | Path) -> Path:
    """把项目相对路径解析为不依赖当前工作目录的绝对路径。"""
    path = Path(relative)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require_mapping(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = data.get(key)
    field_path = f"{path}.{key}"
    if value is None:
        raise SettingsError(f"Missing required field: {field_path}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {field_path}")
    return value


def _require_value(data: dict[str, Any], key: str, path: str) -> Any:
    field_path = f"{path}.{key}"
    if key not in data or data[key] is None:
        raise SettingsError(f"Missing required field: {field_path}")
    return data[key]


def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: dict[str, Any], key: str, path: str) -> list[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    """LLM 运行参数。"""

    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding 运行参数。"""

    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class VectorStoreSettings:
    """向量存储参数。"""

    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    """候选召回与融合参数。"""

    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    """重排序参数。"""

    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    """评估参数。"""

    enabled: bool
    provider: str
    metrics: list[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    """日志与链路追踪参数。"""

    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class IngestionSettings:
    """离线摄取参数。"""

    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int


@dataclass(frozen=True)
class Settings:
    """所有模块共享的类型化配置。"""

    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """把已解析的 YAML mapping 转换为 Settings。"""
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")
        ingestion = data.get("ingestion")

        ingestion_settings = None
        if ingestion is not None:
            if not isinstance(ingestion, dict):
                raise SettingsError("Expected mapping for field: settings.ingestion")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
            )

        return cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(
                    vector_store, "persist_directory", "vector_store"
                ),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(
                    observability, "structured_logging", "observability"
                ),
            ),
            ingestion=ingestion_settings,
        )


def validate_settings(settings: Settings) -> None:
    """校验构造后仍需满足的关键配置约束。"""
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")


def load_settings(path: str | Path | None = None) -> Settings:
    """从 YAML 文件加载类型化配置。"""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
