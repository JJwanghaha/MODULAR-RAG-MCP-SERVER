"""F1/F2：通过公开接口验证 Trace 基础数据、采集与 JSONL 日志行为。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import logging
import math
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

import pytest

from src.core.settings import ObservabilitySettings
from src.core.trace import TraceCollector, TraceContext
from src.core.trace import trace_context as trace_context_module
from src.observability import logger as logger_module
from src.observability.logger import JSONFormatter, get_logger, get_trace_logger, write_trace


pytestmark = pytest.mark.unit


def _utc_datetime(value: str) -> datetime:
    """把 ISO 时间解析为带时区的 datetime，供 UTC 断言复用。"""
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() == timedelta(0)
    return parsed


@pytest.fixture
def monotonic_clock(monkeypatch):
    """用可推进的单调时钟验证总耗时和阶段耗时彼此独立。"""
    clock = {"value": 100.0}
    monkeypatch.setattr(trace_context_module.time, "monotonic", lambda: clock["value"])
    return clock


@pytest.fixture
def isolated_logger_name():
    """为每个用例分配独立 logger，并在结束时只清理本用例新增的 Handler。"""
    created = []

    def new_name():
        name = f"test.trace.{uuid.uuid4().hex}"
        logger = logging.getLogger(name)
        created.append((logger, tuple(logger.handlers), logger.level, logger.propagate))
        return name

    yield new_name
    for logger, original_handlers, original_level, original_propagate in created:
        for handler in tuple(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_trace_context_exposes_uuid_utc_timestamps_and_upstream_fields():
    """默认 Trace 使用 UUID 与 UTC ISO 时间，并保留上游顶层字段。"""
    trace = TraceContext()
    payload = trace.to_dict()

    assert str(uuid.UUID(trace.trace_id)) == trace.trace_id
    assert trace.trace_type == "query"
    assert _utc_datetime(trace.started_at).tzinfo is not None
    assert trace.finished_at is None
    assert set(payload) == {
        "trace_id", "trace_type", "started_at", "finished_at", "total_elapsed_ms", "stages", "metadata",
    }
    assert payload["trace_id"] == trace.trace_id
    assert payload["finished_at"] is None
    assert payload["stages"] == []
    assert payload["metadata"] == {}
    assert "status" not in payload


def test_trace_context_accepts_ingestion_and_rejects_unknown_trace_type():
    """摄取 Trace 可显式创建，未支持的类型在构造时被拒绝。"""
    assert TraceContext(trace_type="ingestion").trace_type == "ingestion"
    with pytest.raises(ValueError, match="trace_type"):
        TraceContext(trace_type="maintenance")


def test_total_elapsed_time_is_independent_from_parallel_stage_durations(monotonic_clock):
    """总耗时按单调时钟测量，不把可并行阶段的耗时相加。"""
    trace = TraceContext()
    trace.record_stage("dense_search", {"count": 4}, elapsed_ms=80)
    trace.record_stage("sparse_search", {"count": 3}, elapsed_ms=120)
    monotonic_clock["value"] = 100.25

    assert trace.elapsed_ms("dense_search") == 80
    assert trace.elapsed_ms("sparse_search") == 120
    assert trace.elapsed_ms() == pytest.approx(250)
    assert trace.to_dict()["total_elapsed_ms"] == 250


def test_finish_is_idempotent_and_freezes_total_elapsed_time(monotonic_clock):
    """首次 finish 固定结束时间和总耗时，后续调用不再推进结束点。"""
    trace = TraceContext()
    monotonic_clock["value"] = 100.125
    trace.finish()
    finished_at = trace.finished_at
    monotonic_clock["value"] = 101.0
    trace.finish()

    assert finished_at is not None
    assert _utc_datetime(finished_at).tzinfo is not None
    assert trace.finished_at == finished_at
    assert trace.elapsed_ms() == pytest.approx(125)
    assert trace.to_dict()["finished_at"] == finished_at


def test_repeated_stage_names_keep_history_and_expose_latest_data_and_duration():
    """同名阶段保留每次事件，查询数据与阶段耗时读取最后一次记录。"""
    trace = TraceContext()
    trace.record_stage("retrieve", {"source": "first"}, elapsed_ms=2.5)
    trace.record_stage("retrieve", {"source": "latest", "hits": 3}, elapsed_ms=5.25)
    stages = trace.to_dict()["stages"]

    assert len(stages) == 2
    assert trace.get_stage_data("retrieve") == {"source": "latest", "hits": 3}
    assert trace.elapsed_ms("retrieve") == 5.25
    assert all(_utc_datetime(entry["timestamp"]).tzinfo is not None for entry in stages)


def test_stage_without_duration_keeps_data_but_has_no_elapsed_value():
    """未提供时长的阶段仍可查询数据，耗时查询明确表示无记录。"""
    trace = TraceContext()
    trace.record_stage("load", {"文件": "手册.md"})

    assert trace.get_stage_data("load") == {"文件": "手册.md"}
    with pytest.raises(KeyError):
        trace.elapsed_ms("load")
    assert trace.get_stage_data("missing") is None


def test_record_stage_rejects_invalid_names_data_and_elapsed_values():
    """阶段名、数据类型及时长校验拒绝空值、布尔数和非有限负数。"""
    trace = TraceContext()
    for name in ("", "  ", None):
        with pytest.raises(ValueError):
            trace.record_stage(name, {})
    with pytest.raises(ValueError):
        trace.record_stage("valid", [])
    for duration in (-0.01, math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError):
            trace.record_stage("valid", {}, elapsed_ms=duration)


def test_to_dict_copies_top_level_collections_and_retains_nested_data_reference():
    """序列化结果与 Trace 顶层分离，但嵌套阶段数据仍是调用方数据。"""
    trace = TraceContext()
    trace.record_stage("answer", {"text": "原始正文"})
    exported = trace.to_dict()
    exported["trace_id"] = "外部修改"
    exported["stages"].append({"stage": "external", "data": {}})
    exported["stages"][0]["data"]["text"] = "共享的嵌套数据"

    current = trace.to_dict()
    assert current["trace_id"] != "外部修改"
    assert len(current["stages"]) == 1
    assert trace.get_stage_data("answer") == {"text": "共享的嵌套数据"}


def test_collector_construction_and_disabled_settings_do_not_write_or_finish_trace(tmp_path):
    """构造 Collector 不落盘，禁用配置下 collect 不建目录也不结束 Trace。"""
    path = tmp_path / "disabled" / "traces.jsonl"
    collector = TraceCollector(path)
    trace = TraceContext()
    assert not path.parent.exists()

    settings = SimpleNamespace(observability=ObservabilitySettings(
        log_level="INFO", trace_enabled=False, trace_file=str(path), structured_logging=False,
    ))
    disabled = TraceCollector.from_settings(settings)
    disabled.collect(trace)

    assert collector.path == path
    assert disabled.path == path
    assert trace.finished_at is None
    assert not path.parent.exists()


def test_enabled_settings_append_complete_unicode_trace_as_one_jsonl_record(tmp_path):
    """启用配置会结束 Trace，并将含中文和换行的原始文本完整追加为一行。"""
    path = tmp_path / "enabled" / "traces.jsonl"
    settings = SimpleNamespace(observability=ObservabilitySettings(
        log_level="INFO", trace_enabled=True, trace_file=str(path), structured_logging=True,
    ))
    trace = TraceContext(trace_type="query")
    query = "第一行：如何检索？\n第二行：保留原文"
    chunk = "中文片段\n" + "证据" * 1200 + "尾部标记"
    trace.record_stage("retrieve", {
        "original_query": query, "keywords": ["知识库"], "chunks": [{"text": chunk}],
    }, elapsed_ms=18.75)

    TraceCollector.from_settings(settings).collect(trace)

    raw = path.read_text(encoding="utf-8")
    stored = json.loads(raw)
    assert raw.count("\n") == 1
    assert stored["trace_id"] == trace.trace_id
    assert stored["trace_type"] == "query"
    assert stored["finished_at"] == trace.finished_at
    assert stored["stages"][0]["data"]["original_query"] == query
    assert stored["stages"][0]["data"]["chunks"][0]["text"] == chunk
    assert "status" not in stored


def test_collect_called_twice_appends_two_records_for_the_same_trace(tmp_path):
    """同一对象重复 collect 按调用追加两行，结束时间仍保持不变。"""
    path = tmp_path / "repeat.jsonl"
    collector = TraceCollector(path)
    trace = TraceContext()
    collector.collect(trace)
    first_finished_at = trace.finished_at
    collector.collect(trace)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert [record["trace_id"] for record in records] == [trace.trace_id, trace.trace_id]
    assert trace.finished_at == first_finished_at


def test_collector_disk_failure_warns_by_exception_type_without_raw_error(tmp_path, monkeypatch, caplog):
    """落盘失败只记录异常类型，不泄露底层错误消息且不抛给业务调用方。"""
    path = tmp_path / "io-failure" / "traces.jsonl"
    trace = TraceContext()
    original_open = Path.open

    def fail_selected_path(self, *args, **kwargs):
        if self == path:
            raise OSError("SENTINEL_DISK_ERROR")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(logger_module.Path, "open", fail_selected_path)
    with caplog.at_level(logging.WARNING, logger="src.core.trace.trace_collector"):
        TraceCollector(path).collect(trace)

    assert trace.finished_at is not None
    assert "Trace persistence failed: OSError" in caplog.text
    assert "SENTINEL_DISK_ERROR" not in caplog.text
    assert not path.exists()


def test_write_trace_redacts_sensitive_fields_and_preserves_full_trace_text(tmp_path):
    """字段过滤覆盖凭据、向量和图片载荷，同时保留完整中文查询与证据正文。"""
    class UnknownSDKValue:
        def __repr__(self):
            return "SENTINEL_OBJECT_REPR"

    cycle = {}
    cycle["self"] = cycle
    query = "问题中出现 api_key=正文保留\n第二行仍完整"
    chunk_text = "片段开头\n" + "知识证据" * 1500 + "\n片段末尾标记"
    source_path = tmp_path / "来源.md"
    trace_data = {
        "trace_id": "trace-safe-fields",
        "trace_type": "query",
        "original_query": query,
        "keywords": ["中文关键词"],
        "chunks": [{
            "text": chunk_text,
            "GEMINI_API_KEY": "SENTINEL_API_KEY",
            "headers": {"Authorization": "SENTINEL_HEADER"},
            "authorization": "SENTINEL_AUTHORIZATION",
            "cookie": "SENTINEL_COOKIE",
            "password": "SENTINEL_PASSWORD",
            "access_token": "SENTINEL_ACCESS_TOKEN",
            "dense_vector": [0.1, 0.2],
            "dense_vectors": [[0.2, 0.3]],
            "vector": [0.4, 0.5],
            "vectors": [[0.5, 0.6]],
            "embeddings": [[0.3, 0.4]],
            "base64": "SENTINEL_BASE64",
            "image_data": "SENTINEL_IMAGE_FIELD",
        }],
        "image": {"type": "image", "data": "SENTINEL_IMAGE_DATA"},
        "before": "前置正文",
        "after": "后置正文",
        "source_path": source_path,
        "non_finite": math.nan,
        "unknown_object": UnknownSDKValue(),
        "cycle": cycle,
    }

    write_trace(trace_data, tmp_path / "safe.jsonl")

    raw = (tmp_path / "safe.jsonl").read_text(encoding="utf-8")
    stored = json.loads(raw)
    assert raw.count("\n") == 1
    assert set(stored) == set(trace_data)
    assert stored["original_query"] == query
    assert stored["chunks"][0]["text"] == chunk_text
    assert stored["before"] == "前置正文"
    assert stored["after"] == "后置正文"
    assert stored["source_path"] == str(source_path)
    assert stored["non_finite"] is None
    assert stored["unknown_object"] == "<UnknownSDKValue>"
    assert stored["cycle"]["self"] == "<cycle>"
    assert stored["chunks"][0]["GEMINI_API_KEY"] == "[REDACTED]"
    assert stored["chunks"][0]["headers"] == "[REDACTED]"
    assert stored["chunks"][0]["authorization"] == "[REDACTED]"
    assert stored["chunks"][0]["cookie"] == "[REDACTED]"
    assert stored["chunks"][0]["password"] == "[REDACTED]"
    assert stored["chunks"][0]["access_token"] == "[REDACTED]"
    assert stored["chunks"][0]["dense_vector"] == "[REDACTED]"
    assert stored["chunks"][0]["dense_vectors"] == "[REDACTED]"
    assert stored["chunks"][0]["vector"] == "[REDACTED]"
    assert stored["chunks"][0]["vectors"] == "[REDACTED]"
    assert stored["chunks"][0]["embeddings"] == "[REDACTED]"
    assert stored["chunks"][0]["base64"] == "[REDACTED]"
    assert stored["chunks"][0]["image_data"] == "[REDACTED]"
    assert stored["image"]["data"] == "[REDACTED]"
    for sentinel in (
        "SENTINEL_API_KEY", "SENTINEL_HEADER", "SENTINEL_AUTHORIZATION",
        "SENTINEL_COOKIE", "SENTINEL_PASSWORD", "SENTINEL_ACCESS_TOKEN",
        "SENTINEL_BASE64", "SENTINEL_IMAGE_FIELD", "SENTINEL_IMAGE_DATA",
        "SENTINEL_OBJECT_REPR",
    ):
        assert sentinel not in raw


def test_json_formatter_keeps_log_message_filters_extra_and_records_exception_type():
    """JSON 日志保留调用方消息，过滤敏感 extra，异常只写类型而非原始信息。"""
    record = logging.LogRecord(
        "trace.formatter.test",
        logging.ERROR,
        __file__,
        42,
        "服务日志正文 api_key=正文保留",
        (),
        None,
    )
    record.api_key = "SENTINEL_EXTRA_KEY"
    record.context = {"original_query": "原始问题", "vector": [0.1, 0.2]}
    try:
        raise RuntimeError("SENTINEL_EXCEPTION_MESSAGE")
    except RuntimeError:
        record.exc_info = sys.exc_info()

    raw = JSONFormatter().format(record)
    stored = json.loads(raw)

    assert stored["level"] == "ERROR"
    assert stored["logger"] == "trace.formatter.test"
    assert stored["message"] == "服务日志正文 api_key=正文保留"
    assert _utc_datetime(stored["timestamp"]).tzinfo is not None
    assert stored["api_key"] == "[REDACTED]"
    assert stored["context"] == {"original_query": "原始问题", "vector": "[REDACTED]"}
    assert stored["exception"] == "RuntimeError"
    assert "SENTINEL_EXTRA_KEY" not in raw
    assert "SENTINEL_EXCEPTION_MESSAGE" not in raw


def test_trace_logger_reuses_path_writes_one_json_record_and_stays_off_stdio(
    tmp_path, capsys, isolated_logger_name
):
    """相同 logger 名称和路径复用输出，日志写 JSONL 文件且不传播到终端。"""
    path = tmp_path / "trace-logger.jsonl"
    name = isolated_logger_name()
    first = get_trace_logger(path, name=name)
    second = get_trace_logger(path, name=name)

    assert first is second
    first.info("追踪阶段完成", extra={"trace_id": "trace-1", "stage": "retrieve"})
    captured = capsys.readouterr()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert captured.out == ""
    assert captured.err == ""
    assert len(records) == 1
    assert records[0]["message"] == "追踪阶段完成"
    assert records[0]["trace_id"] == "trace-1"
    assert records[0]["stage"] == "retrieve"
    assert records[0]["level"] == "INFO"


def test_trace_logger_rejects_rebinding_same_name_to_another_path(tmp_path, isolated_logger_name):
    """已经绑定文件的 Trace logger 拒绝用同名写入另一个路径。"""
    name = isolated_logger_name()
    first_path = tmp_path / "first" / "traces.jsonl"
    second_path = tmp_path / "second" / "traces.jsonl"
    get_trace_logger(first_path, name=name)

    with pytest.raises(ValueError, match="another path"):
        get_trace_logger(second_path, name=name)
    assert first_path.parent.exists()
    assert not second_path.parent.exists()


def test_regular_logger_remains_human_readable_on_stderr(capsys, isolated_logger_name):
    """普通 logger 使用人类可读 stderr 输出，不向 stdout 回显。"""
    logger = get_logger(isolated_logger_name(), "INFO")
    logger.info("普通运行日志")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "INFO" in captured.err
    assert "普通运行日志" in captured.err


def test_concurrent_logger_and_direct_writer_append_parseable_records_without_loss(
    tmp_path, isolated_logger_name
):
    """多线程混合追加日志记录与原始 Trace 时，每一行均可解析且无记录丢失。"""
    path = tmp_path / "concurrent" / "traces.jsonl"
    logger = get_trace_logger(path, name=isolated_logger_name())
    workers = 5
    records_per_kind = 12

    def append_records(worker):
        for index in range(records_per_kind):
            write_trace(
                {
                    "trace_id": f"direct-{worker}-{index}",
                    "trace_type": "ingestion",
                    "stages": [],
                },
                path,
            )
            logger.info(
                "并发日志",
                extra={"kind": "logger_record", "sequence": index, "worker": worker},
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(append_records, range(workers)))

    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    direct = [record for record in records if "trace_type" in record]
    logged = [record for record in records if "level" in record]

    assert len(records) == workers * records_per_kind * 2
    assert len(direct) == workers * records_per_kind
    assert len(logged) == workers * records_per_kind
    assert all(record["trace_type"] == "ingestion" for record in direct)
    assert all(record["kind"] == "logger_record" for record in logged)
    assert len({record["trace_id"] for record in direct}) == len(direct)
