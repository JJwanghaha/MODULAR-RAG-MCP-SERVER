"""F2：保留 stderr 运行日志，增加 JSONL 格式器与 Trace 追加写入。"""

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path

DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")
_WRITE_LOCK = threading.RLock()
_SENSITIVE_FIELDS = {
    "authorization", "headers", "cookie", "cookies", "password", "secret", "clientsecret", "token",
    "apitoken", "bearertoken", "accesstoken", "refreshtoken",
    "base64", "imagedata", "imagebase64", "imagebytes", "densevector", "densevectors", "sparsevector",
    "sparsevectors", "queryvector", "vector", "vectors", "embeddings",
}


def _log_value(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    """保留文本，不猜测正文中的秘密；过滤明确字段，不用任意对象 repr 泄漏内部状态。"""
    if isinstance(value, dict):
        if id(value) in seen:
            return "<cycle>"
        nested = seen | {id(value)}
        result = {}
        mime = value.get("mimeType", value.get("mime_type"))
        kind = value.get("type")
        image_block = isinstance(kind, str) and kind == "image" or isinstance(mime, str) and mime.startswith("image/")
        for key, item in value.items():
            normalized = "".join(char for char in str(key).lower() if char.isalnum())
            if normalized in _SENSITIVE_FIELDS or normalized.endswith("apikey"):
                result[str(key)] = "[REDACTED]"
            elif normalized == "data" and image_block:
                result[str(key)] = "[REDACTED]"
            elif normalized == "embedding" and isinstance(item, (list, tuple)):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _log_value(item, nested)
        return result
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return "<cycle>"
        return [_log_value(item, seen | {id(value)}) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def get_logger(name: str = "modular-rag", log_level: str | None = "INFO") -> logging.Logger:
    """返回一个不会重复添加 Handler 的标准日志对象。"""
    logger = logging.getLogger(name)
    logger.setLevel((log_level or "INFO").upper())

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    return logger


class JSONFormatter(logging.Formatter):
    """每条运行日志为一个 JSON 对象；异常只写类型，不复制原服务异常/traceback。"""

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno",
        "module", "msecs", "message", "msg", "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload = {"timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                   "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        payload.update({key: value for key, value in record.__dict__.items() if key not in self._INTERNAL_ATTRS and key not in payload})
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(_log_value(payload), ensure_ascii=False, allow_nan=False)


class _TraceFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # 与直接 write_trace 共用本进程锁；不宣称跨进程文件写入原子性。
        with _WRITE_LOCK:
            super().emit(record)


def get_trace_logger(traces_path: str | Path = DEFAULT_TRACES_PATH, *, name: str = "modular-rag.trace") -> logging.Logger:
    """显式调用才开文件；相同名称/路径复用 handler，同名换路径拒绝，避免写错文件。"""
    path = resolve_path(traces_path)
    logger = logging.getLogger(name)
    with _WRITE_LOCK:
        owned = [handler for handler in logger.handlers if isinstance(handler, _TraceFileHandler)]
        if owned and any(Path(handler.baseFilename) != path for handler in owned):
            raise ValueError("Trace logger name is already bound to another path")
        if not owned:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = _TraceFileHandler(path, encoding="utf-8")
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def write_trace(trace_dict: dict[str, Any], traces_path: str | Path = DEFAULT_TRACES_PATH) -> None:
    """直接追加一个完整 Trace JSON 行，不经过 logging 信封，不写 stdout。"""
    path = resolve_path(traces_path)
    line = json.dumps(_log_value(trace_dict), ensure_ascii=False, allow_nan=False)
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
