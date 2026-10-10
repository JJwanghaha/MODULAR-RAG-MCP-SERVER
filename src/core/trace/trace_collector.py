"""F1/F2：完成 Trace 后写 JSONL；日志存储失败不应导致业务失败。"""

import logging
from pathlib import Path

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext
from src.observability.logger import DEFAULT_TRACES_PATH, write_trace

logger = logging.getLogger(__name__)


class TraceCollector:
    def __init__(self, traces_path: str | Path = DEFAULT_TRACES_PATH, *, enabled: bool = True):
        self._path = resolve_path(traces_path)
        self.enabled = enabled

    @classmethod
    def from_settings(cls, settings) -> "TraceCollector":
        """为后续 F3/F4 接入现有 trace_enabled/trace_file；当前业务入口尚未调用。"""
        return cls(settings.observability.trace_file, enabled=settings.observability.trace_enabled)

    @property
    def path(self) -> Path:
        return self._path

    def collect(self, trace: TraceContext) -> None:
        """禁用时不创建目录；启用时自动 finish，再追加，不暗中缓存或重试。"""
        if not self.enabled:
            return
        if trace.finished_at is None:
            trace.finish()
        try:
            write_trace(trace.to_dict(), self._path)
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            logger.warning("Trace persistence failed: %s", type(exc).__name__)
