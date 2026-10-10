"""F1：沿用上游的一次执行 Trace、阶段数据与单调时钟计时。"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal


@dataclass
class TraceContext:
    """允许记录原问题/片段/前后正文；敏感字段过滤在写入层，不改变文本策略。"""

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _start_mono: float = field(default_factory=lambda: time.monotonic(), init=False, repr=False)
    _finish_mono: float | None = field(default=None, init=False, repr=False)
    _stage_timings: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace_type not in {"query", "ingestion"}:
            raise ValueError("trace_type must be query or ingestion")

    def record_stage(self, stage_name: str, data: dict[str, Any], elapsed_ms: float | None = None) -> None:
        """调用方测量阶段耗时，原数据按上游保留；不自动截取文本或计算阶段总和。"""
        if not isinstance(stage_name, str) or not stage_name.strip() or not isinstance(data, dict):
            raise ValueError("Stage needs a nonblank name and dictionary data")
        entry = {"stage": stage_name, "timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
        if elapsed_ms is not None:
            if not isinstance(elapsed_ms, (int, float)) or isinstance(elapsed_ms, bool) or not isfinite(elapsed_ms) or elapsed_ms < 0:
                raise ValueError("elapsed_ms must be a finite nonnegative number")
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    def finish(self) -> None:
        """冻结总耗时；重复 finish 不推迟结束时间。"""
        if self._finish_mono is None:
            self._finish_mono = time.monotonic()
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """总耗时独立测量，不能把并行的阶段耗时直接相加。"""
        if stage_name is not None:
            return self._stage_timings[stage_name]
        end = time.monotonic() if self._finish_mono is None else self._finish_mono
        return (end - self._start_mono) * 1000

    def to_dict(self) -> dict[str, Any]:
        """按上游字段序列化；只复制顶层，嵌套 data 仍共享，未执行落盘过滤。"""
        return {"trace_id": self.trace_id, "trace_type": self.trace_type, "started_at": self.started_at,
                "finished_at": self.finished_at, "total_elapsed_ms": round(self.elapsed_ms(), 2),
                "stages": list(self.stages), "metadata": dict(self.metadata)}

    def get_stage_data(self, stage_name: str) -> dict[str, Any] | None:
        """同名阶段返回最后记录的数据；没有该阶段时返回 None。"""
        for entry in reversed(self.stages):
            if entry["stage"] == stage_name:
                return entry["data"]
        return None
