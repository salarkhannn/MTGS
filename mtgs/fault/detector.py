"""Failure detection helpers for collective operations."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureSignal:
    detected_at: float
    rank: int
    step: int
    reason: str
    error_code: str


class FailureDetector:
    """Maps runtime exceptions and timeouts into structured failure signals."""

    def __init__(self, *, rank: int, timeout_s: float = 5.0) -> None:
        self.rank = rank
        self.timeout_s = timeout_s

    def detect_exception(self, exc: BaseException, *, step: int) -> FailureSignal:
        message = str(exc).lower()
        if "timeout" in message:
            code = "collective_timeout"
        elif "out of memory" in message or "oom" in message:
            code = "oom"
        elif "connection" in message or "closed" in message:
            code = "rank_missing"
        else:
            code = "distributed_error"
        return FailureSignal(
            detected_at=time.time(),
            rank=self.rank,
            step=step,
            reason=str(exc),
            error_code=code,
        )

    def detect_elapsed(self, *, started_at: float, step: int) -> FailureSignal | None:
        elapsed = time.perf_counter() - started_at
        if elapsed <= self.timeout_s:
            return None
        return FailureSignal(
            detected_at=time.time(),
            rank=self.rank,
            step=step,
            reason=f"operation exceeded {self.timeout_s:.3f}s",
            error_code="collective_timeout",
        )
