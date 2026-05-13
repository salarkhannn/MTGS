"""Throughput measurement and CSV export."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThroughputRecord:
    timestamp: float
    rank: int
    epoch: int
    step: int
    tokens: int
    duration_s: float
    tokens_per_second: float
    inclusive_tokens_per_second: float
    loss: float
    status: str
    warmup_excluded: bool


class ThroughputLogger:
    """Tracks step and epoch token throughput.

    `tokens_per_second` is zero during warmup-excluded steps so downstream
    analysis can aggregate the metric directly without needing another flag.
    `inclusive_tokens_per_second` always reports the measured step throughput.
    """

    header = [
        "timestamp",
        "rank",
        "epoch",
        "step",
        "tokens",
        "duration_s",
        "tokens_per_second",
        "inclusive_tokens_per_second",
        "loss",
        "status",
        "warmup_excluded",
    ]

    def __init__(self, path: str | Path, rank: int, warmup_steps: int = 0) -> None:
        self.path = Path(path)
        self.rank = rank
        self.warmup_steps = warmup_steps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.header).writeheader()

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def record_step(
        self,
        *,
        epoch: int,
        step: int,
        tokens: int,
        start_ns: int,
        loss: float,
        status: str = "ok",
    ) -> ThroughputRecord:
        end_ns = self.now_ns()
        duration_s = max((end_ns - start_ns) / 1_000_000_000, 1e-12)
        inclusive = tokens / duration_s
        warmup_excluded = step <= self.warmup_steps
        measured = 0.0 if warmup_excluded else inclusive
        record = ThroughputRecord(
            timestamp=time.time(),
            rank=self.rank,
            epoch=epoch,
            step=step,
            tokens=tokens,
            duration_s=duration_s,
            tokens_per_second=measured,
            inclusive_tokens_per_second=inclusive,
            loss=loss,
            status=status,
            warmup_excluded=warmup_excluded,
        )
        self._append(record)
        return record

    def _append(self, record: ThroughputRecord) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.header)
            writer.writerow(record.__dict__)
