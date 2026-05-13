"""ETTR event timing and summary export."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ETTREvent:
    event_id: str
    rank: int
    step: int
    reason: str
    detect_time: float
    resume_time: float

    @property
    def ettr_ms(self) -> float:
        return (self.resume_time - self.detect_time) * 1000


class ETTRTimer:
    def __init__(self) -> None:
        self._pending: dict[str, tuple[int, int, str, float]] = {}
        self.events: list[ETTREvent] = []

    def mark_detected(
        self,
        event_id: str,
        *,
        rank: int,
        step: int,
        reason: str,
        timestamp: float | None = None,
    ) -> None:
        self._pending[event_id] = (rank, step, reason, timestamp or time.time())

    def mark_resumed(
        self,
        event_id: str,
        *,
        timestamp: float | None = None,
    ) -> ETTREvent:
        rank, step, reason, detect_time = self._pending.pop(event_id)
        event = ETTREvent(
            event_id=event_id,
            rank=rank,
            step=step,
            reason=reason,
            detect_time=detect_time,
            resume_time=timestamp or time.time(),
        )
        self.events.append(event)
        return event

    def write_events(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "event_id",
                    "rank",
                    "step",
                    "reason",
                    "detect_time",
                    "resume_time",
                    "ettr_ms",
                ],
            )
            writer.writeheader()
            for event in self.events:
                writer.writerow(
                    {
                        "event_id": event.event_id,
                        "rank": event.rank,
                        "step": event.step,
                        "reason": event.reason,
                        "detect_time": event.detect_time,
                        "resume_time": event.resume_time,
                        "ettr_ms": event.ettr_ms,
                    }
                )

    def summary(self) -> dict[str, float]:
        values = sorted(event.ettr_ms for event in self.events)
        if not values:
            return {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "worst_ms": 0.0}
        p95_index = min(len(values) - 1, int(0.95 * (len(values) - 1)))
        return {
            "count": float(len(values)),
            "median_ms": statistics.median(values),
            "p95_ms": values[p95_index],
            "worst_ms": max(values),
        }
