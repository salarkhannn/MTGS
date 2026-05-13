"""Structured trace export for communication, copy, and rollback events."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TraceEvent:
    name: str
    rank: int
    transaction_id: str
    started_at: float
    ended_at: float
    duration_ms: float
    metadata: dict[str, str]


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    @contextmanager
    def record(
        self,
        name: str,
        *,
        rank: int,
        transaction_id: str,
        **metadata: str,
    ) -> Iterator[None]:
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.events.append(
                TraceEvent(
                    name=name,
                    rank=rank,
                    transaction_id=transaction_id,
                    started_at=start,
                    ended_at=end,
                    duration_ms=(end - start) * 1000,
                    metadata=metadata,
                )
            )

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([asdict(event) for event in self.events], indent=2),
            encoding="utf-8",
        )
