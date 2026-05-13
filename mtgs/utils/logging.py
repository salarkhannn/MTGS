"""Structured JSONL logging helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSONL logger used by runners and scripts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
