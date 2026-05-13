"""Async copy stream management for shadow snapshots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .allocator import ShadowSnapshot, allocate_model_snapshot


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for shadow copy streams") from exc
    return torch


@dataclass
class ShadowCopyResult:
    snapshot: ShadowSnapshot
    latency_ms: float


class ShadowCopyManager:
    """Captures and restores per-step model snapshots."""

    def __init__(
        self,
        *,
        pin_memory: bool = True,
        max_bytes: int | None = None,
    ) -> None:
        self.pin_memory = pin_memory
        self.max_bytes = max_bytes
        self.snapshot: ShadowSnapshot | None = None
        self.copy_latency_ms = 0.0
        torch_mod = _require_torch()
        self.stream = torch_mod.cuda.Stream() if torch_mod.cuda.is_available() else None

    def capture(
        self,
        model: Any,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> ShadowCopyResult:
        torch_mod = _require_torch()
        start = time.perf_counter_ns()
        if self.stream is not None:
            with torch_mod.cuda.stream(self.stream):
                snapshot = allocate_model_snapshot(
                    model,
                    pin_memory=self.pin_memory,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    max_bytes=self.max_bytes,
                )
        else:
            snapshot = allocate_model_snapshot(
                model,
                pin_memory=False,
                optimizer=optimizer,
                scheduler=scheduler,
                max_bytes=self.max_bytes,
            )
        self.synchronize()
        self.copy_latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.snapshot = snapshot
        return ShadowCopyResult(snapshot=snapshot, latency_ms=self.copy_latency_ms)

    def synchronize(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()

    def clear(self) -> None:
        self.snapshot = None
        self.copy_latency_ms = 0.0
