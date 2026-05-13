"""Rollback helpers for restoring CPU shadow state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .allocator import ShadowSnapshot


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for rollback") from exc
    return torch


@dataclass(frozen=True)
class RollbackResult:
    restored_tensors: int
    duration_ms: float
    reason: str


def _move_state_to_device(value: Any, device: Any) -> Any:
    torch_mod = _require_torch()
    if torch_mod.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_device(item, device) for item in value)
    return value


def rollback_model_state(
    model: Any,
    snapshot: ShadowSnapshot,
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    reason: str = "abort",
) -> RollbackResult:
    """Restore model, optimizer, and scheduler state from a shadow snapshot."""

    start = time.perf_counter_ns()
    module = getattr(model, "module", model)
    current_state = module.state_dict()
    restored: dict[str, Any] = {}
    for name, shadow_tensor in snapshot.tensors.items():
        device = current_state[name].device
        restored[name] = shadow_tensor.to(device)
    module.load_state_dict(restored, strict=True)

    if optimizer is not None and snapshot.optimizer_state is not None:
        first_param = next(module.parameters(), None)
        device = first_param.device if first_param is not None else "cpu"
        optimizer.load_state_dict(_move_state_to_device(snapshot.optimizer_state, device))

    if scheduler is not None and snapshot.scheduler_state is not None:
        scheduler.load_state_dict(snapshot.scheduler_state)

    duration_ms = (time.perf_counter_ns() - start) / 1_000_000
    return RollbackResult(
        restored_tensors=len(snapshot.tensors),
        duration_ms=duration_ms,
        reason=reason,
    )
