"""Pinned CPU shadow-state allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for shadow state allocation") from exc
    return torch


@dataclass
class ShadowSnapshot:
    """CPU-resident shadow state for rollback."""

    tensors: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    total_bytes: int = 0
    pinned: bool = False
    granularity: str = "full-model"


def _can_pin(torch_mod: Any, requested: bool) -> bool:
    return bool(requested and torch_mod.cuda.is_available())


def _clone_tensor_to_cpu(tensor: Any, pin_memory: bool) -> Any:
    torch_mod = _require_torch()
    cpu_tensor = torch_mod.empty_like(tensor, device="cpu", pin_memory=pin_memory)
    cpu_tensor.copy_(tensor.detach().to("cpu"), non_blocking=pin_memory)
    return cpu_tensor


def _clone_state_to_cpu(value: Any, pin_memory: bool) -> Any:
    torch_mod = _require_torch()
    if torch_mod.is_tensor(value):
        return _clone_tensor_to_cpu(value, pin_memory)
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item, pin_memory) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item, pin_memory) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item, pin_memory) for item in value)
    return value


def estimate_snapshot_bytes(model: Any) -> int:
    module = getattr(model, "module", model)
    return int(sum(tensor.numel() * tensor.element_size() for tensor in module.state_dict().values()))


def allocate_model_snapshot(
    model: Any,
    *,
    pin_memory: bool = True,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    max_bytes: int | None = None,
) -> ShadowSnapshot:
    """Clone model, optimizer, and scheduler state into CPU shadow storage."""

    torch_mod = _require_torch()
    use_pinned = _can_pin(torch_mod, pin_memory)
    module = getattr(model, "module", model)
    state = module.state_dict()
    total_bytes = estimate_snapshot_bytes(module)
    if max_bytes is not None and total_bytes > max_bytes:
        raise MemoryError(f"shadow snapshot needs {total_bytes} bytes, budget is {max_bytes}")

    tensors = {
        name: _clone_tensor_to_cpu(tensor, use_pinned)
        for name, tensor in state.items()
    }
    optimizer_state = (
        _clone_state_to_cpu(optimizer.state_dict(), use_pinned)
        if optimizer is not None
        else None
    )
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    return ShadowSnapshot(
        tensors=tensors,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        total_bytes=total_bytes,
        pinned=use_pinned,
    )
