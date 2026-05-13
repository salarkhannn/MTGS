"""Small wrappers around torch.distributed setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    initialized: bool

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for distributed execution") from exc
    return torch


def _select_backend(torch_mod: Any, requested: str | None) -> str:
    if requested:
        return requested
    return "nccl" if torch_mod.cuda.is_available() else "gloo"


def init_distributed(
    backend: str | None = None,
    timeout_s: float = 120.0,
) -> DistributedContext:
    """Initialize process group when launched with torchrun.

    A single-process invocation returns an uninitialized context, which keeps
    local smoke tests lightweight.
    """

    torch_mod = _require_torch()
    dist = torch_mod.distributed
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    selected = _select_backend(torch_mod, backend)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=selected,
            init_method="env://",
            timeout=timedelta(seconds=timeout_s),
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=selected,
        initialized=dist.is_available() and dist.is_initialized(),
    )


def cleanup_distributed() -> None:
    torch_mod = _require_torch()
    dist = torch_mod.distributed
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
