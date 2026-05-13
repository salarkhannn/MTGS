"""Dataloader utilities for distributed training."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required to build distributed samplers") from exc
    return torch


def build_distributed_sampler(
    dataset: Any,
    world_size: int,
    rank: int,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> Any:
    torch_mod = _require_torch()
    distributed = torch_mod.utils.data.distributed
    return distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
