"""Dataloader utilities for distributed training."""

from __future__ import annotations

import math
from typing import Any, Dict, List


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


def compute_shard_indices(
    dataset_size: int,
    world_size: int,
    rank: int,
    drop_last: bool = True,
) -> List[int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within world_size")

    if drop_last:
        num_samples = dataset_size // world_size
        total_size = num_samples * world_size
        indices = list(range(total_size))
    else:
        num_samples = math.ceil(dataset_size / world_size)
        total_size = num_samples * world_size
        indices = list(range(dataset_size))
        indices.extend(list(range(total_size - dataset_size)))

    return indices[rank:total_size:world_size]


def validate_no_overlap(
    dataset_size: int,
    world_size: int,
    drop_last: bool = True,
) -> Dict[str, Any]:
    shards = [
        set(compute_shard_indices(dataset_size, world_size, rank, drop_last))
        for rank in range(world_size)
    ]
    overlap_found = False
    for idx, shard in enumerate(shards):
        for other in shards[idx + 1 :]:
            if shard.intersection(other):
                overlap_found = True
                break
        if overlap_found:
            break

    return {
        "overlap_found": overlap_found,
        "per_rank_counts": [len(shard) for shard in shards],
    }
