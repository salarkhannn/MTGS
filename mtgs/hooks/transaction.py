"""Two-phase commit primitives for MTGS gradient transactions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TransactionDecision(str, Enum):
    COMMIT = "commit"
    ABORT = "abort"


@dataclass(frozen=True)
class TransactionResult:
    transaction_id: str
    decision: TransactionDecision
    votes: list[int]
    reason: str
    duration_ms: float

    @property
    def committed(self) -> bool:
        return self.decision == TransactionDecision.COMMIT


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for 2PC transactions") from exc
    return torch


class TransactionManager:
    """Prepare, vote, decide, and broadcast 2PC outcomes."""

    def __init__(
        self,
        *,
        coordinator_rank: int = 0,
        timeout_s: float = 5.0,
        process_group: Any | None = None,
    ) -> None:
        self.coordinator_rank = coordinator_rank
        self.timeout_s = timeout_s
        self.process_group = process_group

    def vote_and_decide(
        self,
        *,
        transaction_id: str,
        local_healthy: bool,
        reason: str = "",
        device: Any | None = None,
    ) -> TransactionResult:
        torch_mod = _require_torch()
        dist = torch_mod.distributed
        start = time.perf_counter_ns()
        device = device or torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")

        if not dist.is_available() or not dist.is_initialized():
            decision = (
                TransactionDecision.COMMIT
                if local_healthy
                else TransactionDecision.ABORT
            )
            return TransactionResult(
                transaction_id=transaction_id,
                decision=decision,
                votes=[1 if local_healthy else 0],
                reason=reason if not local_healthy else "all_votes_yes",
                duration_ms=(time.perf_counter_ns() - start) / 1_000_000,
            )

        group = self.process_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        prepare = torch_mod.tensor([1], dtype=torch_mod.int32, device=device)
        dist.broadcast(prepare, src=self.coordinator_rank, group=group)
        if int(prepare.item()) != 1:
            local_healthy = False
            reason = reason or "prepare_rejected"

        vote = torch_mod.tensor(
            [1 if local_healthy else 0],
            dtype=torch_mod.int32,
            device=device,
        )
        gathered = [torch_mod.zeros_like(vote) for _ in range(world_size)]
        try:
            dist.all_gather(gathered, vote, group=group)
            votes = [int(item.item()) for item in gathered]
            should_commit = all(item == 1 for item in votes)
            decision_value = 1 if should_commit else 0
        except Exception:
            votes = [0 if i == rank else -1 for i in range(world_size)]
            should_commit = False
            decision_value = 0
            reason = reason or "vote_timeout_or_collective_error"

        decision_tensor = torch_mod.tensor([decision_value], dtype=torch_mod.int32, device=device)
        dist.broadcast(decision_tensor, src=self.coordinator_rank, group=group)
        decision = (
            TransactionDecision.COMMIT
            if int(decision_tensor.item()) == 1
            else TransactionDecision.ABORT
        )
        return TransactionResult(
            transaction_id=transaction_id,
            decision=decision,
            votes=votes,
            reason=reason if decision == TransactionDecision.ABORT else "all_votes_yes",
            duration_ms=(time.perf_counter_ns() - start) / 1_000_000,
        )
