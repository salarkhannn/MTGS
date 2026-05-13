"""PyTorch DDP communication hook for MTGS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..shadow.copy_stream import ShadowCopyManager
from ..shadow.rollback import rollback_model_state
from ..utils.logging import JsonlLogger
from .transaction import TransactionDecision, TransactionManager, TransactionResult


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for MTGS comm hooks") from exc
    return torch


@dataclass
class MTGSState:
    """State object carried by the DDP comm hook."""

    model: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    process_group: Any | None = None
    enable_hook: bool = True
    enable_shadow: bool = True
    enable_2pc: bool = True
    debug: bool = False
    rank: int = 0
    shadow_manager: ShadowCopyManager = field(default_factory=ShadowCopyManager)
    transaction_manager: TransactionManager = field(default_factory=TransactionManager)
    logger: JsonlLogger | None = None
    current_step: int = 0
    snapshot_step: int | None = None
    last_transaction: TransactionResult | None = None
    force_abort: bool = False

    def start_step(self, step: int) -> None:
        self.current_step = step
        self.snapshot_step = None
        self.force_abort = False

    def snapshot_if_needed(self) -> None:
        if not self.enable_shadow or self.snapshot_step == self.current_step:
            return
        if self.model is None:
            return
        result = self.shadow_manager.capture(
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.snapshot_step = self.current_step
        self._log(
            "shadow_copied",
            step=self.current_step,
            shadow_bytes=result.snapshot.total_bytes,
            latency_ms=result.latency_ms,
            pinned=result.snapshot.pinned,
        )

    def rollback(self, reason: str) -> None:
        if self.model is None or self.shadow_manager.snapshot is None:
            self._log("rollback_skipped", step=self.current_step, reason=reason)
            return
        result = rollback_model_state(
            self.model,
            self.shadow_manager.snapshot,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            reason=reason,
        )
        self._log(
            "rollback_complete",
            step=self.current_step,
            reason=reason,
            restored_tensors=result.restored_tensors,
            duration_ms=result.duration_ms,
        )

    def complete_step(self) -> None:
        self.shadow_manager.clear()
        self.snapshot_step = None

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, rank=self.rank, **fields)


def _completed_future(tensor: Any) -> Any:
    torch_mod = _require_torch()
    future = torch_mod.futures.Future()
    future.set_result(tensor)
    return future


def _bucket_index(bucket: Any) -> int:
    value = getattr(bucket, "index", None)
    if callable(value):
        return int(value())
    return int(value or 0)


def _all_reduce_bucket(state: MTGSState, tensor: Any) -> Any:
    torch_mod = _require_torch()
    dist = torch_mod.distributed
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    group = state.process_group
    dist.all_reduce(tensor, group=group)
    tensor.div_(dist.get_world_size(group))
    return tensor


def mtgs_comm_hook(state: MTGSState, bucket: Any) -> Any:
    """Wrap gradient synchronization in snapshot + 2PC decision logic."""

    torch_mod = _require_torch()
    buffer = bucket.buffer()
    if not state.enable_hook:
        return _completed_future(_all_reduce_bucket(state, buffer))

    bucket_id = _bucket_index(bucket)
    transaction_id = f"step{state.current_step}:bucket{bucket_id}"
    state.snapshot_if_needed()
    reduced = _all_reduce_bucket(state, buffer)
    finite = bool(torch_mod.isfinite(reduced).all().item())
    local_healthy = finite and not state.force_abort
    reason = "non_finite_gradient" if not finite else "forced_abort" if state.force_abort else ""

    if state.enable_2pc:
        result = state.transaction_manager.vote_and_decide(
            transaction_id=transaction_id,
            local_healthy=local_healthy,
            reason=reason,
            device=reduced.device,
        )
    else:
        result = TransactionResult(
            transaction_id=transaction_id,
            decision=TransactionDecision.COMMIT if local_healthy else TransactionDecision.ABORT,
            votes=[1 if local_healthy else 0],
            reason=reason or "local_only",
            duration_ms=0.0,
        )

    state.last_transaction = result
    state._log(
        "transaction_decided",
        step=state.current_step,
        transaction_id=transaction_id,
        decision=result.decision.value,
        votes=result.votes,
        reason=result.reason,
        duration_ms=result.duration_ms,
    )
    if not result.committed:
        state.rollback(result.reason)
        reduced.zero_()

    return _completed_future(reduced)


def register_mtgs_comm_hook(model: Any, state: MTGSState) -> None:
    """Register MTGS hook when available, with a clean fallback for tests."""

    state.model = model
    if not hasattr(model, "register_comm_hook"):
        raise TypeError("model must be DistributedDataParallel-like")
    model.register_comm_hook(state, mtgs_comm_hook)
