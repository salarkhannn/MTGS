# Deliverable 3 MTGS Core Implementation Evidence

## Communication Hook Interception

MTGS intercepts PyTorch DDP gradient synchronization through
`DistributedDataParallel.register_comm_hook`, implemented in
`mtgs/hooks/comm_hook.py`. The hook receives each gradient bucket, captures a
pre-sync shadow snapshot for the current step, runs the all-reduce fallback, and
wraps the bucket decision in a two-phase commit vote.

Fallback behavior is explicit: `--mtgs-disable-hook` bypasses MTGS logic and
uses the default all-reduce path. `--mtgs-disable-2pc` keeps local health checks
but skips distributed voting. `--mtgs-disable-shadow` disables snapshot capture
for overhead isolation experiments.

## Shadow State Policy

The implemented granularity is a full-model snapshot per training step. Model
state is cloned to CPU before gradient synchronization, with pinned CPU memory
used when CUDA is available. CPU-only test environments fall back to regular CPU
tensors because PyTorch does not expose a pinned-memory allocator without an
accelerator backend.

Copy timing is tied to the first gradient bucket observed for a step. The hook
captures exactly one snapshot per step and releases it after the optimizer step
commits. `ShadowCopyManager` records copy latency in milliseconds and enforces
an optional byte budget through `max_bytes`.

## 2PC Vote and Abort Policy

Rank 0 acts as the coordinator. The voting policy is all-rank quorum:

- prepare: coordinator broadcasts readiness for the transaction
- vote: every rank votes `1` only if the reduced gradient bucket is finite and
  no local abort flag is pending
- decision: coordinator commits only when every vote is `1`; otherwise all ranks
  receive abort

Abort triggers include non-finite gradients, forced local aborts used by tests,
prepare rejection, and collective exceptions during vote gathering. The rollback
completion criterion is a strict reload of model state from the CPU shadow
snapshot plus optimizer and scheduler state restoration when those states were
captured.

## Integration Runner

`python -m mtgs.trainer --mode mtgs` enables the MTGS runner. The local forced
failure path uses `--mtgs-force-abort-step N`, which snapshots the model,
simulates an abort at the selected step, restores the state, and records
`rollback_complete` plus `step_aborted` events in JSONL logs.
