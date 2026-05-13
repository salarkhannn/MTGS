# MTGS Novelty and Trade-Off Analysis

## Novel Improvement

MTGS wraps DDP gradient bucket synchronization in a micro-transaction:
pre-step CPU shadow snapshot, bucket all-reduce, 2PC vote, deterministic
commit/abort, and in-memory rollback. The novelty is the combination of PyTorch
communication hook interception with transaction-style rollback at gradient sync
boundaries.

## Code Evidence

- `mtgs/hooks/comm_hook.py`: DDP hook interception and per-bucket transaction IDs
- `mtgs/hooks/transaction.py`: prepare, vote collection, and decision broadcast
- `mtgs/shadow/`: CPU shadow allocation, copy-stream management, and rollback
- `mtgs/trainer.py`: unified baseline/MTGS runner with enable/disable flags
- `mtgs/profiling/ettr_timer.py`: failure-detect to resume ETTR measurement

## Trade-Offs

MTGS trades CPU memory and copy latency for faster recovery. It is most useful
when failures are frequent enough that disk checkpoint restart time dominates
steady-state overhead. It is least useful for very stable clusters, models whose
shadow state exceeds CPU RAM, or networks where extra control collectives are
more expensive than a conventional checkpoint.

The target operating envelope remains the D2 hypothesis: sub-second local
recovery with less than 5 percent throughput degradation in comparable no-fault
runs. The local smoke run validates the mechanics; final cloud runs are needed
for hardware-backed performance claims.
