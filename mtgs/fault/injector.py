"""SIGKILL fault injection daemon with dry-run safety."""

from __future__ import annotations

import argparse
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.logging import JsonlLogger


@dataclass(frozen=True)
class RankProcess:
    rank: int
    pid: int
    label: str = "worker"


@dataclass(frozen=True)
class FaultInjectionEvent:
    timestamp: float
    target_rank: int
    pid: int
    policy: str
    dry_run: bool
    action: str
    metadata: dict[str, str]


def parse_rank_pid(value: str) -> RankProcess:
    rank, pid = value.split(":", 1)
    return RankProcess(rank=int(rank), pid=int(pid))


def select_target(
    processes: list[RankProcess],
    *,
    policy: str,
    iteration: int,
    protected_ranks: set[int] | None = None,
    specific_rank: int | None = None,
    rng: random.Random | None = None,
) -> RankProcess:
    protected = protected_ranks or set()
    candidates = [proc for proc in processes if proc.rank not in protected]
    if not candidates:
        raise ValueError("no killable rank processes after applying protections")

    if policy == "specific":
        if specific_rank is None:
            raise ValueError("specific policy requires specific_rank")
        for proc in candidates:
            if proc.rank == specific_rank:
                return proc
        raise ValueError(f"rank {specific_rank} is not killable")
    if policy == "round_robin":
        return candidates[iteration % len(candidates)]
    if policy == "random":
        return (rng or random).choice(candidates)
    raise ValueError(f"unknown target policy: {policy}")


def _safe_to_kill(pid: int, protected_pids: set[int]) -> bool:
    return pid > 1 and pid not in protected_pids and pid != os.getpid()


def inject_fault(
    target: RankProcess,
    *,
    policy: str,
    dry_run: bool,
    protected_pids: set[int] | None = None,
) -> FaultInjectionEvent:
    protected = protected_pids or set()
    if not _safe_to_kill(target.pid, protected):
        raise ValueError(f"refusing to kill protected or invalid pid {target.pid}")

    action = "dry_run_logged"
    if not dry_run:
        os.kill(target.pid, signal.SIGKILL)
        action = "sigkill_sent"

    return FaultInjectionEvent(
        timestamp=time.time(),
        target_rank=target.rank,
        pid=target.pid,
        policy=policy,
        dry_run=dry_run,
        action=action,
        metadata={"label": target.label},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTGS fault injection daemon")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--policy", choices=["random", "round_robin", "specific"], default="random")
    parser.add_argument("--target-rank", type=int, default=None)
    parser.add_argument("--protect-rank", action="append", type=int, default=[0])
    parser.add_argument("--rank-pid", action="append", default=[], help="rank:pid mapping")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-path", default="experiments/results/fault_injection.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processes = [parse_rank_pid(item) for item in args.rank_pid]
    if not processes:
        raise SystemExit("--rank-pid is required, e.g. --rank-pid 1:12345")
    logger = JsonlLogger(Path(args.log_path))
    rng = random.Random(args.seed)

    for iteration in range(args.iterations):
        if iteration > 0:
            time.sleep(args.interval)
        target = select_target(
            processes,
            policy=args.policy,
            iteration=iteration,
            protected_ranks=set(args.protect_rank),
            specific_rank=args.target_rank,
            rng=rng,
        )
        event = inject_fault(
            target,
            policy=args.policy,
            dry_run=args.dry_run,
            protected_pids={os.getpid()},
        )
        logger.log("fault_injected", **asdict(event))
        print(asdict(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
