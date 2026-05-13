#!/usr/bin/env python3
"""Single-node multi-process MTGS DDP smoke test.

This avoids torchrun's elastic TCP rendezvous so it works on Windows CPU-only
builds that do not include libuv support.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTGS DDP smoke test")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    return parser.parse_args()


def _worker(
    rank: int,
    world_size: int,
    init_uri: str,
    seq_length: int,
    vocab_size: int,
    hidden_size: int,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

    import torch
    import torch.distributed as dist

    from mtgs.baseline import BaselineModelConfig, build_model, build_optimizer
    from mtgs.hooks.comm_hook import MTGSState, register_mtgs_comm_hook
    from mtgs.hooks.transaction import TransactionDecision, TransactionManager
    from mtgs.shadow.copy_stream import ShadowCopyManager

    dist.init_process_group(
        backend="gloo",
        init_method=init_uri,
        rank=rank,
        world_size=world_size,
    )
    model = build_model(
        BaselineModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_length=seq_length,
        )
    )
    optimizer = build_optimizer(model, learning_rate=1e-3)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    state = MTGSState(
        model=ddp,
        optimizer=optimizer,
        rank=rank,
        shadow_manager=ShadowCopyManager(pin_memory=False),
        transaction_manager=TransactionManager(),
    )
    register_mtgs_comm_hook(ddp, state)
    state.start_step(1)

    generator = torch.Generator().manual_seed(100 + rank)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, seq_length),
        generator=generator,
    )
    outputs = ddp(input_ids=input_ids, labels=input_ids)
    outputs["loss"].backward()
    optimizer.step()

    assert state.last_transaction is not None
    assert state.last_transaction.decision == TransactionDecision.COMMIT
    dist.destroy_process_group()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "mtgs-ddp-store"
        init_uri = store_path.as_uri()
        try:
            mp.spawn(
                _worker,
                args=(
                    args.world_size,
                    init_uri,
                    args.seq_length,
                    args.vocab_size,
                    args.hidden_size,
                ),
                nprocs=args.world_size,
                join=True,
            )
        except Exception as exc:
            message = str(exc)
            if "unsupported gloo device" in message:
                print(f"MTGS DDP smoke skipped: {message}")
                return 0
            raise
    print(f"MTGS DDP smoke passed with world_size={args.world_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
