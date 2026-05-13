"""Distributed training entrypoint for baseline and MTGS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .baseline import (
    BaselineModelConfig,
    SyntheticTokenDataset,
    build_model,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from .config import SeedConfig
from .dataloader import build_distributed_sampler
from .hooks.comm_hook import MTGSState, register_mtgs_comm_hook
from .hooks.transaction import TransactionManager
from .profiling.ettr_timer import ETTRTimer
from .profiling.throughput import ThroughputLogger
from .shadow.copy_stream import ShadowCopyManager
from .repro import set_seed
from .utils.distributed import cleanup_distributed, init_distributed
from .utils.logging import JsonlLogger


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required to run training") from exc
    return torch


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTGS distributed training runner")
    parser.add_argument("--mode", choices=["baseline", "mtgs"], default="baseline")
    parser.add_argument("--model-name", default="synthetic")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="experiments/results/local_baseline")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--ettr-path", default="")
    parser.add_argument("--mtgs-disable-hook", action="store_true")
    parser.add_argument("--mtgs-disable-shadow", action="store_true")
    parser.add_argument("--mtgs-disable-2pc", action="store_true")
    parser.add_argument("--mtgs-debug", action="store_true")
    parser.add_argument(
        "--mtgs-force-abort-step",
        type=int,
        default=0,
        help="Inject a local MTGS abort decision at the selected step for tests.",
    )
    return parser.parse_args(argv)


def resolve_device(torch_mod: Any, requested: str, local_rank: int) -> Any:
    if requested == "cuda" or (requested == "auto" and torch_mod.cuda.is_available()):
        torch_mod.cuda.set_device(local_rank)
        return torch_mod.device("cuda", local_rank)
    return torch_mod.device("cpu")


def make_dataloader(args: argparse.Namespace, context: Any) -> Any:
    torch_mod = _require_torch()
    dataset = SyntheticTokenDataset(
        size=args.dataset_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    sampler = None
    shuffle = True
    if context.world_size > 1:
        sampler = build_distributed_sampler(
            dataset,
            world_size=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        shuffle = False
    return torch_mod.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
    )


def _infinite_batches(loader: Any) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def run(args: argparse.Namespace) -> int:
    torch_mod = _require_torch()
    set_seed(SeedConfig(seed=args.seed))
    context = init_distributed(args.backend)
    device = resolve_device(torch_mod, args.device, context.local_rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ThroughputLogger(
        output_dir / f"throughput_rank{context.rank}.csv",
        rank=context.rank,
        warmup_steps=args.warmup_steps,
    )
    events = JsonlLogger(output_dir / f"train_rank{context.rank}.jsonl")
    ettr_timer = ETTRTimer()
    ettr_path = Path(args.ettr_path) if args.ettr_path else output_dir / "ettr_events.csv"
    events.log(
        "run_start",
        mode=args.mode,
        rank=context.rank,
        world_size=context.world_size,
        backend=context.backend,
        device=str(device),
    )

    model_config = BaselineModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        max_seq_length=args.seq_length,
        learning_rate=args.learning_rate,
    )
    model = build_model(model_config, args.model_name).to(device)
    optimizer = build_optimizer(model, args.learning_rate)
    scheduler = torch_mod.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    if args.resume and args.checkpoint_path:
        state = load_checkpoint(
            args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=str(device),
        )
        events.log("checkpoint_loaded", path=args.checkpoint_path, step=state["step"])

    mtgs_state: MTGSState | None = None
    if context.initialized:
        ddp = torch_mod.nn.parallel.DistributedDataParallel
        model = ddp(model, device_ids=[context.local_rank] if device.type == "cuda" else None)
        if args.mode == "mtgs":
            mtgs_state = MTGSState(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                process_group=None,
                enable_hook=not args.mtgs_disable_hook,
                enable_shadow=not args.mtgs_disable_shadow,
                enable_2pc=not args.mtgs_disable_2pc,
                debug=args.mtgs_debug,
                rank=context.rank,
                shadow_manager=ShadowCopyManager(pin_memory=device.type == "cuda"),
                transaction_manager=TransactionManager(process_group=None),
                logger=events,
            )
            register_mtgs_comm_hook(model, mtgs_state)
            events.log(
                "mtgs_hook_registered",
                rank=context.rank,
                enable_hook=mtgs_state.enable_hook,
                enable_shadow=mtgs_state.enable_shadow,
                enable_2pc=mtgs_state.enable_2pc,
            )
    elif args.mode == "mtgs":
        mtgs_state = MTGSState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            enable_hook=not args.mtgs_disable_hook,
            enable_shadow=not args.mtgs_disable_shadow,
            enable_2pc=not args.mtgs_disable_2pc,
            debug=args.mtgs_debug,
            rank=context.rank,
            shadow_manager=ShadowCopyManager(pin_memory=False),
            logger=events,
        )

    loader = make_dataloader(args, context)
    batches = _infinite_batches(loader)
    losses: list[float] = []

    for step in range(1, args.steps + 1):
        if mtgs_state is not None:
            mtgs_state.start_step(step)
            if not context.initialized:
                mtgs_state.snapshot_if_needed()
                mtgs_state.force_abort = args.mtgs_force_abort_step == step

        start_ns = metrics.now_ns()
        batch = next(batches)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        if mtgs_state is not None and not context.initialized and mtgs_state.force_abort:
            event_id = f"rank{context.rank}-step{step}-forced-abort"
            ettr_timer.mark_detected(
                event_id,
                rank=context.rank,
                step=step,
                reason="forced_abort",
            )
            mtgs_state.rollback("forced_abort")
            events.log("step_aborted", rank=context.rank, step=step, reason="forced_abort")
            optimizer.zero_grad(set_to_none=True)
            mtgs_state.complete_step()
            event = ettr_timer.mark_resumed(event_id)
            ettr_timer.write_events(ettr_path)
            events.log(
                "ettr_recorded",
                rank=context.rank,
                step=step,
                event_id=event.event_id,
                ettr_ms=event.ettr_ms,
            )
            continue

        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        tokens = int(input_ids.numel())
        record = metrics.record_step(
            epoch=0,
            step=step,
            tokens=tokens,
            start_ns=start_ns,
            loss=loss_value,
        )
        events.log(
            "step_complete",
            rank=context.rank,
            step=step,
            loss=loss_value,
            tokens_per_second=record.tokens_per_second,
        )

        if (
            args.checkpoint_path
            and args.checkpoint_every > 0
            and step % args.checkpoint_every == 0
        ):
            save_checkpoint(
                args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                step=step,
                extra={"mode": args.mode},
            )
            events.log("checkpoint_saved", path=args.checkpoint_path, step=step)

        if mtgs_state is not None:
            mtgs_state.complete_step()

    if args.checkpoint_path and args.checkpoint_every == 0:
        save_checkpoint(
            args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            step=args.steps,
            extra={"mode": args.mode},
        )
        events.log("checkpoint_saved", path=args.checkpoint_path, step=args.steps)

    events.log(
        "run_complete",
        rank=context.rank,
        steps=args.steps,
        first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None,
    )
    (output_dir / "ettr_summary.json").write_text(
        json.dumps(ettr_timer.summary(), indent=2),
        encoding="utf-8",
    )
    cleanup_distributed()
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
