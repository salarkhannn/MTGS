#!/usr/bin/env python3
"""Validate baseline/MTGS training CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MTGS training outputs")
    parser.add_argument("--throughput-csv", required=True)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--require-loss-decrease", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def validate_training_csv(
    path: Path,
    min_steps: int,
    require_loss_decrease: bool,
) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    errors: list[str] = []
    if len(rows) < min_steps:
        errors.append(f"expected at least {min_steps} rows, found {len(rows)}")

    losses = [float(row["loss"]) for row in rows if row.get("loss")]
    measured = [
        float(row["tokens_per_second"])
        for row in rows
        if float(row.get("tokens_per_second", "0") or 0) > 0
    ]
    if not measured:
        errors.append("no warmup-excluded throughput values were recorded")

    if require_loss_decrease and len(losses) >= 2 and losses[-1] > losses[0]:
        errors.append(f"loss did not decrease: first={losses[0]}, last={losses[-1]}")

    return {
        "path": str(path),
        "rows": len(rows),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_measured_tokens_per_second": (
            sum(measured) / len(measured) if measured else 0.0
        ),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    result = validate_training_csv(
        Path(args.throughput_csv),
        min_steps=args.min_steps,
        require_loss_decrease=args.require_loss_decrease,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
