#!/usr/bin/env python3
"""Experiment matrix launcher for baseline and MTGS runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mtgs.repro import environment_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTGS experiment matrix")
    parser.add_argument("--config", default="experiments/configs/full_matrix.yaml")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_run_id(item: dict[str, Any], repeat: int) -> str:
    return f"{item['id']}_{item['mode']}_{item.get('nodes', 1)}n_rep{repeat}"


def build_command(item: dict[str, Any], run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mtgs.trainer",
        "--mode",
        item["mode"],
        "--steps",
        str(item.get("steps", 10)),
        "--dataset-size",
        str(item.get("dataset_size", 64)),
        "--batch-size",
        str(item.get("batch_size", 4)),
        "--seq-length",
        str(item.get("seq_length", 16)),
        "--vocab-size",
        str(item.get("vocab_size", 64)),
        "--hidden-size",
        str(item.get("hidden_size", 32)),
        "--device",
        item.get("device", "cpu"),
        "--output-dir",
        str(run_dir),
    ]
    if item.get("mode") == "mtgs" and item.get("force_abort_step", 0):
        command.extend(["--mtgs-force-abort-step", str(item["force_abort_step"])])
    return command


def write_run_metadata(run_dir: Path, item: dict[str, Any], command: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(item, sort_keys=False), encoding="utf-8")
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (run_dir / "env_fingerprint.json").write_text(
        json.dumps(environment_fingerprint(), indent=2),
        encoding="utf-8",
    )


def expand_matrix(config: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    runs: list[tuple[dict[str, Any], int]] = []
    for item in config.get("runs", []):
        repetitions = int(item.get("repetitions", 1))
        for repeat in range(1, repetitions + 1):
            runs.append((item, repeat))
    return runs


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    runs = expand_matrix(config)
    if args.limit:
        runs = runs[: args.limit]

    output_dir = Path(args.output_dir)
    for item, repeat in runs:
        run_id = build_run_id(item, repeat)
        run_dir = output_dir / run_id
        command = build_command(item, run_dir)
        write_run_metadata(run_dir, {**item, "repeat": repeat, "run_id": run_id}, command)
        if args.dry_run:
            print(json.dumps({"run_id": run_id, "command": command}))
            continue

        started = time.time()
        result = subprocess.run(command, check=False, text=True)
        status = {
            "run_id": run_id,
            "returncode": result.returncode,
            "started_at": started,
            "ended_at": time.time(),
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
