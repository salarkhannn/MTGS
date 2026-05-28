#!/usr/bin/env python3
"""One-command local verification for MTGS."""

from __future__ import annotations

import argparse
import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "pytest", "tests", "-q"],
    [
        sys.executable,
        "-m",
        "mtgs.trainer",
        "--mode",
        "baseline",
        "--steps",
        "2",
        "--dataset-size",
        "16",
        "--batch-size",
        "4",
        "--seq-length",
        "8",
        "--vocab-size",
        "32",
        "--hidden-size",
        "16",
        "--device",
        "cpu",
        "--output-dir",
        "experiments/results/verify_baseline",
    ],
    [
        sys.executable,
        "-m",
        "mtgs.trainer",
        "--mode",
        "mtgs",
        "--steps",
        "2",
        "--dataset-size",
        "16",
        "--batch-size",
        "4",
        "--seq-length",
        "8",
        "--vocab-size",
        "32",
        "--hidden-size",
        "16",
        "--device",
        "cpu",
        "--output-dir",
        "experiments/results/verify_mtgs",
        "--mtgs-force-abort-step",
        "1",
    ],
    [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        "experiments/configs/full_matrix.yaml",
        "--output-dir",
        "experiments/results/verify_matrix",
        "--dry-run",
        "--limit",
        "2",
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local D3 verification")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = COMMANDS[1:] if args.skip_tests else COMMANDS
    for command in commands:
        print("+ " + " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
