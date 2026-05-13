#!/usr/bin/env python3
"""Generate churn scenario commands and summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CHURN_LEVELS = {
    "low": 600,
    "medium": 300,
    "high": 120,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTGS churn simulation wrapper")
    parser.add_argument("--output-dir", default="experiments/results/churn")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--level", choices=sorted(CHURN_LEVELS), action="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_scenario(level: str, output_dir: Path, duration_minutes: float) -> dict[str, object]:
    interval = CHURN_LEVELS[level]
    return {
        "level": level,
        "kill_interval_s": interval,
        "target_policy": "round_robin",
        "duration_minutes": duration_minutes,
        "trainer_command": [
            sys.executable,
            "-m",
            "mtgs.trainer",
            "--mode",
            "mtgs",
            "--steps",
            "10",
            "--output-dir",
            str(output_dir / level / "trainer"),
        ],
        "injector_command": [
            sys.executable,
            "-m",
            "mtgs.fault.injector",
            "--interval",
            str(interval),
            "--iterations",
            "1",
            "--policy",
            "round_robin",
            "--dry-run",
            "--rank-pid",
            "1:99999",
            "--log-path",
            str(output_dir / level / "fault_injection.jsonl"),
        ],
    }


def main() -> int:
    args = parse_args()
    levels = args.level or ["low", "medium", "high"]
    output_dir = Path(args.output_dir)
    scenarios = [
        build_scenario(level, output_dir, args.duration_minutes)
        for level in levels
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "churn_summary.json"
    summary_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")

    for scenario in scenarios:
        print(json.dumps(scenario))
        if not args.dry_run:
            subprocess.run(scenario["trainer_command"], check=True)
            subprocess.run(scenario["injector_command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
