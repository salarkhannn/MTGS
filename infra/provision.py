#!/usr/bin/env python3
"""Minimal, provider-agnostic provisioning template for MTGS.

This script is intentionally simple so it can be validated locally. It records
provisioning intent to a state file to keep runs idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTGS provisioning template")
    parser.add_argument("--provider", default="generic", help="Cloud provider name")
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--gpu-type", default="t4")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--name-prefix", default="mtgs")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--control-port", type=int, default=29501)
    parser.add_argument(
        "--extra-ports",
        default="",
        help="Comma-separated additional TCP ports to open",
    )
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_ports(port_list: str) -> list[int]:
    ports: list[int] = []
    for item in port_list.split(","):
        value = item.strip()
        if not value:
            continue
        ports.append(int(value))
    return ports


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    ports = [args.ssh_port, args.nccl_port, args.control_port]
    ports.extend(parse_ports(args.extra_ports))
    return {
        "provider": args.provider,
        "node_count": args.node_count,
        "gpu_type": args.gpu_type,
        "region": args.region,
        "name_prefix": args.name_prefix,
        "firewall": {
            "protocol": "tcp",
            "ports": sorted(set(ports)),
        },
    }


def main() -> int:
    args = parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    state_dir = Path(args.state_dir) if args.state_dir else root_dir / "infra" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"{args.name_prefix}-nodes.json"

    if state_file.exists() and not args.force:
        print(f"State file already exists: {state_file}")
        print("Provisioning skipped. Use --force to overwrite.")
        return 0

    plan = build_plan(args)
    print("Provisioning plan:")
    print(json.dumps(plan, indent=2))

    if args.provider != "generic":
        print("Provider-specific provisioning is not implemented in this template.")
        print("Set --provider generic for a local dry-run plan only.")
        return 1

    if args.dry_run:
        print("Dry run complete. No state file written.")
        return 0

    state_file.write_text(json.dumps({"plan": plan}, indent=2), encoding="utf-8")
    print(f"State written to {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
