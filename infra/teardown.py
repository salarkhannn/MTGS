#!/usr/bin/env python3
"""Minimal, provider-agnostic teardown template for MTGS.

For the generic provider, this removes the local state file to indicate that
resources would be released by a real provider-specific implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTGS teardown template")
    parser.add_argument("--provider", default="generic", help="Cloud provider name")
    parser.add_argument("--name-prefix", default="mtgs")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    state_dir = Path(args.state_dir) if args.state_dir else root_dir / "infra" / "state"
    state_file = state_dir / f"{args.name_prefix}-nodes.json"

    if not state_file.exists():
        print(f"No state file found at {state_file}")
        print("Nothing to tear down.")
        return 0

    plan = json.loads(state_file.read_text(encoding="utf-8")).get("plan", {})
    print("Teardown plan:")
    print(json.dumps(plan, indent=2))

    if args.provider != "generic":
        print("Provider-specific teardown is not implemented in this template.")
        return 1

    if args.dry_run:
        print("Dry run complete. State file kept.")
        return 0

    if not args.force:
        print("Refusing to remove state without --force.")
        return 1

    state_file.unlink()
    print(f"State file removed: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
