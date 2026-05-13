#!/usr/bin/env python3
"""Validate the D3 four-node environment plan.

The checks are intentionally local and deterministic: they verify the planned
topology before any cloud spend, then optionally write a JSON validation record
that can be archived with the experiment run.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PORTS = {22, 29500, 29501}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MTGS cluster topology")
    parser.add_argument("--topology", default="infra/cluster_topology.yaml")
    parser.add_argument("--expected-nodes", type=int, default=4)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_topology(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_topology(topology: dict[str, Any], expected_nodes: int) -> dict[str, Any]:
    nodes = topology.get("nodes", [])
    ports = set(topology.get("network", {}).get("open_tcp_ports", []))
    cidr = ipaddress.ip_network(topology.get("network", {}).get("private_cidr"))

    errors: list[str] = []
    if len(nodes) != expected_nodes:
        errors.append(f"expected {expected_nodes} nodes, found {len(nodes)}")

    missing_ports = sorted(REQUIRED_PORTS - ports)
    if missing_ports:
        errors.append(f"missing required TCP ports: {missing_ports}")

    ranks = [node.get("rank") for node in nodes]
    if ranks != list(range(expected_nodes)):
        errors.append(f"ranks must be contiguous 0..{expected_nodes - 1}, found {ranks}")

    hostnames = [node.get("hostname") for node in nodes]
    if len(set(hostnames)) != len(hostnames):
        errors.append("hostnames must be unique")

    ips = [node.get("private_ip") for node in nodes]
    if len(set(ips)) != len(ips):
        errors.append("private IPs must be unique")
    for ip in ips:
        if ipaddress.ip_address(ip) not in cidr:
            errors.append(f"private IP {ip} is outside {cidr}")

    images = {node.get("machine_image") for node in nodes}
    gpu_types = {node.get("gpu_type") for node in nodes}
    if len(images) != 1:
        errors.append(f"machine images must match across nodes, found {sorted(images)}")
    if len(gpu_types) != 1:
        errors.append(f"GPU types must match across nodes, found {sorted(gpu_types)}")

    return {
        "provider": topology.get("provider"),
        "region": topology.get("region"),
        "zone": topology.get("zone"),
        "expected_nodes": expected_nodes,
        "ports": sorted(ports),
        "hostnames": hostnames,
        "private_ips": ips,
        "gpu_type": next(iter(gpu_types), None),
        "machine_image": next(iter(images), None),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    topology = load_topology(Path(args.topology))
    result = validate_topology(topology, args.expected_nodes)
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
