#!/usr/bin/env python3
"""Parse run outputs and generate comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MTGS experiment results")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="docs/figures")
    parser.add_argument("--table-path", default="docs/results_summary.md")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    mode = "unknown"
    fault_profile = "unknown"
    nodes = 1
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("mode:"):
                mode = line.split(":", 1)[1].strip()
            if line.startswith("fault_profile:"):
                fault_profile = line.split(":", 1)[1].strip()
            if line.startswith("nodes:"):
                nodes = int(line.split(":", 1)[1].strip())

    throughput_files = sorted(run_dir.glob("throughput_rank*.csv"))
    rows: list[dict[str, str]] = []
    for path in throughput_files:
        rows.extend(_read_csv(path))
    measured = [
        float(row["tokens_per_second"])
        for row in rows
        if float(row.get("tokens_per_second", "0") or 0) > 0
    ]
    losses = [float(row["loss"]) for row in rows if row.get("loss")]

    ettr_summary_path = run_dir / "ettr_summary.json"
    ettr = {"median_ms": 0.0, "p95_ms": 0.0, "worst_ms": 0.0, "count": 0.0}
    if ettr_summary_path.exists():
        ettr.update(json.loads(ettr_summary_path.read_text(encoding="utf-8")))

    return {
        "run_id": run_dir.name,
        "mode": mode,
        "fault_profile": fault_profile,
        "nodes": nodes,
        "mean_tokens_per_second": sum(measured) / len(measured) if measured else 0.0,
        "first_loss": losses[0] if losses else math.nan,
        "last_loss": losses[-1] if losses else math.nan,
        "ettr_count": ettr["count"],
        "ettr_median_ms": ettr["median_ms"],
        "ettr_p95_ms": ettr["p95_ms"],
        "ettr_worst_ms": ettr["worst_ms"],
    }


def compute_improvements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = [
        row["mean_tokens_per_second"]
        for row in rows
        if row["mode"] == "baseline" and row["mean_tokens_per_second"] > 0
    ]
    baseline_tps = baselines[0] if baselines else 0.0
    one_node_by_mode: dict[str, float] = {}
    for row in rows:
        if row["nodes"] == 1 and row["mean_tokens_per_second"] > 0:
            one_node_by_mode.setdefault(row["mode"], row["mean_tokens_per_second"])

    enriched: list[dict[str, Any]] = []
    for row in rows:
        degradation = 0.0
        if baseline_tps and row["mode"] == "mtgs":
            degradation = ((baseline_tps - row["mean_tokens_per_second"]) / baseline_tps) * 100
        one_node = one_node_by_mode.get(row["mode"], row["mean_tokens_per_second"])
        scaling_efficiency = (
            row["mean_tokens_per_second"] / (row["nodes"] * one_node)
            if one_node
            else 0.0
        )
        enriched.append(
            {
                **row,
                "throughput_degradation_pct": degradation,
                "scaling_efficiency": scaling_efficiency,
            }
        )
    return enriched


def write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Run | Mode | Fault | Mean tokens/s | Loss first->last | "
        "ETTR median ms | Throughput degradation % |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    body = ""
    for row in rows:
        body += (
            f"| {row['run_id']} | {row['mode']} | {row['fault_profile']} | "
            f"{row['mean_tokens_per_second']:.2f} | "
            f"{row['first_loss']:.4f}->{row['last_loss']:.4f} | "
            f"{row['ettr_median_ms']:.2f} | "
            f"{row['throughput_degradation_pct']:.2f} |\n"
        )
    path.write_text("# Local Smoke Results\n\n" + header + body, encoding="utf-8")


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    labels = [row["run_id"] for row in rows]
    throughput = [row["mean_tokens_per_second"] for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, throughput, color=["#3b82f6" if row["mode"] == "baseline" else "#10b981" for row in rows])
    plt.ylabel("Mean tokens/s")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_comparison.png", dpi=160)
    plt.close()

    ettr_rows = [row for row in rows if row["ettr_count"]]
    if ettr_rows:
        plt.figure(figsize=(8, 4))
        plt.bar(
            [row["run_id"] for row in ettr_rows],
            [row["ettr_median_ms"] for row in ettr_rows],
            color="#f97316",
        )
        plt.ylabel("Median ETTR (ms)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "ettr_comparison.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(
        [row["fault_profile"] for row in rows],
        [row["mean_tokens_per_second"] for row in rows],
        color="#6366f1",
    )
    plt.ylabel("Mean tokens/s")
    plt.xlabel("Fault profile")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_churn.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda item: item["nodes"])
        plt.plot(
            [row["nodes"] for row in mode_rows],
            [row["scaling_efficiency"] for row in mode_rows],
            marker="o",
            label=mode,
        )
    plt.ylabel("Scaling efficiency")
    plt.xlabel("Nodes")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_efficiency.png", dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = [
        summarize_run(run_dir)
        for run_dir in sorted(results_dir.iterdir())
        if run_dir.is_dir()
    ]
    rows = compute_improvements(rows)
    write_markdown_table(rows, Path(args.table_path))
    write_plots(rows, Path(args.output_dir))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
