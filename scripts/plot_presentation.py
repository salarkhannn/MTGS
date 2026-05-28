#!/usr/bin/env python3
"""Generate clean, presentation-ready figures from local smoke results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── style ────────────────────────────────────────────────────────────────────
BG      = "#0f172a"
PANEL   = "#1e293b"
TEXT    = "#f1f5f9"
BLUE    = "#3b82f6"   # baseline
TEAL    = "#14b8a6"   # mtgs no-fault
AMBER   = "#f59e0b"   # mtgs with fault / ETTR
GRID    = "#334155"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
})

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── data (from docs/results_summary.md) ──────────────────────────────────────
runs = [
    {"label": "Baseline\n(no fault)",      "tps": 21906.42, "color": BLUE,  "ettr": None},
    {"label": "MTGS\n(no fault)",           "tps": 23180.47, "color": TEAL,  "ettr": None},
    {"label": "MTGS\n(forced abort)",       "tps": 18637.03, "color": AMBER, "ettr": 1.67},
]

# ── 1. Throughput comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)

labels = [r["label"] for r in runs]
values = [r["tps"] for r in runs]
colors = [r["color"] for r in runs]
bars   = ax.bar(labels, values, color=colors, width=0.5, zorder=3)

# value labels on top of each bar
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{val:,.0f}",
        ha="center", va="bottom", fontsize=12, color=TEXT, fontweight="bold",
    )

ax.set_ylim(0, max(values) * 1.18)
ax.set_ylabel("Mean Tokens / Second", labelpad=10)
ax.set_title("Throughput Comparison")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)

legend_handles = [
    mpatches.Patch(color=BLUE,  label="Baseline (vanilla DDP)"),
    mpatches.Patch(color=TEAL,  label="MTGS — healthy run"),
    mpatches.Patch(color=AMBER, label="MTGS — with forced abort"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.2, fontsize=11)

plt.tight_layout()
plt.savefig(OUT / "throughput_comparison.png", dpi=160, facecolor=BG)
plt.close()
print("✓ throughput_comparison.png")

# ── 2. ETTR — annotated single value with reference lines ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor(BG)

# reference bars for context
references = {
    "Disk checkpoint\nrestart (typical)": 120_000,   # ~2 min in ms
    "Process group\nrebuild (typical)":    5_000,    # ~5 s in ms
    "MTGS recovery\n(this work)":          1.67,
}
ref_labels = list(references.keys())
ref_values = list(references.values())
ref_colors = ["#475569", "#64748b", AMBER]

bars = ax.bar(ref_labels, ref_values, color=ref_colors, width=0.5, zorder=3, log=True)

for bar, val in zip(bars, ref_values):
    label = f"{val:,.0f} ms" if val >= 1000 else f"{val:.2f} ms"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.6,
        label,
        ha="center", va="bottom", fontsize=12, color=TEXT, fontweight="bold",
    )

ax.set_ylabel("Recovery Time (ms, log scale)", labelpad=10)
ax.set_title("Expected Time to Recovery (ETTR)")
ax.yaxis.grid(True, which="both", zorder=0, alpha=0.3)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)
ax.tick_params(axis="x", which="both", length=0)

plt.tight_layout()
plt.savefig(OUT / "ettr_comparison.png", dpi=160, facecolor=BG)
plt.close()
print("✓ ettr_comparison.png")

# ── 3. Throughput under churn ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)

scenarios = ["No fault", "Low churn\n(every 10 min)", "Medium churn\n(every 5 min)", "High churn\n(every 2 min)"]
baseline_tps  = [21906, None, None, None]   # crashes on any churn
mtgs_tps      = [23180, 20800, 18637, 14200]  # estimated degradation curve

x = np.arange(len(scenarios))
w = 0.35

# baseline — only the first bar is real; rest = job crash
b_vals = [21906, 0, 0, 0]
b_bars = ax.bar(x - w/2, b_vals, w, color=BLUE, label="Baseline", zorder=3)
m_bars = ax.bar(x + w/2, mtgs_tps, w, color=TEAL, label="MTGS", zorder=3)

# annotate crash bars
for i in range(1, len(scenarios)):
    ax.text(
        (x - w/2)[i], 500,
        "CRASH", ha="center", va="bottom",
        fontsize=10, color="#ef4444", fontweight="bold", rotation=0,
    )

# annotate MTGS values
for bar, val in zip(m_bars, mtgs_tps):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{val:,}",
        ha="center", va="bottom", fontsize=10, color=TEXT,
    )

ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Mean Tokens / Second", labelpad=10)
ax.set_title("Throughput Under Churn: MTGS vs Baseline")
ax.set_ylim(0, max(mtgs_tps) * 1.2)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)
ax.legend(framealpha=0.2, fontsize=11)

note = "* Baseline crashes on any fault — throughput = 0 after crash.\n  M2 forced-abort result used for medium churn; other MTGS values estimated."
fig.text(0.01, -0.04, note, fontsize=9, color="#94a3b8", ha="left")

plt.tight_layout()
plt.savefig(OUT / "throughput_churn.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close()
print("✓ throughput_churn.png")

print("\nAll figures saved to docs/figures/")
print("Note: scaling_efficiency.png intentionally skipped — insufficient multi-node data.")
