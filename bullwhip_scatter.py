#!/usr/bin/env python
"""
bullwhip_scatter.py
====================
Generate 8 scatter plots demonstrating the Bullwhip Effect
in a 2-echelon, multi-DC supply chain.

Topology:  Supplier → 2 DCs → 15 Retailers
  DC 0 → Retailers [R_2 … R_8]   (agent_ids 2–8,  retailer indices 0–6)
  DC 1 → Retailers [R_9 … R_16]  (agent_ids 9–16, retailer indices 7–14)

Data source:  step_trajectory_ep1.xlsx inside each model's eval directory.
  Columns used per SKU k:  demand_{k}, order_{k}   (k ∈ {0, 1, 2})

Output (saved to  bullwhip_output/):
  Task 1 — Retailer Level (4 images):
    retailer_scatter_<model>.png
  Task 2 — DC Level (4 images):
    dc_scatter_<model>.png

Usage:
    python bullwhip_scatter.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Path to each model's step-trajectory Excel (episode 1)
XLSX_PATHS: dict[str, str] = {
    "GNN-HAPPO":       "evaluation_results/eval_gnn/step_trajectory_ep1.xlsx",
    "Standard HAPPO":  "evaluation_results/eval_base/step_trajectory_ep1.xlsx",
    "MAPPO":           "evaluation_results/eval_mappo/step_trajectory_ep1.xlsx",
    "S-s Heuristic":   "evaluation_results/basestock_v1/step_trajectory_ep1.xlsx",
}

# Color palette  (one per model, used in both tasks)
COLORS: dict[str, str] = {
    "S-s Heuristic":   "#888888",   # Gray
    "MAPPO":           "#E8880A",   # Orange
    "Standard HAPPO":  "#D03030",   # Red
    "GNN-HAPPO":       "#2E6FBB",   # Blue
}

# DC → retailer agent-id mapping  (from config: dc_0=[0..6], dc_1=[7..14]
# agent_ids = n_dcs + retailer_index, n_dcs=2)
DC_ASSIGNMENTS: dict[int, list[int]] = {
    0: [2, 3, 4, 5, 6, 7, 8],       # DC 0 serves agent_ids 2–8
    1: [9, 10, 11, 12, 13, 14, 15, 16],  # DC 1 serves agent_ids 9–16
}

N_SKUS = 3
EVAL_DAYS = 90          # use only the first 90 steps
OUT_DIR = Path("bullwhip_output")

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_trajectory(path: str) -> pd.DataFrame:
    """Load step_trajectory xlsx; validate it exists."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p}")
        sys.exit(1)
    df = pd.read_excel(p)
    return df


def _agent_daily_series(df: pd.DataFrame, agent_id: int,
                         max_steps: int = EVAL_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (total_demand, total_orders) arrays of length ≤ max_steps
    for a single agent, summed across all SKUs.
    """
    sub = df[(df["agent_id"] == agent_id) & (df["step"] <= max_steps)].sort_values("step")
    demand_cols = [f"demand_{k}" for k in range(N_SKUS)]
    order_cols  = [f"order_{k}"  for k in range(N_SKUS)]
    demand = sub[demand_cols].sum(axis=1).values
    orders = sub[order_cols].sum(axis=1).values
    return demand, orders


def _safe_cv(arr: np.ndarray) -> float:
    """Coefficient of Variation = std / mean.  Returns 0 if mean ≈ 0."""
    m = np.mean(arr)
    if abs(m) < 1e-9:
        return 0.0
    return float(np.std(arr, ddof=1) / m)


# ═══════════════════════════════════════════════════════════════════════════
# Task 1  —  Retailer-Level Demand vs. Order Volatility
# ═══════════════════════════════════════════════════════════════════════════

def plot_retailer_scatter(model_name: str, df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each of the 15 retailers compute:
        x = std(daily customer demand)
        y = std(daily retailer order quantity)
    and create a scatter plot with a y=x reference line.
    """
    retailer_ids = sorted(
        [aid for aids in DC_ASSIGNMENTS.values() for aid in aids]
    )

    x_vals, y_vals = [], []
    for rid in retailer_ids:
        demand, orders = _agent_daily_series(df, rid)
        x_vals.append(np.std(demand, ddof=1))
        y_vals.append(np.std(orders, ddof=1))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    color = COLORS.get(model_name, "#333333")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # y = x reference
    lim = max(x_vals.max(), y_vals.max()) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#999999", zorder=1,
            label="y = x  (no amplification)")

    # Scatter
    ax.scatter(x_vals, y_vals, s=80, c=color, edgecolors="white",
               linewidths=0.6, alpha=0.85, zorder=2)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Std-Dev of Customer Demand", fontsize=12)
    ax.set_ylabel("Std-Dev of Retailer Order Qty", fontsize=12)
    ax.set_title(f"Retailer Level: Demand vs. Order Volatility - {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fname = out_dir / f"retailer_scatter_{model_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {fname}")


def plot_retailer_scatter_combined(all_dfs: dict[str, pd.DataFrame],
                                   out_dir: Path) -> None:
    """
    Single scatter plot with all 4 models overlaid.
    Each model's 15 retailers are plotted in the model's colour.
        x = std(daily customer demand)
        y = std(daily retailer order quantity)
    """
    retailer_ids = sorted(
        [aid for aids in DC_ASSIGNMENTS.values() for aid in aids]
    )

    fig, ax = plt.subplots(figsize=(7.5, 7))

    global_max = 0.0
    for model_name, df in all_dfs.items():
        x_vals, y_vals = [], []
        for rid in retailer_ids:
            demand, orders = _agent_daily_series(df, rid)
            x_vals.append(np.std(demand, ddof=1))
            y_vals.append(np.std(orders, ddof=1))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        global_max = max(global_max, x_vals.max(), y_vals.max())

        color = COLORS.get(model_name, "#333333")
        ax.scatter(x_vals, y_vals, s=80, c=color, edgecolors="white",
                   linewidths=0.6, alpha=0.80, zorder=2, label=model_name)

    # y = x reference line
    lim = global_max * 1.15
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#999999", zorder=1,
            label="y = x  (no amplification)")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Customer Demand (units)", fontsize=12)
    ax.set_ylabel("Retailer Order Quantity (units)", fontsize=12)
    ax.set_title("Retailer Level: Demand vs. Order Volatility - All Models",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fname = out_dir / "retailer_scatter_ALL_MODELS.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Task 2  —  DC-Level Demand CV vs. Order CV
# ═══════════════════════════════════════════════════════════════════════════

def plot_dc_scatter(model_name: str, df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each DC:
        Aggregate Downstream Demand = sum of demand from its assigned retailers
        DC Order Qty                = order placed by the DC to the supplier
    Normalise volatility via CV = std / mean.
        x = CV(aggregate downstream demand)
        y = CV(DC order qty)
    Scatter both DCs on the same plot.
    """
    color = COLORS.get(model_name, "#333333")

    x_vals, y_vals, labels = [], [], []

    for dc_id, retailer_ids in DC_ASSIGNMENTS.items():
        # Aggregate downstream demand (sum of all assigned retailers' demand per day)
        agg_demand = np.zeros(EVAL_DAYS)
        for rid in retailer_ids:
            demand, _ = _agent_daily_series(df, rid)
            # Trim/pad to EVAL_DAYS in case of length mismatch
            n = min(len(demand), EVAL_DAYS)
            agg_demand[:n] += demand[:n]

        # DC's own order to supplier
        _, dc_orders = _agent_daily_series(df, dc_id)
        n = min(len(dc_orders), EVAL_DAYS)
        dc_orders_padded = np.zeros(EVAL_DAYS)
        dc_orders_padded[:n] = dc_orders[:n]

        cv_demand = _safe_cv(agg_demand)
        cv_order  = _safe_cv(dc_orders_padded)

        x_vals.append(cv_demand)
        y_vals.append(cv_order)
        labels.append(f"DC {dc_id}")

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(6.5, 6))

    lim = max(x_vals.max(), y_vals.max(), 0.01) * 1.30
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#999999", zorder=1,
            label="y = x  (no amplification)")

    ax.scatter(x_vals, y_vals, s=140, c=color, edgecolors="white",
               linewidths=0.8, alpha=0.85, zorder=2, marker="s")

    # Annotate each DC
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x_vals[i], y_vals[i]),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=10, fontweight="bold")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("CV of Aggregate Downstream Demand", fontsize=12)
    ax.set_ylabel("CV of DC Order Qty to Supplier", fontsize=12)
    ax.set_title(f"DC Level: Demand CV vs. Order CV - {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fname = out_dir / f"dc_scatter_{model_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  BULLWHIP EFFECT -- Scatter Plot Generator")
    print("=" * 65 + "\n")

    # Load all trajectories first (needed for the combined plot)
    all_dfs: dict[str, pd.DataFrame] = {}
    for model_name, xlsx_path in XLSX_PATHS.items():
        all_dfs[model_name] = _load_trajectory(xlsx_path)

    for model_name, df in all_dfs.items():
        print(f"-- {model_name} --")

        # Task 1: Retailer scatter (per-model)
        plot_retailer_scatter(model_name, df, OUT_DIR)

        # Task 2: DC scatter (per-model)
        plot_dc_scatter(model_name, df, OUT_DIR)
        print()

    # Combined retailer scatter (all models in one picture)
    print("-- Combined Retailer Scatter --")
    plot_retailer_scatter_combined(all_dfs, OUT_DIR)
    print()

    print("=" * 65)
    print(f"  All plots saved to: {OUT_DIR.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
