#!/usr/bin/env python
"""
Validate_Results.py
====================
Master validation script for comparing three inventory management policies:
  • GNN-HAPPO        (proposed, GNN-based multi-agent RL)
  • Standard HAPPO   (baseline, MLP-based multi-agent RL)
  • (s,S) Heuristic  (classical inventory heuristic)

Input:  Three results_*.csv files, each with 100 rows (one per episode).
        Columns: Episode_Index, Total_Cost, Fill_Rate, Lost_Sales, Avg_Inventory

Output:
  • final_validation_report.csv          – summary table (Mean ± 95% CI)
  • comparison_boxplot_cost.png          – Total Cost box plot
  • comparison_boxplot_fill_rate.png     – Fill Rate box plot
  • Printed summary + t-test results to console

Usage:
  python Validate_Results.py

  Optionally override paths:
  python Validate_Results.py \\
        --gnn_csv   evaluation_results/eval_gnn/results_gnn_happo.csv \\
        --happo_csv evaluation_results/eval_base/results_standard_happo.csv \\
        --ss_csv    evaluation_results/ss_v1/results_ss_heuristic.csv \\
        --out_dir   validation_output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ===========================================================================
# ① CONFIGURABLE FILE PATHS  ← edit these if you change filenames / locations
# ===========================================================================
PATHS: dict[str, str] = {
    "GNN-HAPPO":      "evaluation_results/eval_gnn/results_gnn_happo.csv",
    "Standard-HAPPO": "evaluation_results/eval_base/results_standard_happo.csv",
    "S-s-Heuristic":  "evaluation_results/basestock_v1/results_ss_heuristic.csv",
}

# Metrics to analyse (must match CSV column names exactly)
METRICS = ["Total_Cost", "Fill_Rate", "Lost_Sales", "Avg_Inventory"]

# Number of episodes (used in CI formula)
N_EPISODES = 30

# ===========================================================================
# Helpers
# ===========================================================================

def load_data(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load CSVs; abort with a clear message if any file is missing."""
    data = {}
    for model_name, path_str in paths.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[ERROR] File not found for '{model_name}': {p}")
            print("        Please run the corresponding test script first, or update PATHS.")
            sys.exit(1)
        df = pd.read_csv(p)
        # Validate required columns
        missing = [c for c in ["Episode_Index"] + METRICS if c not in df.columns]
        if missing:
            print(f"[ERROR] '{model_name}' CSV is missing columns: {missing}")
            sys.exit(1)
        if len(df) < N_EPISODES:
            print(f"[WARN ] '{model_name}' has only {len(df)} episodes (expected {N_EPISODES}).")
        data[model_name] = df
        print(f"[OK]  Loaded {len(df):3d} episodes for '{model_name}'  ← {p}")
    return data


def compute_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute Mean and 95% CI for each metric and model.
    CI = 1.96 × (std / √N)
    Returns a DataFrame indexed by (model, metric) with columns [mean, ci, std].
    """
    rows = []
    for model, df in data.items():
        n = len(df)
        for metric in METRICS:
            values = df[metric].dropna().values
            mean = float(np.mean(values))
            std  = float(np.std(values, ddof=1))
            ci   = 1.96 * (std / np.sqrt(n))
            rows.append({"model": model, "metric": metric,
                         "mean": mean, "std": std, "ci": ci, "n": n})
    return pd.DataFrame(rows)


def build_summary_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot stats into the thesis-ready format:
    Metric | GNN-HAPPO (Mean ± CI) | Standard HAPPO (Mean ± CI) | S-s-Heuristic (Mean ± CI)
    """
    model_names = list(PATHS.keys())  # preserve insertion order
    records = []
    for metric in METRICS:
        row = {"Metric": metric}
        for model in model_names:
            sub = stats_df[(stats_df["model"] == model) & (stats_df["metric"] == metric)]
            if sub.empty:
                row[model] = "N/A"
            else:
                m  = sub["mean"].values[0]
                ci = sub["ci"].values[0]
                row[model] = f"{m:.4f} ± {ci:.4f}"
        records.append(row)
    return pd.DataFrame(records)


def run_paired_ttests(data: dict[str, pd.DataFrame]) -> dict[str, tuple]:
    """
    Paired t-tests between GNN-HAPPO and each other model for Total_Cost.
    Returns {competitor: (t_stat, p_value)}.
    """
    gnn_name = "GNN-HAPPO"
    if gnn_name not in data:
        print("[WARN] 'GNN-HAPPO' not found in data; skipping t-tests.")
        return {}

    gnn_costs = data[gnn_name]["Total_Cost"].values
    results = {}
    for model_name, df in data.items():
        if model_name == gnn_name:
            continue
        other_costs = df["Total_Cost"].values
        n = min(len(gnn_costs), len(other_costs))
        t_stat, p_val = stats.ttest_rel(gnn_costs[:n], other_costs[:n])
        results[model_name] = (float(t_stat), float(p_val))
    return results


# ===========================================================================
# Visualisation
# ===========================================================================

PALETTE = {
    "GNN-HAPPO":      "#2E86AB",   # teal-blue   (proposed)
    "Standard-HAPPO": "#A23B72",   # purple      (baseline)
    "S-s-Heuristic":  "#F18F01",   # amber       (heuristic)
}


def plot_boxplots(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Generate two separate files:
      - comparison_boxplot_cost.png
      - comparison_boxplot_fill_rate.png
    Professional styling (seaborn whitegrid).
    """
    sns.set_theme(style="whitegrid", font_scale=1.15)

    model_names = list(PATHS.keys())
    colors      = [PALETTE.get(m, "#888888") for m in model_names]

    # --- helper: collect values in consistent model order ---
    def ordered_lists(metric: str):
        return [data[m][metric].values for m in model_names]

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=PALETTE.get(m, "#888"), label=m)
        for m in model_names
    ]

    # ----- Figure 1: Total Cost -----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig1.suptitle("Policy Comparison: 100-Episode Evaluation (seed=42)",
                  fontsize=14, fontweight="bold", y=1.02)
    bp1 = ax1.boxplot(
        ordered_lists("Total_Cost"),
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax1.set_title("Total Cost Comparison", fontweight="bold")
    ax1.set_ylabel("Total Cost (per episode)")
    ax1.set_xticks(range(1, len(model_names) + 1))
    ax1.set_xticklabels(model_names, rotation=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig1.legend(handles=legend_patches, loc="lower center",
                ncol=len(model_names), bbox_to_anchor=(0.5, -0.05),
                frameon=False, fontsize=11)
    fig1.tight_layout()
    cost_path = out_dir / "comparison_boxplot_cost.png"
    fig1.savefig(cost_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[OK]  Saved Total Cost box plot → {cost_path}")

    # ----- Figure 2: Fill Rate -----
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.suptitle("Policy Comparison: 100-Episode Evaluation (seed=42)",
                  fontsize=14, fontweight="bold", y=1.02)
    bp2 = ax2.boxplot(
        ordered_lists("Fill_Rate"),
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax2.set_title("Fill Rate Comparison", fontweight="bold")
    ax2.set_ylabel("Fill Rate (% service level)")
    ax2.set_xticks(range(1, len(model_names) + 1))
    ax2.set_xticklabels(model_names, rotation=12)
    ax2.axhline(y=95, color="red", linestyle="--", linewidth=1.3,
                label="Target 95 %", alpha=0.7)
    ax2.legend(loc="lower right", fontsize=10)

    fig2.legend(handles=legend_patches, loc="lower center",
                ncol=len(model_names), bbox_to_anchor=(0.5, -0.05),
                frameon=False, fontsize=11)
    fig2.tight_layout()
    fill_rate_path = out_dir / "comparison_boxplot_fill_rate.png"
    fig2.savefig(fill_rate_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK]  Saved Fill Rate box plot  → {fill_rate_path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master validation: compare GNN-HAPPO, Standard-HAPPO, and (s,S) heuristic."
    )
    parser.add_argument("--gnn_csv",   type=str, default=None,
                        help="Override path for GNN-HAPPO CSV")
    parser.add_argument("--happo_csv", type=str, default=None,
                        help="Override path for Standard-HAPPO CSV")
    parser.add_argument("--ss_csv",    type=str, default=None,
                        help="Override path for (s,S) Heuristic CSV")
    parser.add_argument("--out_dir",   type=str, default="validation_output",
                        help="Directory for all output files (default: validation_output)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI overrides of PATHS
    paths = dict(PATHS)   # copy
    if args.gnn_csv:
        paths["GNN-HAPPO"] = args.gnn_csv
    if args.happo_csv:
        paths["Standard-HAPPO"] = args.happo_csv
    if args.ss_csv:
        paths["S-s-Heuristic"] = args.ss_csv

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MASTER VALIDATION SCRIPT  —  Multi-Echelon Inventory Comparison")
    print("=" * 70 + "\n")

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("── Loading episode CSVs ──────────────────────────────────────────")
    data = load_data(paths)
    print()

    # ── 2. Compute statistics ────────────────────────────────────────────────
    print("── Computing statistics (Mean ± 95 % CI) ────────────────────────")
    stats_df = compute_stats(data)

    # ── 3. Summary table ────────────────────────────────────────────────────
    summary = build_summary_table(stats_df)
    report_path = out_dir / "final_validation_report.csv"
    summary.to_csv(report_path, index=False)
    print(f"\n{'─'*70}")
    print("  SUMMARY TABLE  (Mean ± 95 % CI,  N=100 episodes per model)")
    print(f"{'─'*70}")
    # pretty-print with fixed column widths
    col_widths = [max(len(str(v)) for v in [col] + summary[col].tolist()) + 2
                  for col in summary.columns]
    header = "".join(str(c).ljust(w) for c, w in zip(summary.columns, col_widths))
    print(header)
    print("─" * len(header))
    for _, row in summary.iterrows():
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print(f"\n[OK]  Saved report → {report_path}\n")

    # ── 4. Paired t-tests ───────────────────────────────────────────────────
    print(f"{'─'*70}")
    print("  PAIRED T-TEST  —  GNN-HAPPO vs. competitors  (metric: Total_Cost)")
    print(f"{'─'*70}")
    ttest_results = run_paired_ttests(data)
    for competitor, (t_stat, p_val) in ttest_results.items():
        significance = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ not significant"
        direction = ""
        gnn_mean    = data["GNN-HAPPO"]["Total_Cost"].mean()
        other_mean  = data[competitor]["Total_Cost"].mean()
        if gnn_mean < other_mean:
            pct = (other_mean - gnn_mean) / other_mean * 100
            direction = f"GNN-HAPPO is {pct:.1f}% cheaper"
        else:
            pct = (gnn_mean - other_mean) / other_mean * 100
            direction = f"GNN-HAPPO is {pct:.1f}% MORE expensive"
        print(f"  vs {competitor:<20}  t={t_stat:+.4f}  p={p_val:.6f}  {significance}")
        print(f"                           → {direction}")
    print()

    # ── 5. Box plots ────────────────────────────────────────────────────────
    print("── Generating visualisations ─────────────────────────────────────")
    plot_boxplots(data, out_dir)

    print("\n" + "=" * 70)
    print(f"  All outputs saved to: {out_dir.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
