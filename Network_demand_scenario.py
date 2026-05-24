"""Render demand-scenario comparison figures from Network_demand_scenario_input.xlsx.

Reads two sheets ('Total_Cost', 'Fill_Rate') from the input workbook and produces:
  • total_cost_comparison.png   — average Total Cost (VND, thousands)
  • fill_rate_comparison.png    — average Fill Rate (%)

Both are 1×3 grouped-bar figures (one subplot per network instance), 4 bars per
scenario group (one per model), shared legend below the figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT_XLSX = ROOT / "Network_demand_scenario_input.xlsx"
OUT_COST = ROOT / "NetworkDemand_cost_comparison.png"
OUT_FILL = ROOT / "NetworkDemand_fill_rate_comparison.png"

INSTANCE_ORDER = [
    ("Instance 1", "1x7",  "Instance 1 (1×7)"),
    ("Instance 2", "2x15", "Instance 2 (2×15)"),
    ("Instance 3", "2x30", "Instance 3 (2×30)"),
    ("Instance 4", "4x40", "Instance 4 (4×40)"),
]
SCENARIOS = ["S1-Balanced", "S2-High", "S3-Extreme"]
MODELS = ["(s,S) Policy", "MAPPO", "HAPPO", "GNN-HAPPO"]
COLORS = ["#4C72B0", "#DD8452", "#55A479", "#C44E52"]
MODEL_COLOR = dict(zip(MODELS, COLORS))


def load(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    # forward-fill merged Instance / Network Size columns
    df["Instance"] = df["Instance"].ffill()
    df["Network Size"] = df["Network Size"].ffill()
    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    return df


def get_value(df: pd.DataFrame, network: str, scenario: str, model: str):
    row = df[(df["Network Size"].astype(str) == network) & (df["Scenario"] == scenario)]
    if row.empty:
        return np.nan
    val = row.iloc[0][model]
    return float(val) if pd.notna(val) else np.nan


def render(df: pd.DataFrame, *, ylabel: str, title: str, out_path: Path,
           value_format: str, divide_by: float = 1.0, ylim: tuple | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    flat_axes = axes.flatten()

    bar_width = 0.2
    x_positions = np.arange(len(SCENARIOS))

    for ax, (_, network, subtitle) in zip(flat_axes, INSTANCE_ORDER):
        for j, model in enumerate(MODELS):
            heights = [get_value(df, network, s, model) / divide_by for s in SCENARIOS]
            offset = (j - (len(MODELS) - 1) / 2) * bar_width
            bars = ax.bar(
                x_positions + offset, heights, bar_width,
                color=MODEL_COLOR[model], edgecolor="white", linewidth=0.5,
                label=model,
            )
            labels = [
                ("" if np.isnan(h) else format(h, value_format)) for h in heights
            ]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(SCENARIOS, rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # y-label on both leftmost panels (top-left and bottom-left of the 2x2 grid)
    axes[0, 0].set_ylabel(ylabel, fontsize=11)
    axes[1, 0].set_ylabel(ylabel, fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # one shared legend below
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in MODELS]
    fig.legend(handles, MODELS, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09, top=0.93, hspace=0.30)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_XLSX}. Run build_network_demand_scenario_input.py first."
        )

    cost_df = load("Total_Cost")
    fill_df = load("Fill_Rate")

    # Cost: raw VND in input → divide by 1000 for display
    render(
        cost_df,
        ylabel="Average Total Cost (VND, thousands)",
        title="Average Total Cost Across Demand Scenarios",
        out_path=OUT_COST,
        value_format=",.1f",
        divide_by=1000.0,
    )

    render(
        fill_df,
        ylabel="Average Fill Rate (%)",
        title="Average Fill Rate Across Demand Scenarios",
        out_path=OUT_FILL,
        value_format=".1f",
        divide_by=1.0,
        ylim=(0, 100),
    )


if __name__ == "__main__":
    main()
