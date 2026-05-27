"""Render demand-scenario comparison charts from Network_demand_scenario_input.xlsx.

Reads two sheets ('Total_Cost', 'Fill_Rate') and produces:
  - NetworkDemand_cost_comparison.png   -- average Total Cost (VND, thousands)
  - NetworkDemand_fill_rate_comparison.png -- average Fill Rate (%)

Layout: 2 rows x 5 columns (one subplot per network), 4 grouped bars per
scenario (one per model).  Designed for slide presentation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT_XLSX = ROOT / "Network_demand_scenario_input.xlsx"
OUT_COST = ROOT / "NetworkDemand_cost_comparison.png"
OUT_FILL = ROOT / "NetworkDemand_fill_rate_comparison.png"

NETWORK_ORDER = [
    ("Network 1",  "1x3",  "1x3\n(1 DC, 3 Ret)"),
    ("Network 2",  "1x7",  "1x7\n(1 DC, 7 Ret)"),
    ("Network 3",  "1x10", "1x10\n(1 DC, 10 Ret)"),
    ("Network 4",  "2x15", "2x15\n(2 DC, 15 Ret)"),
    ("Network 5",  "2x20", "2x20\n(2 DC, 20 Ret)"),
    ("Network 6",  "2x30", "2x30\n(2 DC, 30 Ret)"),
    ("Network 7",  "2x40", "2x40\n(2 DC, 40 Ret)"),
    ("Network 8",  "4x15", "4x15\n(4 DC, 15 Ret)"),
    ("Network 9",  "4x30", "4x30\n(4 DC, 30 Ret)"),
    ("Network 10", "4x40", "4x40\n(4 DC, 40 Ret)"),
]
SCENARIOS = ["S1-Balanced", "S2-High", "S3-Extreme"]
SCENARIO_SHORT = ["S1", "S2", "S3"]
MODELS = ["(s,S) Policy", "MAPPO", "HAPPO", "GNN-HAPPO"]
COLORS = ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3"]
MODEL_COLOR = dict(zip(MODELS, COLORS))


def load(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    df["Instance"] = df["Instance"].ffill()
    df["Network Size"] = df["Network Size"].ffill()
    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    return df


def get_value(df: pd.DataFrame, network: str, scenario: str, model: str):
    row = df[(df["Network Size"].astype(str) == network) &
             (df["Scenario"] == scenario)]
    if row.empty:
        return np.nan
    val = row.iloc[0][model]
    return float(val) if pd.notna(val) else np.nan


def render(df: pd.DataFrame, *, ylabel: str, title: str, out_path: Path,
           value_format: str, divide_by: float = 1.0,
           ylim: tuple | None = None) -> None:
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 9), sharey=False)

    bar_width = 0.18
    x_positions = np.arange(len(SCENARIOS))

    for idx, (_, network, subtitle) in enumerate(NETWORK_ORDER):
        r_idx, c_idx = divmod(idx, ncols)
        ax = axes[r_idx][c_idx]

        for j, model in enumerate(MODELS):
            heights = [get_value(df, network, s, model) / divide_by
                       for s in SCENARIOS]
            offset = (j - (len(MODELS) - 1) / 2) * bar_width
            bars = ax.bar(
                x_positions + offset, heights, bar_width,
                color=MODEL_COLOR[model], edgecolor="white", linewidth=0.4,
                label=model if idx == 0 else None,
            )
            labels = [
                ("" if (np.isnan(h) if isinstance(h, float) else False)
                 else format(h, value_format))
                for h in heights
            ]
            ax.bar_label(bars, labels=labels, padding=1, fontsize=6.5,
                         fontweight="bold")

        ax.set_title(subtitle, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(SCENARIO_SHORT, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if c_idx == 0:
            ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in MODELS]
    fig.legend(handles, MODELS, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=11,
               handlelength=1.5, handletextpad=0.5, columnspacing=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.38, wspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_XLSX}. Run build_network_demand_scenario_input.py first."
        )

    cost_df = load("Total_Cost")
    fill_df = load("Fill_Rate")

    render(
        cost_df,
        ylabel="Avg Total Cost (VND, thousands)",
        title="Average Total Cost Across Demand Scenarios (10 Networks)",
        out_path=OUT_COST,
        value_format=",.0f",
        divide_by=1000.0,
    )

    render(
        fill_df,
        ylabel="Avg Fill Rate (%)",
        title="Average Fill Rate Across Demand Scenarios (10 Networks)",
        out_path=OUT_FILL,
        value_format=".1f",
        divide_by=1.0,
        ylim=(0, 105),
    )


if __name__ == "__main__":
    main()
