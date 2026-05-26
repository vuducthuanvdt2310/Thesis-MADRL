"""Build Benchmark_Comparison_GNN_vs_Baselines.xlsx across all 10 networks.

Aggregates per-episode metrics for each (model, network) pair and produces
a multi-sheet Excel report comparing the proposed GNN-HAPPO against the
three baselines: (s,S) Policy, MAPPO, HAPPO.

Sheets
------
1. Overview           Compact matrices (Mean Total Cost & Mean Fill Rate)
2. Cost_Comparison    Per-network mean cost + gap % of 3 baselines vs GNN-HAPPO
3. FillRate_Comparison Per-network mean fill rate + gap (pp & relative %)
4. Detailed_Stats     Mean, Std, Min, Max, 95% CI for every (model, network, metric)
5. Cost_Breakdown     Holding / Backlog / Ordering decomposition
6. Raw_Episodes       All 200 raw episode rows (4 models x 10 networks x 5 ep)
7. Methodology        Formulas and notes
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "evaluation_results"
OUTPUT = ROOT / "Benchmark_Comparison_GNN_vs_Baselines.xlsx"

NETWORKS: list[str] = [
    "1x3", "1x7", "1x10",
    "2x15", "2x20", "2x30", "2x40",
    "4x15", "4x30", "4x40",
]

# Display order matters: baselines first, proposed model last.
MODELS: list[str] = ["(s,S) Policy", "MAPPO", "HAPPO", "GNN-HAPPO"]
BASELINES: list[str] = ["(s,S) Policy", "MAPPO", "HAPPO"]
PROPOSED: str = "GNN-HAPPO"

METRIC_COLS = [
    "Total_Cost", "Fill_Rate", "Lost_Sales", "Avg_Inventory",
    "Total_Holding_Cost", "Total_Backlog_Cost", "Total_Ordering_Cost",
]


def csv_path(model: str, net: str) -> Path:
    """Resolve the per-(model, network) results CSV path."""
    if model == "(s,S) Policy":
        return EVAL / f"basestock_{net}" / f"results_ss_heuristic_{net}.csv"
    if model == "MAPPO":
        # 2x15 was saved without the "standard_" prefix.
        stem = "results_mappo" if net == "2x15" else "results_standard_mappo"
        return EVAL / f"mappo_{net}" / f"{stem}_{net}.csv"
    if model == "HAPPO":
        return EVAL / f"happo_{net}" / f"results_standard_happo_{net}.csv"
    if model == "GNN-HAPPO":
        return EVAL / f"gnn_{net}" / f"results_gnn_happo_{net}.csv"
    raise ValueError(f"Unknown model {model!r}")


def load_all() -> dict[tuple[str, str], pd.DataFrame]:
    """Load every (model, network) episode CSV into a dict."""
    data: dict[tuple[str, str], pd.DataFrame] = {}
    for m in MODELS:
        for n in NETWORKS:
            p = csv_path(m, n)
            if not p.exists():
                raise FileNotFoundError(f"Missing CSV for {m} / {n}: {p}")
            df = pd.read_csv(p)
            df.insert(0, "Network", n)
            df.insert(0, "Model", m)
            data[(m, n)] = df
    return data


# ---------- styling helpers ----------

HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
GNN_FILL = PatternFill("solid", fgColor="E2EFDA")  # highlight proposed-model column
BEST_FILL = PatternFill("solid", fgColor="FFE699")  # highlight per-row best
POS_FONT = Font(color="006100", bold=True)  # green for GNN-beats-baseline
NEG_FONT = Font(color="9C0006", bold=True)  # red for GNN-worse
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")
RIGHT = Alignment(horizontal="right", vertical="center")
THIN = Side(border_style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws: Worksheet, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def style_block(ws: Worksheet, r1: int, c1: int, r2: int, c2: int) -> None:
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ws.cell(row=r, column=c).border = BORDER


def autosize(ws: Worksheet, widths: Iterable[int]) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ---------- stats helpers ----------

def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Return mean, std, min, max, 95% CI half-width (t-approx via 1.96 for n=5 is rough;
    we use t critical value for the actual sample size)."""
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    # 95% CI half-width using normal approx; for n=5 this is approximate but consistent
    # with the existing Validate_Results.py convention in the repo.
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": half,
    }


# ---------- sheet builders ----------

def build_overview(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    ws.cell(row=1, column=1, value="Benchmark Overview: GNN-HAPPO vs Baselines (10 networks)").font = Font(bold=True, size=14)
    ws.cell(row=2, column=1, value="Values are mean across 5 evaluation episodes per (model, network).").font = Font(italic=True, color="595959")

    # ---- Matrix 1: Mean Total Cost (VND thousands) ----
    start = 4
    ws.cell(row=start, column=1, value="Mean Total Cost (VND thousands)  -  lower is better").font = Font(bold=True, size=12)
    headers = ["Network", *MODELS, "Best Model"]
    for j, h in enumerate(headers, start=1):
        ws.cell(row=start + 1, column=j, value=h)
    style_header(ws, start + 1, len(headers))

    for i, net in enumerate(NETWORKS):
        r = start + 2 + i
        ws.cell(row=r, column=1, value=net).alignment = CENTER
        row_values: list[float] = []
        for j, m in enumerate(MODELS, start=2):
            v = data[(m, net)]["Total_Cost"].mean() / 1000.0
            row_values.append(v)
            cell = ws.cell(row=r, column=j, value=v)
            cell.number_format = "#,##0.00"
            cell.alignment = RIGHT
        # best (min) for cost
        best_idx = int(np.argmin(row_values))
        ws.cell(row=r, column=best_idx + 2).fill = BEST_FILL
        ws.cell(row=r, column=2 + len(MODELS), value=MODELS[best_idx]).alignment = CENTER
        # GNN-HAPPO column highlight (always the last model col, index = len(MODELS) + 1)
        ws.cell(row=r, column=len(MODELS) + 1).fill = GNN_FILL
    style_block(ws, start + 1, 1, start + 1 + len(NETWORKS), len(headers))

    # ---- Matrix 2: Mean Fill Rate (%) ----
    start2 = start + 3 + len(NETWORKS) + 1
    ws.cell(row=start2, column=1, value="Mean Fill Rate (%)  -  higher is better").font = Font(bold=True, size=12)
    for j, h in enumerate(headers, start=1):
        ws.cell(row=start2 + 1, column=j, value=h)
    style_header(ws, start2 + 1, len(headers))

    for i, net in enumerate(NETWORKS):
        r = start2 + 2 + i
        ws.cell(row=r, column=1, value=net).alignment = CENTER
        row_values = []
        for j, m in enumerate(MODELS, start=2):
            v = float(data[(m, net)]["Fill_Rate"].mean())
            row_values.append(v)
            cell = ws.cell(row=r, column=j, value=v)
            cell.number_format = "0.00"
            cell.alignment = RIGHT
        best_idx = int(np.argmax(row_values))
        ws.cell(row=r, column=best_idx + 2).fill = BEST_FILL
        ws.cell(row=r, column=2 + len(MODELS), value=MODELS[best_idx]).alignment = CENTER
        ws.cell(row=r, column=len(MODELS) + 1).fill = GNN_FILL
    style_block(ws, start2 + 1, 1, start2 + 1 + len(NETWORKS), len(headers))

    # ---- Aggregate-across-networks bottom row ----
    start3 = start2 + 3 + len(NETWORKS) + 1
    ws.cell(row=start3, column=1, value="Average across all 10 networks").font = Font(bold=True, size=12)
    sub_headers = ["Metric", *MODELS]
    for j, h in enumerate(sub_headers, start=1):
        ws.cell(row=start3 + 1, column=j, value=h)
    style_header(ws, start3 + 1, len(sub_headers))

    avg_cost_row = start3 + 2
    avg_fill_row = start3 + 3
    ws.cell(row=avg_cost_row, column=1, value="Avg Total Cost (VND thousands)").alignment = LEFT
    ws.cell(row=avg_fill_row, column=1, value="Avg Fill Rate (%)").alignment = LEFT
    for j, m in enumerate(MODELS, start=2):
        c_vals = [data[(m, n)]["Total_Cost"].mean() / 1000.0 for n in NETWORKS]
        f_vals = [data[(m, n)]["Fill_Rate"].mean() for n in NETWORKS]
        c_cell = ws.cell(row=avg_cost_row, column=j, value=float(np.mean(c_vals)))
        c_cell.number_format = "#,##0.00"
        c_cell.alignment = RIGHT
        f_cell = ws.cell(row=avg_fill_row, column=j, value=float(np.mean(f_vals)))
        f_cell.number_format = "0.00"
        f_cell.alignment = RIGHT
    ws.cell(row=avg_cost_row, column=len(MODELS) + 1).fill = GNN_FILL
    ws.cell(row=avg_fill_row, column=len(MODELS) + 1).fill = GNN_FILL
    style_block(ws, start3 + 1, 1, start3 + 3, len(sub_headers))

    autosize(ws, [16, 16, 16, 16, 18, 18])
    ws.freeze_panes = "B5"


def build_cost_comparison(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    headers = [
        "Network",
        "Baseline",
        "Baseline Cost (VND thousands)",
        "GNN-HAPPO Cost (VND thousands)",
        "Absolute Gap (VND thousands)",
        "Relative Cost Reduction (%)",
    ]
    ws.append(headers)
    style_header(ws, 1, len(headers))

    row = 2
    for net in NETWORKS:
        gnn_mean = float(data[(PROPOSED, net)]["Total_Cost"].mean())
        block_start = row
        for b in BASELINES:
            b_mean = float(data[(b, net)]["Total_Cost"].mean())
            abs_gap = b_mean - gnn_mean
            rel_red = (b_mean - gnn_mean) / b_mean * 100.0 if b_mean else 0.0

            ws.cell(row=row, column=1, value=net).alignment = CENTER
            ws.cell(row=row, column=2, value=b).alignment = LEFT
            ws.cell(row=row, column=3, value=b_mean / 1000.0).number_format = "#,##0.00"
            ws.cell(row=row, column=4, value=gnn_mean / 1000.0).number_format = "#,##0.00"
            ws.cell(row=row, column=5, value=abs_gap / 1000.0).number_format = "#,##0.00;[Red]-#,##0.00"
            rel_cell = ws.cell(row=row, column=6, value=rel_red)
            rel_cell.number_format = "0.00"
            rel_cell.font = POS_FONT if rel_red >= 0 else NEG_FONT
            for c in range(3, len(headers) + 1):
                ws.cell(row=row, column=c).alignment = RIGHT
            row += 1
        # merge Network across the 3 baseline rows
        ws.merge_cells(start_row=block_start, start_column=1, end_row=row - 1, end_column=1)
        ws.cell(row=block_start, column=1).alignment = CENTER

    style_block(ws, 1, 1, row - 1, len(headers))

    # ---- Summary block at bottom: average gap per baseline ----
    row += 1
    ws.cell(row=row, column=1, value="Average across 10 networks").font = Font(bold=True)
    style_header(ws, row, len(headers))
    row += 1
    for b in BASELINES:
        b_means = np.array([data[(b, n)]["Total_Cost"].mean() for n in NETWORKS])
        gnn_means = np.array([data[(PROPOSED, n)]["Total_Cost"].mean() for n in NETWORKS])
        avg_b = float(b_means.mean()) / 1000.0
        avg_g = float(gnn_means.mean()) / 1000.0
        abs_gap = avg_b - avg_g
        # Average of per-network relative reductions (avoids size dominance)
        per_net_rel = (b_means - gnn_means) / b_means * 100.0
        avg_rel = float(per_net_rel.mean())

        ws.cell(row=row, column=1, value="ALL").alignment = CENTER
        ws.cell(row=row, column=2, value=b).alignment = LEFT
        ws.cell(row=row, column=3, value=avg_b).number_format = "#,##0.00"
        ws.cell(row=row, column=4, value=avg_g).number_format = "#,##0.00"
        ws.cell(row=row, column=5, value=abs_gap).number_format = "#,##0.00;[Red]-#,##0.00"
        rel_cell = ws.cell(row=row, column=6, value=avg_rel)
        rel_cell.number_format = "0.00"
        rel_cell.font = POS_FONT if avg_rel >= 0 else NEG_FONT
        for c in range(3, len(headers) + 1):
            ws.cell(row=row, column=c).alignment = RIGHT
        row += 1
    style_block(ws, row - len(BASELINES) - 1, 1, row - 1, len(headers))

    autosize(ws, [12, 16, 24, 24, 22, 24])
    ws.freeze_panes = "C2"


def build_fill_rate_comparison(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    headers = [
        "Network",
        "Baseline",
        "Baseline Fill Rate (%)",
        "GNN-HAPPO Fill Rate (%)",
        "Absolute Gap (pp)",
        "Relative Gap (%)",
    ]
    ws.append(headers)
    style_header(ws, 1, len(headers))

    row = 2
    for net in NETWORKS:
        gnn_mean = float(data[(PROPOSED, net)]["Fill_Rate"].mean())
        block_start = row
        for b in BASELINES:
            b_mean = float(data[(b, net)]["Fill_Rate"].mean())
            abs_gap = gnn_mean - b_mean
            rel_gap = abs_gap / b_mean * 100.0 if b_mean else 0.0

            ws.cell(row=row, column=1, value=net).alignment = CENTER
            ws.cell(row=row, column=2, value=b).alignment = LEFT
            ws.cell(row=row, column=3, value=b_mean).number_format = "0.0000"
            ws.cell(row=row, column=4, value=gnn_mean).number_format = "0.0000"
            abs_cell = ws.cell(row=row, column=5, value=abs_gap)
            abs_cell.number_format = "0.0000;[Red]-0.0000"
            abs_cell.font = POS_FONT if abs_gap >= 0 else NEG_FONT
            rel_cell = ws.cell(row=row, column=6, value=rel_gap)
            rel_cell.number_format = "0.0000"
            rel_cell.font = POS_FONT if rel_gap >= 0 else NEG_FONT
            for c in range(3, len(headers) + 1):
                ws.cell(row=row, column=c).alignment = RIGHT
            row += 1
        ws.merge_cells(start_row=block_start, start_column=1, end_row=row - 1, end_column=1)
        ws.cell(row=block_start, column=1).alignment = CENTER

    style_block(ws, 1, 1, row - 1, len(headers))

    # ---- Summary block ----
    row += 1
    ws.cell(row=row, column=1, value="Average across 10 networks").font = Font(bold=True)
    style_header(ws, row, len(headers))
    row += 1
    for b in BASELINES:
        b_means = np.array([data[(b, n)]["Fill_Rate"].mean() for n in NETWORKS])
        gnn_means = np.array([data[(PROPOSED, n)]["Fill_Rate"].mean() for n in NETWORKS])
        avg_b = float(b_means.mean())
        avg_g = float(gnn_means.mean())
        abs_gap = avg_g - avg_b
        per_net_rel = (gnn_means - b_means) / b_means * 100.0
        avg_rel = float(per_net_rel.mean())

        ws.cell(row=row, column=1, value="ALL").alignment = CENTER
        ws.cell(row=row, column=2, value=b).alignment = LEFT
        ws.cell(row=row, column=3, value=avg_b).number_format = "0.0000"
        ws.cell(row=row, column=4, value=avg_g).number_format = "0.0000"
        abs_cell = ws.cell(row=row, column=5, value=abs_gap)
        abs_cell.number_format = "0.0000;[Red]-0.0000"
        abs_cell.font = POS_FONT if abs_gap >= 0 else NEG_FONT
        rel_cell = ws.cell(row=row, column=6, value=avg_rel)
        rel_cell.number_format = "0.0000"
        rel_cell.font = POS_FONT if avg_rel >= 0 else NEG_FONT
        for c in range(3, len(headers) + 1):
            ws.cell(row=row, column=c).alignment = RIGHT
        row += 1
    style_block(ws, row - len(BASELINES) - 1, 1, row - 1, len(headers))

    autosize(ws, [12, 16, 22, 24, 18, 18])
    ws.freeze_panes = "C2"


def build_detailed_stats(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    headers = [
        "Network", "Model", "Metric",
        "Mean", "Std", "Min", "Max", "95% CI Half-Width", "N Episodes",
    ]
    ws.append(headers)
    style_header(ws, 1, len(headers))

    metrics_to_report = [
        ("Total_Cost", "Total Cost (VND)"),
        ("Fill_Rate", "Fill Rate (%)"),
        ("Lost_Sales", "Lost Sales (units)"),
        ("Avg_Inventory", "Avg Inventory (units)"),
    ]

    row = 2
    for net in NETWORKS:
        for m in MODELS:
            df = data[(m, net)]
            n_ep = len(df)
            for col, label in metrics_to_report:
                vals = df[col].to_numpy(dtype=float)
                s = summary_stats(vals)
                ws.cell(row=row, column=1, value=net).alignment = CENTER
                ws.cell(row=row, column=2, value=m).alignment = LEFT
                ws.cell(row=row, column=3, value=label).alignment = LEFT
                for j, key in enumerate(["mean", "std", "min", "max", "ci95"], start=4):
                    c = ws.cell(row=row, column=j, value=s[key])
                    c.number_format = "#,##0.0000"
                    c.alignment = RIGHT
                ws.cell(row=row, column=9, value=n_ep).alignment = CENTER
                # Highlight GNN-HAPPO rows
                if m == PROPOSED:
                    for c in range(1, len(headers) + 1):
                        ws.cell(row=row, column=c).fill = GNN_FILL
                row += 1

    style_block(ws, 1, 1, row - 1, len(headers))
    autosize(ws, [10, 14, 22, 18, 16, 16, 16, 20, 12])
    ws.freeze_panes = "D2"


def build_cost_breakdown(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    headers = [
        "Network", "Model",
        "Total Cost (VND k)", "Holding (VND k)", "Backlog (VND k)", "Ordering (VND k)",
        "Holding %", "Backlog %", "Ordering %",
    ]
    ws.append(headers)
    style_header(ws, 1, len(headers))

    row = 2
    for net in NETWORKS:
        for m in MODELS:
            df = data[(m, net)]
            total = float(df["Total_Cost"].mean())
            hold = float(df["Total_Holding_Cost"].mean())
            back = float(df["Total_Backlog_Cost"].mean())
            order = float(df["Total_Ordering_Cost"].mean())
            ws.cell(row=row, column=1, value=net).alignment = CENTER
            ws.cell(row=row, column=2, value=m).alignment = LEFT
            ws.cell(row=row, column=3, value=total / 1000.0).number_format = "#,##0.00"
            ws.cell(row=row, column=4, value=hold / 1000.0).number_format = "#,##0.00"
            ws.cell(row=row, column=5, value=back / 1000.0).number_format = "#,##0.00"
            ws.cell(row=row, column=6, value=order / 1000.0).number_format = "#,##0.00"
            for j, comp in zip((7, 8, 9), (hold, back, order)):
                pct = comp / total * 100.0 if total else 0.0
                c = ws.cell(row=row, column=j, value=pct)
                c.number_format = "0.00"
            for c in range(3, len(headers) + 1):
                ws.cell(row=row, column=c).alignment = RIGHT
            if m == PROPOSED:
                for c in range(1, len(headers) + 1):
                    ws.cell(row=row, column=c).fill = GNN_FILL
            row += 1

    style_block(ws, 1, 1, row - 1, len(headers))
    autosize(ws, [10, 14, 20, 18, 18, 18, 12, 12, 12])
    ws.freeze_panes = "C2"


def build_raw_episodes(ws: Worksheet, data: dict[tuple[str, str], pd.DataFrame]) -> None:
    frames = []
    for net in NETWORKS:
        for m in MODELS:
            frames.append(data[(m, net)])
    big = pd.concat(frames, ignore_index=True)
    # Ensure column order
    cols = ["Model", "Network", "Episode_Index", *METRIC_COLS]
    big = big[cols]
    for j, h in enumerate(cols, start=1):
        ws.cell(row=1, column=j, value=h)
    style_header(ws, 1, len(cols))

    for i, rec in enumerate(big.itertuples(index=False), start=2):
        for j, val in enumerate(rec, start=1):
            cell = ws.cell(row=i, column=j, value=val)
            if isinstance(val, float):
                cell.number_format = "#,##0.0000"
                cell.alignment = RIGHT
        ws.cell(row=i, column=1).alignment = LEFT
        ws.cell(row=i, column=2).alignment = CENTER
        ws.cell(row=i, column=3).alignment = CENTER
        if rec.Model == PROPOSED:
            for c in range(1, len(cols) + 1):
                ws.cell(row=i, column=c).fill = GNN_FILL

    style_block(ws, 1, 1, len(big) + 1, len(cols))
    autosize(ws, [14, 10, 12, 16, 14, 14, 16, 18, 18, 18])
    ws.freeze_panes = "D2"


def build_methodology(ws: Worksheet) -> None:
    ws.cell(row=1, column=1, value="Methodology & Formulas").font = Font(bold=True, size=14)

    lines = [
        ("", ""),
        ("Data Source", "evaluation_results/<model>_<network>/results_*.csv  (5 episodes per file)"),
        ("Networks", ", ".join(NETWORKS)),
        ("Models", ", ".join(MODELS) + "   (GNN-HAPPO = proposed)"),
        ("", ""),
        ("Per-(model, network) aggregation", "Mean of 5 evaluation episodes."),
        ("", ""),
        ("Cost Gap formula", "Relative_Cost_Reduction(%) = (Baseline_Cost - GNN_Cost) / Baseline_Cost x 100"),
        ("", "  Positive = GNN-HAPPO reduces cost relative to the baseline (better)."),
        ("", "  Absolute_Gap = Baseline_Cost - GNN_Cost  (positive = GNN cheaper)."),
        ("", ""),
        ("Fill-Rate Gap formula", "Absolute_Gap (percentage points, pp) = GNN_FillRate - Baseline_FillRate"),
        ("", "Relative_Gap (%) = (GNN_FillRate - Baseline_FillRate) / Baseline_FillRate x 100"),
        ("", "  Positive = GNN-HAPPO serves more demand than the baseline (better)."),
        ("", ""),
        ("'Average across 10 networks' (summary rows)",
            "Baseline_Cost and GNN_Cost are simple means of per-network means; Relative Reduction is the mean of per-network relative reductions (each network weighted equally, regardless of size)."),
        ("", ""),
        ("Stats columns (Detailed_Stats sheet)",
            "Mean, Std (sample, ddof=1), Min, Max over 5 episodes. 95% CI half-width approximated as 1.96 * Std / sqrt(N)."),
        ("", ""),
        ("Cost breakdown",
            "Holding, Backlog, Ordering means are taken across the 5 evaluation episodes. Percent columns are component / total across the same average episode."),
        ("", ""),
        ("Highlighting", "Green fill = GNN-HAPPO column/row. Yellow fill = best model on that row in the Overview sheet. Green/red font in Gap columns = sign of the gap."),
    ]
    for i, (k, v) in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=k).font = Font(bold=bool(k))
        ws.cell(row=i, column=1).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row=i, column=2, value=v).alignment = Alignment(vertical="top", wrap_text=True)

    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 110
    ws.row_dimensions[1].height = 22


def main() -> None:
    data = load_all()

    wb = Workbook()
    wb.remove(wb.active)
    build_overview(wb.create_sheet("Overview"), data)
    build_cost_comparison(wb.create_sheet("Cost_Comparison"), data)
    build_fill_rate_comparison(wb.create_sheet("FillRate_Comparison"), data)
    build_detailed_stats(wb.create_sheet("Detailed_Stats"), data)
    build_cost_breakdown(wb.create_sheet("Cost_Breakdown"), data)
    build_raw_episodes(wb.create_sheet("Raw_Episodes"), data)
    build_methodology(wb.create_sheet("Methodology"), )

    if OUTPUT.exists():
        try:
            OUTPUT.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot delete {OUTPUT}. Close the file in Excel and re-run."
            ) from e

    wb.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
