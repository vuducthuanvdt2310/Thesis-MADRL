"""Generate the input workbook Network_demand_scenario_input.xlsx.

Two sheets ('Total_Cost' and 'Fill_Rate'), each with one row per
(Instance, Scenario) pair and one column per model. The Instance 2 (2x15) rows
are pre-populated from evaluation_results/robustness_comparison/robustness_summary.csv
so the user can see the expected format and the magnitudes. The Instance 1 (1x7)
and Instance 3 (2x30) rows are left blank for the user to fill in.

Run once to (re)generate the template. The plotting script reads from it.
"""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
ROBUST_CSV = ROOT / "evaluation_results" / "robustness_comparison" / "robustness_summary.csv"
OUTPUT = ROOT / "Network_demand_scenario_input.xlsx"

INSTANCES = [
    ("Instance 1", "1x7"),
    ("Instance 2", "2x15"),
    ("Instance 3", "2x30"),
    ("Instance 4", "4x40"),
]
SCENARIOS = ["S1-Balanced", "S2-High", "S3-Extreme"]
MODELS = ["(s,S) Policy", "MAPPO", "HAPPO", "GNN-HAPPO"]

# robustness_summary.csv uses these model labels — map to our display names
ROBUST_MODEL_MAP = {
    "(s,S) BaseStock": "(s,S) Policy",
    "MAPPO": "MAPPO",
    "HAPPO": "HAPPO",
    "GNN-HAPPO": "GNN-HAPPO",
}


def load_2x15_prefill() -> dict:
    """Return {(scenario, model): (cost, fill)} from robustness_summary.csv."""
    if not ROBUST_CSV.exists():
        return {}
    df = pd.read_csv(ROBUST_CSV)
    out: dict = {}
    for _, r in df.iterrows():
        model = ROBUST_MODEL_MAP.get(r["Model"], r["Model"])
        out[(r["Scenario"], model)] = (float(r["Mean_Total_Cost"]), float(r["Mean_Fill_Rate_%"]))
    return out


# --- styling ---
HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
BLOCK_FILL_PREFILLED = PatternFill("solid", fgColor="E2EFDA")  # green = filled
BLOCK_FILL_EMPTY = PatternFill("solid", fgColor="FFF2CC")       # yellow = please fill
THIN = Side(border_style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")
RIGHT = Alignment(horizontal="right", vertical="center")


def style_header(ws, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def build_sheet(ws, metric: str, prefill: dict) -> None:
    """metric is 'cost' or 'fill'; column index in prefill tuple is 0 or 1."""
    headers = ["Instance", "Network Size", "Scenario", *MODELS]
    ws.append(headers)
    style_header(ws, len(headers))

    row = 2
    for inst_label, net in INSTANCES:
        is_prefilled = (net == "2x15")
        block_fill = BLOCK_FILL_PREFILLED if is_prefilled else BLOCK_FILL_EMPTY
        block_start = row
        for s in SCENARIOS:
            ws.cell(row=row, column=1, value=inst_label)
            ws.cell(row=row, column=2, value=net)
            ws.cell(row=row, column=3, value=s)
            for j, m in enumerate(MODELS, start=4):
                cell = ws.cell(row=row, column=j)
                if is_prefilled and (s, m) in prefill:
                    val = prefill[(s, m)][0 if metric == "cost" else 1]
                    cell.value = val
                cell.number_format = "#,##0.00" if metric == "cost" else "0.00"
                cell.fill = block_fill
                cell.alignment = RIGHT
                cell.border = BORDER

            for c in (1, 2, 3):
                ws.cell(row=row, column=c).border = BORDER
                ws.cell(row=row, column=c).alignment = LEFT if c == 1 else CENTER
            row += 1

        # merge Instance + Network across the 3 scenario rows for this instance
        ws.merge_cells(start_row=block_start, start_column=1, end_row=row - 1, end_column=1)
        ws.merge_cells(start_row=block_start, start_column=2, end_row=row - 1, end_column=2)
        ws.cell(row=block_start, column=1).alignment = CENTER
        ws.cell(row=block_start, column=2).alignment = CENTER

    widths = [12, 14, 16, 16, 16, 16, 16]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "D2"


def build_notes_sheet(ws) -> None:
    ws.append(["Network_demand_scenario_input.xlsx — instructions"])
    ws["A1"].font = Font(bold=True, size=14)
    notes = [
        "",
        "Fill in the YELLOW cells with the mean Total Cost (VND, NOT divided by 1000) and",
        "mean Fill Rate (%) per (Instance × Scenario × Model). GREEN cells are pre-filled",
        "from evaluation_results/robustness_comparison/robustness_summary.csv (2x15 only).",
        "",
        "Sheets:",
        "  • Total_Cost  — values in raw VND (the plotting script divides by 1000).",
        "  • Fill_Rate   — values in % (0–100).",
        "",
        "Scenarios:",
        "  • S1-Balanced  — low-stress, matches training distribution",
        "  • S2-High      — elevated demand means, mixed stress",
        "  • S3-Extreme   — original means + high volatility",
        "",
        "Instances:",
        "  • Instance 1 (1x7)   — 1 DC × 7 retailers",
        "  • Instance 2 (2x15)  — 2 DCs × 15 retailers (thesis base, pre-filled)",
        "  • Instance 3 (2x30)  — 2 DCs × 30 retailers",
        "  • Instance 4 (4x40)  — 4 DCs × 40 retailers",
        "",
        "Once filled, run:  python Network_demand_scenario.py",
        "Outputs: total_cost_comparison.png and fill_rate_comparison.png",
    ]
    for line in notes:
        ws.append([line])
    ws.column_dimensions["A"].width = 95


def main() -> None:
    prefill = load_2x15_prefill()
    wb = Workbook()
    wb.remove(wb.active)
    build_notes_sheet(wb.create_sheet("Instructions"))
    build_sheet(wb.create_sheet("Total_Cost"), "cost", prefill)
    build_sheet(wb.create_sheet("Fill_Rate"), "fill", prefill)
    wb.save(OUTPUT)
    print(f"Wrote {OUTPUT}")
    if not prefill:
        print(f"  (warning: {ROBUST_CSV} not found — 2x15 rows left blank)")


if __name__ == "__main__":
    main()
