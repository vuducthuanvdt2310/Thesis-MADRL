#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bullwhip Effect Analysis: 4 Models x 10 Networks
=================================================

Computes three bullwhip metrics at Retailer and DC levels from
step_trajectory_ep1.xlsx per (model, network).

Metrics (per SKU, then averaged across SKUs):
  1. BWE_var  = Var(Orders_t) / Var(Demand_t)          (Lee et al. 1997)
  2. BWE_cv   = CV(Orders_t)  / CV(Demand_t)           (Fransoo & Wouters 2000)
  3. OAR      = Std(Orders_t) / Std(Demand_t)           (Order Amplification Ratio)

Levels:
  - Retailer: per-retailer time series, then mean across retailers
  - DC:       aggregate demand arriving at DC vs DC orders to supplier

Output: Bullwhip_Analysis.xlsx with sheets:
  - Retailer_BWE   : 10 networks x 4 models (BWE_var)
  - DC_BWE         : 10 networks x 4 models (BWE_var)
  - All_Metrics    : full detail (all 3 metrics x 2 levels)
  - Summary        : GNN-HAPPO reduction % vs each baseline
  - Methodology    : formulas + references

Usage:
    python bullwhip_analysis.py
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
OUTPUT = ROOT / "Bullwhip_Analysis.xlsx"

NETWORKS = ["1x3", "1x7", "1x10", "2x15", "2x20", "2x30", "2x40",
            "4x15", "4x30", "4x40"]

MODELS = {
    "(s,S) Policy": "basestock",
    "MAPPO":        "mappo",
    "HAPPO":        "happo",
    "GNN-HAPPO":    "gnn",
}
MODEL_NAMES = list(MODELS.keys())
PROPOSED = "GNN-HAPPO"
BASELINES = ["(s,S) Policy", "MAPPO", "HAPPO"]
N_SKUS = 3

NET_TOPOLOGY = {
    "1x3": (1, 3), "1x7": (1, 7), "1x10": (1, 10),
    "2x15": (2, 15), "2x20": (2, 20), "2x30": (2, 30), "2x40": (2, 40),
    "4x15": (4, 15), "4x30": (4, 30), "4x40": (4, 40),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def traj_path(model_folder: str, net: str) -> Path:
    return EVAL / f"{model_folder}_{net}" / "step_trajectory_ep1.xlsx"


def load_traj(model_folder: str, net: str) -> pd.DataFrame:
    p = traj_path(model_folder, net)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_excel(p)


# ---------------------------------------------------------------------------
# Bullwhip computation
# ---------------------------------------------------------------------------

def safe_ratio(num: float, den: float) -> float:
    if den == 0 or np.isnan(den):
        return float("nan")
    return num / den


def compute_agent_bwe(df: pd.DataFrame, agent_id: int) -> dict:
    """Compute 3 BWE metrics for one agent, averaged across SKUs."""
    agent_df = df[df["agent_id"] == agent_id].sort_values("step")
    bwe_vars, bwe_cvs, oars = [], [], []

    for sku in range(N_SKUS):
        demand = agent_df[f"demand_{sku}"].to_numpy(dtype=float)
        orders = agent_df[f"order_{sku}"].to_numpy(dtype=float)

        var_d = float(np.var(demand, ddof=1)) if len(demand) > 1 else 0.0
        var_o = float(np.var(orders, ddof=1)) if len(orders) > 1 else 0.0
        std_d = float(np.std(demand, ddof=1)) if len(demand) > 1 else 0.0
        std_o = float(np.std(orders, ddof=1)) if len(orders) > 1 else 0.0
        mean_d = float(np.mean(demand))
        mean_o = float(np.mean(orders))

        bwe_vars.append(safe_ratio(var_o, var_d))
        cv_d = safe_ratio(std_d, mean_d) if mean_d != 0 else float("nan")
        cv_o = safe_ratio(std_o, mean_o) if mean_o != 0 else float("nan")
        bwe_cvs.append(safe_ratio(cv_o, cv_d))
        oars.append(safe_ratio(std_o, std_d))

    def nanmean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "BWE_var": nanmean(bwe_vars),
        "BWE_cv":  nanmean(bwe_cvs),
        "OAR":     nanmean(oars),
    }


def compute_dc_bwe(df: pd.DataFrame, dc_id: int, retailer_ids: list[int]) -> dict:
    """DC-level BWE: aggregate retailer demand → DC vs DC orders to supplier."""
    dc_df = df[df["agent_id"] == dc_id].sort_values("step")
    steps = sorted(df["step"].unique())

    bwe_vars, bwe_cvs, oars = [], [], []
    for sku in range(N_SKUS):
        # Aggregate demand arriving at this DC = sum of retailer demands
        agg_demand = np.zeros(len(steps))
        for r_id in retailer_ids:
            r_df = df[df["agent_id"] == r_id].sort_values("step")
            agg_demand += r_df[f"demand_{sku}"].to_numpy(dtype=float)

        dc_orders = dc_df[f"order_{sku}"].to_numpy(dtype=float)

        var_d = float(np.var(agg_demand, ddof=1)) if len(agg_demand) > 1 else 0.0
        var_o = float(np.var(dc_orders, ddof=1)) if len(dc_orders) > 1 else 0.0
        std_d = float(np.std(agg_demand, ddof=1)) if len(agg_demand) > 1 else 0.0
        std_o = float(np.std(dc_orders, ddof=1)) if len(dc_orders) > 1 else 0.0
        mean_d = float(np.mean(agg_demand))
        mean_o = float(np.mean(dc_orders))

        bwe_vars.append(safe_ratio(var_o, var_d))
        cv_d = safe_ratio(std_d, mean_d) if mean_d != 0 else float("nan")
        cv_o = safe_ratio(std_o, mean_o) if mean_o != 0 else float("nan")
        bwe_cvs.append(safe_ratio(cv_o, cv_d))
        oars.append(safe_ratio(std_o, std_d))

    def nanmean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "BWE_var": nanmean(bwe_vars),
        "BWE_cv":  nanmean(bwe_cvs),
        "OAR":     nanmean(oars),
    }


def analyze_one(model_label: str, net: str) -> dict:
    """Return retailer-level and DC-level BWE for one (model, network)."""
    folder = MODELS[model_label]
    df = load_traj(folder, net)
    n_dcs, n_retailers = NET_TOPOLOGY[net]
    n_agents = n_dcs + n_retailers

    # DC assignments: evenly split retailers across DCs
    retailers_per_dc = n_retailers // n_dcs
    dc_assignments: dict[int, list[int]] = {}
    for dc in range(n_dcs):
        start = n_dcs + dc * retailers_per_dc
        end = n_dcs + (dc + 1) * retailers_per_dc
        if dc == n_dcs - 1:
            end = n_agents
        dc_assignments[dc] = list(range(start, end))

    # Retailer-level: mean across all retailers
    retailer_bwes = []
    for r_id in range(n_dcs, n_agents):
        retailer_bwes.append(compute_agent_bwe(df, r_id))

    def avg_metric(dicts, key):
        vals = [d[key] for d in dicts if not np.isnan(d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    ret_bwe_var = avg_metric(retailer_bwes, "BWE_var")
    ret_bwe_cv  = avg_metric(retailer_bwes, "BWE_cv")
    ret_oar     = avg_metric(retailer_bwes, "OAR")

    # DC-level: mean across all DCs
    dc_bwes = []
    for dc_id, r_ids in dc_assignments.items():
        dc_bwes.append(compute_dc_bwe(df, dc_id, r_ids))

    dc_bwe_var = avg_metric(dc_bwes, "BWE_var")
    dc_bwe_cv  = avg_metric(dc_bwes, "BWE_cv")
    dc_oar     = avg_metric(dc_bwes, "OAR")

    return {
        "ret_BWE_var": ret_bwe_var, "ret_BWE_cv": ret_bwe_cv, "ret_OAR": ret_oar,
        "dc_BWE_var":  dc_bwe_var,  "dc_BWE_cv":  dc_bwe_cv,  "dc_OAR":  dc_oar,
    }


def run_all() -> list[dict]:
    rows = []
    total = len(MODEL_NAMES) * len(NETWORKS)
    done = 0
    for net in NETWORKS:
        for model in MODEL_NAMES:
            done += 1
            print(f"[{done}/{total}] {model:<16} {net}")
            try:
                res = analyze_one(model, net)
                rows.append({"Network": net, "Model": model, **res})
            except Exception as e:
                print(f"  [SKIP] {e}")
                rows.append({
                    "Network": net, "Model": model,
                    "ret_BWE_var": np.nan, "ret_BWE_cv": np.nan, "ret_OAR": np.nan,
                    "dc_BWE_var": np.nan, "dc_BWE_cv": np.nan, "dc_OAR": np.nan,
                })
    return rows


# ===========================================================================
# Excel builder
# ===========================================================================

HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
GNN_FILL = PatternFill("solid", fgColor="E2EFDA")
BEST_FILL = PatternFill("solid", fgColor="FFE699")
POS_FONT = Font(color="006100", bold=True)
NEG_FONT = Font(color="9C0006", bold=True)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
RIGHT = Alignment(horizontal="right", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")
THIN = Side(border_style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws: Worksheet, row: int, n_cols: int):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def autosize(ws: Worksheet, widths: Iterable[int]):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_bwe_matrix(ws: Worksheet, data: list[dict], metric_key: str,
                     title: str) -> None:
    """Compact 10-network x 4-model matrix for one BWE metric."""
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=13)
    ws.cell(row=2, column=1,
            value="Values < 1.0 = bullwhip dampened (good). "
                  "Lower is better. Best per row highlighted yellow."
            ).font = Font(italic=True, color="595959")

    headers = ["Network", *MODEL_NAMES]
    for j, h in enumerate(headers, start=1):
        ws.cell(row=4, column=j, value=h)
    style_header(ws, 4, len(headers))

    for i, net in enumerate(NETWORKS):
        r = 5 + i
        ws.cell(row=r, column=1, value=net).alignment = CENTER
        ws.cell(row=r, column=1).border = BORDER
        vals = []
        for j, m in enumerate(MODEL_NAMES, start=2):
            match = [d for d in data if d["Network"] == net and d["Model"] == m]
            v = match[0][metric_key] if match else float("nan")
            vals.append(v)
            cell = ws.cell(row=r, column=j, value=(None if np.isnan(v) else v))
            cell.number_format = "0.0000"
            cell.alignment = RIGHT
            cell.border = BORDER
            if m == PROPOSED:
                cell.fill = GNN_FILL

        # Highlight best (lowest) non-NaN value
        finite = [(idx, v) for idx, v in enumerate(vals) if not np.isnan(v)]
        if finite:
            best_idx, _ = min(finite, key=lambda x: x[1])
            ws.cell(row=r, column=best_idx + 2).fill = BEST_FILL

    # Average row
    r_avg = 5 + len(NETWORKS) + 1
    ws.cell(row=r_avg, column=1, value="Average").font = Font(bold=True)
    ws.cell(row=r_avg, column=1).alignment = CENTER
    ws.cell(row=r_avg, column=1).border = BORDER
    for j, m in enumerate(MODEL_NAMES, start=2):
        vals = [d[metric_key] for d in data
                if d["Model"] == m and not np.isnan(d[metric_key])]
        avg = float(np.mean(vals)) if vals else float("nan")
        cell = ws.cell(row=r_avg, column=j, value=(None if np.isnan(avg) else avg))
        cell.number_format = "0.0000"
        cell.alignment = RIGHT
        cell.font = Font(bold=True)
        cell.border = BORDER
        if m == PROPOSED:
            cell.fill = GNN_FILL

    autosize(ws, [10, 14, 14, 14, 16])
    ws.freeze_panes = "B5"


def build_all_metrics(ws: Worksheet, data: list[dict]) -> None:
    ws.cell(row=1, column=1, value="All Bullwhip Metrics (detail)"
            ).font = Font(bold=True, size=13)
    headers = [
        "Network", "Model",
        "Ret BWE_var", "Ret BWE_cv", "Ret OAR",
        "DC BWE_var", "DC BWE_cv", "DC OAR",
    ]
    for j, h in enumerate(headers, start=1):
        ws.cell(row=3, column=j, value=h)
    style_header(ws, 3, len(headers))

    r = 4
    for net in NETWORKS:
        for m in MODEL_NAMES:
            match = [d for d in data if d["Network"] == net and d["Model"] == m]
            if not match:
                continue
            d = match[0]
            ws.cell(row=r, column=1, value=net).alignment = CENTER
            ws.cell(row=r, column=2, value=m).alignment = LEFT
            for j, k in enumerate(["ret_BWE_var", "ret_BWE_cv", "ret_OAR",
                                    "dc_BWE_var", "dc_BWE_cv", "dc_OAR"], start=3):
                v = d[k]
                cell = ws.cell(row=r, column=j, value=(None if np.isnan(v) else v))
                cell.number_format = "0.0000"
                cell.alignment = RIGHT
            if m == PROPOSED:
                for c in range(1, len(headers) + 1):
                    ws.cell(row=r, column=c).fill = GNN_FILL
            for c in range(1, len(headers) + 1):
                ws.cell(row=r, column=c).border = BORDER
            r += 1

    autosize(ws, [10, 14, 14, 14, 12, 14, 14, 12])
    ws.freeze_panes = "C4"


def build_summary(ws: Worksheet, data: list[dict]) -> None:
    ws.cell(row=1, column=1,
            value="GNN-HAPPO BWE Reduction vs Baselines"
            ).font = Font(bold=True, size=13)
    ws.cell(row=2, column=1,
            value="Reduction % = (Baseline_BWE - GNN_BWE) / Baseline_BWE x 100. "
                  "Positive = GNN reduces bullwhip."
            ).font = Font(italic=True, color="595959")

    for level, metric_key, start_row, level_label in [
        ("Retailer", "ret_BWE_var", 4, "RETAILER-LEVEL BWE Reduction %"),
        ("DC",       "dc_BWE_var",  4 + len(NETWORKS) + 4, "DC-LEVEL BWE Reduction %"),
    ]:
        ws.cell(row=start_row, column=1, value=level_label
                ).font = Font(bold=True, size=12)
        headers = ["Network", *BASELINES, "GNN-HAPPO BWE"]
        for j, h in enumerate(headers, start=1):
            ws.cell(row=start_row + 1, column=j, value=h)
        style_header(ws, start_row + 1, len(headers))

        for i, net in enumerate(NETWORKS):
            r = start_row + 2 + i
            ws.cell(row=r, column=1, value=net).alignment = CENTER
            ws.cell(row=r, column=1).border = BORDER

            gnn_match = [d for d in data
                         if d["Network"] == net and d["Model"] == PROPOSED]
            gnn_v = gnn_match[0][metric_key] if gnn_match else float("nan")

            for j, bl in enumerate(BASELINES, start=2):
                bl_match = [d for d in data
                            if d["Network"] == net and d["Model"] == bl]
                bl_v = bl_match[0][metric_key] if bl_match else float("nan")
                if np.isnan(bl_v) or np.isnan(gnn_v) or bl_v == 0:
                    red_pct = float("nan")
                else:
                    red_pct = (bl_v - gnn_v) / bl_v * 100.0
                cell = ws.cell(row=r, column=j,
                               value=(None if np.isnan(red_pct) else red_pct))
                cell.number_format = "0.00"
                cell.alignment = RIGHT
                cell.border = BORDER
                if not np.isnan(red_pct):
                    cell.font = POS_FONT if red_pct >= 0 else NEG_FONT

            gnn_cell = ws.cell(row=r, column=len(headers),
                               value=(None if np.isnan(gnn_v) else gnn_v))
            gnn_cell.number_format = "0.0000"
            gnn_cell.alignment = RIGHT
            gnn_cell.border = BORDER
            gnn_cell.fill = GNN_FILL

        # Average row
        r_avg = start_row + 2 + len(NETWORKS)
        ws.cell(row=r_avg, column=1, value="Average").font = Font(bold=True)
        ws.cell(row=r_avg, column=1).border = BORDER
        for j, bl in enumerate(BASELINES, start=2):
            reds = []
            for net in NETWORKS:
                gnn_m = [d for d in data
                         if d["Network"] == net and d["Model"] == PROPOSED]
                bl_m = [d for d in data
                        if d["Network"] == net and d["Model"] == bl]
                gv = gnn_m[0][metric_key] if gnn_m else float("nan")
                bv = bl_m[0][metric_key] if bl_m else float("nan")
                if not (np.isnan(gv) or np.isnan(bv) or bv == 0):
                    reds.append((bv - gv) / bv * 100.0)
            avg = float(np.mean(reds)) if reds else float("nan")
            cell = ws.cell(row=r_avg, column=j,
                           value=(None if np.isnan(avg) else avg))
            cell.number_format = "0.00"
            cell.alignment = RIGHT
            cell.font = Font(bold=True)
            cell.border = BORDER
            if not np.isnan(avg):
                cell.font = POS_FONT if avg >= 0 else NEG_FONT

        gnn_vals = [d[metric_key] for d in data
                    if d["Model"] == PROPOSED and not np.isnan(d[metric_key])]
        gnn_avg = float(np.mean(gnn_vals)) if gnn_vals else float("nan")
        cell = ws.cell(row=r_avg, column=len(headers),
                       value=(None if np.isnan(gnn_avg) else gnn_avg))
        cell.number_format = "0.0000"
        cell.alignment = RIGHT
        cell.font = Font(bold=True)
        cell.border = BORDER
        cell.fill = GNN_FILL

    autosize(ws, [10, 18, 18, 18, 16])
    ws.freeze_panes = "B5"


def build_methodology(ws: Worksheet) -> None:
    ws.cell(row=1, column=1, value="Methodology & References"
            ).font = Font(bold=True, size=14)
    lines = [
        ("", ""),
        ("Data source", "evaluation_results/<model>_<net>/step_trajectory_ep1.xlsx "
                        "(episode 1, all steps, per-agent demand_k and order_k for k=0,1,2 SKUs)"),
        ("", ""),
        ("Metric 1: BWE_var (Classic Bullwhip Ratio)",
            "BWE = Var(Orders_t) / Var(Demand_t)  per SKU, then averaged across SKUs. "
            "Reference: Lee, Padmanabhan & Whang (1997) 'The Bullwhip Effect in Supply Chains'."),
        ("", "  BWE = 1 means orders perfectly track demand variance."),
        ("", "  BWE > 1 means demand amplification (bullwhip)."),
        ("", "  BWE < 1 means demand dampening (smoothing)."),
        ("", ""),
        ("Metric 2: BWE_cv (CV Ratio)",
            "BWE_cv = CV(Orders) / CV(Demand), where CV = Std/Mean. "
            "More robust when comparing across SKUs with different mean demands. "
            "Reference: Fransoo & Wouters (2000) 'Measuring the bullwhip effect in the supply chain'."),
        ("", ""),
        ("Metric 3: OAR (Order Amplification Ratio)",
            "OAR = Std(Orders_t) / Std(Demand_t). Direct measure of order variability "
            "amplification. Simpler than BWE_var, commonly used in simulation studies."),
        ("", ""),
        ("Retailer-level BWE",
            "For each retailer: compute BWE per SKU from the retailer's own demand_k "
            "and order_k time series. Average across SKUs, then average across all "
            "retailers in the network."),
        ("DC-level BWE",
            "Aggregate demand = sum of demand_k across all retailers assigned to the DC. "
            "DC orders = the DC's own order_k to supplier. BWE = Var(DC_orders) / "
            "Var(Agg_demand). Average across SKUs, then across DCs."),
        ("", ""),
        ("BWE Reduction %",
            "(Baseline_BWE - GNN_BWE) / Baseline_BWE x 100. "
            "Positive = GNN-HAPPO reduces bullwhip relative to baseline."),
        ("", ""),
        ("Interpretation for thesis",
            "Lower BWE values demonstrate that the policy produces ordering patterns "
            "closer to actual demand, reducing supply chain nervousness, excess "
            "inventory, and emergency orders. GNN-HAPPO's graph-aware coordination "
            "should yield lower DC-level BWE as it explicitly models DC-Retailer "
            "information flow."),
    ]
    for i, (k, v) in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=k).font = Font(bold=bool(k))
        ws.cell(row=i, column=1).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row=i, column=2, value=v).alignment = Alignment(vertical="top", wrap_text=True)
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 100


def write_excel(data: list[dict]) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    build_bwe_matrix(wb.create_sheet("Retailer_BWE"), data, "ret_BWE_var",
                     "Retailer-Level Bullwhip Ratio (BWE = Var(Orders)/Var(Demand))")
    build_bwe_matrix(wb.create_sheet("DC_BWE"), data, "dc_BWE_var",
                     "DC-Level Bullwhip Ratio (BWE = Var(DC_Orders)/Var(Agg_Demand))")
    build_all_metrics(wb.create_sheet("All_Metrics"), data)
    build_summary(wb.create_sheet("Summary"), data)
    build_methodology(wb.create_sheet("Methodology"))

    if OUTPUT.exists():
        try:
            OUTPUT.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot delete {OUTPUT}. Close the file in Excel and re-run."
            ) from e
    wb.save(OUTPUT)
    print(f"\n[OK] Wrote {OUTPUT}")


def main():
    print("=" * 60)
    print("BULLWHIP EFFECT ANALYSIS")
    print("=" * 60)
    print(f"  Models   : {', '.join(MODEL_NAMES)}")
    print(f"  Networks : {', '.join(NETWORKS)}")
    print(f"  Metrics  : BWE_var, BWE_cv, OAR")
    print(f"  Levels   : Retailer, DC")
    print("=" * 60)

    data = run_all()
    write_excel(data)

    # Quick console summary
    print("\n--- Retailer BWE_var (lower = less bullwhip) ---")
    print(f"{'Network':<8}", end="")
    for m in MODEL_NAMES:
        print(f"{m:>16}", end="")
    print()
    for net in NETWORKS:
        print(f"{net:<8}", end="")
        for m in MODEL_NAMES:
            match = [d for d in data if d["Network"] == net and d["Model"] == m]
            v = match[0]["ret_BWE_var"] if match else float("nan")
            print(f"{v:>16.4f}" if not np.isnan(v) else f"{'N/A':>16}", end="")
        print()


if __name__ == "__main__":
    main()
