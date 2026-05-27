#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paired t-test Validation: GNN-HAPPO vs 3 Baselines x 10 Networks
=================================================================

For each (baseline, network), runs a paired t-test on per-episode
Total_Cost and Fill_Rate between GNN-HAPPO and the baseline.

Pairing: episode i of GNN vs episode i of baseline (same random seed,
same demand realization → valid pairing).

Data source: evaluation_results/<model>_<net>/results_*_<net>.csv
             (5 episodes per file from the standard test scripts)

Output: Validate_Paired_TTest.xlsx with sheets:
  - Cost_TTest      : paired t-test on Total_Cost (lower = better)
  - FillRate_TTest   : paired t-test on Fill_Rate (higher = better)
  - Summary          : compact pass/fail table (10 nets x 3 baselines)
  - Methodology      : formulas, hypotheses, interpretation guide

Usage:
    python validate_t_test.py
    python validate_t_test.py --alpha 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "evaluation_results"
OUTPUT = ROOT / "Validate_Paired_TTest.xlsx"

NETWORKS: list[str] = [
    "1x3", "1x7", "1x10",
    "2x15", "2x20", "2x30", "2x40",
    "4x15", "4x30", "4x40",
]

BASELINES = ["(s,S) Policy", "MAPPO", "HAPPO"]
PROPOSED = "GNN-HAPPO"

# ---------------------------------------------------------------------------
# CSV path resolution (same logic as build_benchmark_comparison_xlsx.py)
# ---------------------------------------------------------------------------

def csv_path(model: str, net: str) -> Path:
    if model == "(s,S) Policy":
        return EVAL / f"basestock_{net}" / f"results_ss_heuristic_{net}.csv"
    if model == "MAPPO":
        stem = "results_mappo" if net == "2x15" else "results_standard_mappo"
        return EVAL / f"mappo_{net}" / f"{stem}_{net}.csv"
    if model == "HAPPO":
        return EVAL / f"happo_{net}" / f"results_standard_happo_{net}.csv"
    if model == "GNN-HAPPO":
        return EVAL / f"gnn_{net}" / f"results_gnn_happo_{net}.csv"
    raise ValueError(f"Unknown model {model!r}")


def load_episodes(model: str, net: str) -> pd.DataFrame:
    p = csv_path(model, net)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Paired t-test logic
# ---------------------------------------------------------------------------

def paired_ttest(gnn_vals: np.ndarray, baseline_vals: np.ndarray,
                 alpha: float, direction: str) -> dict:
    """Run paired t-test and return a result dict.

    direction:
      'lower'  -> H1: GNN < baseline (for cost — lower is better)
      'higher' -> H1: GNN > baseline (for fill rate — higher is better)

    Returns dict with:
      n, mean_gnn, mean_baseline, mean_diff, std_diff,
      t_stat, p_two_tailed, p_one_tailed, significant, conclusion
    """
    n = len(gnn_vals)
    diff = gnn_vals - baseline_vals  # paired differences
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    # scipy paired t-test (two-tailed)
    t_stat, p_two = stats.ttest_rel(gnn_vals, baseline_vals)
    t_stat = float(t_stat)
    p_two = float(p_two)

    # One-tailed p-value
    if direction == "lower":
        # H1: GNN < baseline → diff < 0 → left tail
        p_one = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    else:
        # H1: GNN > baseline → diff > 0 → right tail
        p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

    significant = p_one < alpha

    if direction == "lower":
        gnn_better = mean_diff < 0
        conclusion = ("GNN significantly lower cost"
                      if significant and gnn_better
                      else "GNN significantly higher cost"
                      if significant and not gnn_better
                      else "No significant difference")
    else:
        gnn_better = mean_diff > 0
        conclusion = ("GNN significantly higher fill rate"
                      if significant and gnn_better
                      else "GNN significantly lower fill rate"
                      if significant and not gnn_better
                      else "No significant difference")

    stars = ""
    if p_one < 0.001:
        stars = "***"
    elif p_one < 0.01:
        stars = "**"
    elif p_one < alpha:
        stars = "*"

    return {
        "n": n,
        "mean_gnn": float(np.mean(gnn_vals)),
        "mean_baseline": float(np.mean(baseline_vals)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "df": n - 1,
        "p_two_tailed": p_two,
        "p_one_tailed": p_one,
        "alpha": alpha,
        "significant": significant,
        "gnn_better": gnn_better if significant else None,
        "stars": stars,
        "conclusion": conclusion,
    }


def run_all_tests(alpha: float) -> list[dict]:
    """Run paired t-tests for every (baseline, network, metric) combo.

    Returns a list of flat dicts ready for DataFrame construction.
    """
    rows = []
    for net in NETWORKS:
        gnn_df = load_episodes(PROPOSED, net)
        gnn_cost = gnn_df["Total_Cost"].to_numpy(dtype=float)
        gnn_fill = gnn_df["Fill_Rate"].to_numpy(dtype=float)

        for baseline in BASELINES:
            bl_df = load_episodes(baseline, net)
            bl_cost = bl_df["Total_Cost"].to_numpy(dtype=float)
            bl_fill = bl_df["Fill_Rate"].to_numpy(dtype=float)

            cost_res = paired_ttest(gnn_cost, bl_cost, alpha, "lower")
            fill_res = paired_ttest(gnn_fill, bl_fill, alpha, "higher")

            rows.append({
                "Network": net,
                "Baseline": baseline,
                "Metric": "Total_Cost",
                **{f"cost_{k}": v for k, v in cost_res.items()},
            })
            rows.append({
                "Network": net,
                "Baseline": baseline,
                "Metric": "Fill_Rate",
                **{f"fill_{k}": v for k, v in fill_res.items()},
            })
    return rows


# ===========================================================================
# Excel builder
# ===========================================================================

HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
GNN_WIN = PatternFill("solid", fgColor="E2EFDA")
GNN_LOSE = PatternFill("solid", fgColor="F4CCCC")
NO_SIG = PatternFill("solid", fgColor="FFF2CC")
POS_FONT = Font(color="006100", bold=True)
NEG_FONT = Font(color="9C0006", bold=True)
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


def autosize(ws: Worksheet, widths: Iterable[int]) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_ttest_sheet(ws: Worksheet, results: list[dict], metric: str,
                      alpha: float) -> None:
    pfx = "cost_" if metric == "Total_Cost" else "fill_"
    direction_label = ("lower is better" if metric == "Total_Cost"
                       else "higher is better")
    title = (f"Paired t-test: GNN-HAPPO vs Baselines — {metric} "
             f"({direction_label}, α={alpha})")
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=13)

    headers = [
        "Network", "Baseline",
        "Mean GNN-HAPPO", "Mean Baseline", "Mean Diff (GNN−BL)",
        "Std Diff", "t-statistic", "df", "p (two-tailed)", "p (one-tailed)",
        "Sig?", "Conclusion",
    ]
    for j, h in enumerate(headers, start=1):
        ws.cell(row=3, column=j, value=h)
    style_header(ws, 3, len(headers))

    filtered = [r for r in results if r["Metric"] == metric]

    row = 4
    for net in NETWORKS:
        block_start = row
        for r_data in filtered:
            if r_data["Network"] != net:
                continue
            ws.cell(row=row, column=1, value=net).alignment = CENTER
            ws.cell(row=row, column=2, value=r_data["Baseline"]).alignment = LEFT

            ws.cell(row=row, column=3,
                    value=r_data[f"{pfx}mean_gnn"]).number_format = "#,##0.00"
            ws.cell(row=row, column=4,
                    value=r_data[f"{pfx}mean_baseline"]).number_format = "#,##0.00"
            ws.cell(row=row, column=5,
                    value=r_data[f"{pfx}mean_diff"]).number_format = "#,##0.00"
            ws.cell(row=row, column=6,
                    value=r_data[f"{pfx}std_diff"]).number_format = "#,##0.00"
            ws.cell(row=row, column=7,
                    value=r_data[f"{pfx}t_stat"]).number_format = "0.0000"
            ws.cell(row=row, column=8,
                    value=r_data[f"{pfx}df"]).number_format = "0"
            ws.cell(row=row, column=9,
                    value=r_data[f"{pfx}p_two_tailed"]).number_format = "0.000000"
            ws.cell(row=row, column=10,
                    value=r_data[f"{pfx}p_one_tailed"]).number_format = "0.000000"

            stars = r_data[f"{pfx}stars"]
            sig = r_data[f"{pfx}significant"]
            gnn_better = r_data[f"{pfx}gnn_better"]
            conclusion = r_data[f"{pfx}conclusion"]

            sig_cell = ws.cell(row=row, column=11, value=stars if stars else "n.s.")
            sig_cell.alignment = CENTER
            conc_cell = ws.cell(row=row, column=12, value=conclusion)
            conc_cell.alignment = LEFT

            # Row fill based on outcome
            if sig and gnn_better:
                fill = GNN_WIN
                conc_cell.font = POS_FONT
            elif sig and not gnn_better:
                fill = GNN_LOSE
                conc_cell.font = NEG_FONT
            else:
                fill = NO_SIG

            for c in range(1, len(headers) + 1):
                ws.cell(row=row, column=c).border = BORDER
                if c >= 3 and c <= 10:
                    ws.cell(row=row, column=c).alignment = RIGHT
            ws.cell(row=row, column=11).fill = fill
            ws.cell(row=row, column=12).fill = fill

            row += 1

        if row > block_start + 1:
            ws.merge_cells(start_row=block_start, start_column=1,
                           end_row=row - 1, end_column=1)
            ws.cell(row=block_start, column=1).alignment = CENTER

    autosize(ws, [10, 14, 16, 16, 18, 12, 12, 6, 16, 16, 8, 34])
    ws.freeze_panes = "C4"


def build_summary(ws: Worksheet, results: list[dict], alpha: float) -> None:
    ws.cell(row=1, column=1,
            value=f"Summary: Paired t-test Results (α={alpha})"
            ).font = Font(bold=True, size=14)

    ws.cell(row=3, column=1, value="TOTAL COST (lower is better for GNN)"
            ).font = Font(bold=True, size=12)
    cost_headers = ["Network", "(s,S) Policy", "MAPPO", "HAPPO"]
    for j, h in enumerate(cost_headers, start=1):
        ws.cell(row=4, column=j, value=h)
    style_header(ws, 4, len(cost_headers))

    cost_rows = [r for r in results if r["Metric"] == "Total_Cost"]

    for i, net in enumerate(NETWORKS):
        r = 5 + i
        ws.cell(row=r, column=1, value=net).alignment = CENTER
        ws.cell(row=r, column=1).border = BORDER
        for j, bl in enumerate(BASELINES, start=2):
            match = [x for x in cost_rows
                     if x["Network"] == net and x["Baseline"] == bl]
            if not match:
                continue
            d = match[0]
            sig = d["cost_significant"]
            better = d["cost_gnn_better"]
            stars = d["cost_stars"]
            gap_pct = (d["cost_mean_diff"] / d["cost_mean_baseline"] * 100.0
                       if d["cost_mean_baseline"] != 0 else 0)

            if sig and better:
                label = f"GNN wins ({gap_pct:+.1f}%) {stars}"
                fill = GNN_WIN
                font = POS_FONT
            elif sig and not better:
                label = f"GNN loses ({gap_pct:+.1f}%) {stars}"
                fill = GNN_LOSE
                font = NEG_FONT
            else:
                label = f"n.s. ({gap_pct:+.1f}%)"
                fill = NO_SIG
                font = Font()

            cell = ws.cell(row=r, column=j, value=label)
            cell.alignment = CENTER
            cell.fill = fill
            cell.font = font
            cell.border = BORDER

    # Count wins
    r_count = 5 + len(NETWORKS) + 1
    ws.cell(row=r_count, column=1, value="GNN wins / total").font = Font(bold=True)
    ws.cell(row=r_count, column=1).border = BORDER
    for j, bl in enumerate(BASELINES, start=2):
        wins = sum(1 for x in cost_rows
                   if x["Baseline"] == bl and x["cost_significant"]
                   and x["cost_gnn_better"])
        total = sum(1 for x in cost_rows if x["Baseline"] == bl)
        cell = ws.cell(row=r_count, column=j, value=f"{wins}/{total}")
        cell.alignment = CENTER
        cell.font = Font(bold=True)
        cell.border = BORDER

    # --- Fill Rate section ---
    fr_start = r_count + 3
    ws.cell(row=fr_start, column=1,
            value="FILL RATE (higher is better for GNN)"
            ).font = Font(bold=True, size=12)
    for j, h in enumerate(cost_headers, start=1):
        ws.cell(row=fr_start + 1, column=j, value=h)
    style_header(ws, fr_start + 1, len(cost_headers))

    fill_rows = [r for r in results if r["Metric"] == "Fill_Rate"]

    for i, net in enumerate(NETWORKS):
        r = fr_start + 2 + i
        ws.cell(row=r, column=1, value=net).alignment = CENTER
        ws.cell(row=r, column=1).border = BORDER
        for j, bl in enumerate(BASELINES, start=2):
            match = [x for x in fill_rows
                     if x["Network"] == net and x["Baseline"] == bl]
            if not match:
                continue
            d = match[0]
            sig = d["fill_significant"]
            better = d["fill_gnn_better"]
            stars = d["fill_stars"]
            diff_pp = d["fill_mean_diff"]

            if sig and better:
                label = f"GNN wins ({diff_pp:+.2f}pp) {stars}"
                fill = GNN_WIN
                font = POS_FONT
            elif sig and not better:
                label = f"GNN loses ({diff_pp:+.2f}pp) {stars}"
                fill = GNN_LOSE
                font = NEG_FONT
            else:
                label = f"n.s. ({diff_pp:+.2f}pp)"
                fill = NO_SIG
                font = Font()

            cell = ws.cell(row=r, column=j, value=label)
            cell.alignment = CENTER
            cell.fill = fill
            cell.font = font
            cell.border = BORDER

    r_count2 = fr_start + 2 + len(NETWORKS) + 1
    ws.cell(row=r_count2, column=1, value="GNN wins / total").font = Font(bold=True)
    ws.cell(row=r_count2, column=1).border = BORDER
    for j, bl in enumerate(BASELINES, start=2):
        wins = sum(1 for x in fill_rows
                   if x["Baseline"] == bl and x["fill_significant"]
                   and x["fill_gnn_better"])
        total = sum(1 for x in fill_rows if x["Baseline"] == bl)
        cell = ws.cell(row=r_count2, column=j, value=f"{wins}/{total}")
        cell.alignment = CENTER
        cell.font = Font(bold=True)
        cell.border = BORDER

    autosize(ws, [10, 28, 28, 28])
    ws.freeze_panes = "B5"


def build_methodology(ws: Worksheet, alpha: float) -> None:
    ws.cell(row=1, column=1, value="Methodology").font = Font(bold=True, size=14)
    lines = [
        ("", ""),
        ("Test", "Paired t-test (scipy.stats.ttest_rel)"),
        ("Pairing", "Episode i of GNN-HAPPO is paired with episode i of baseline. "
                     "All evaluations use the same random seed (42), so demand "
                     "realizations are identical across models for the same episode."),
        ("Sample size", "n = 5 episodes per (model, network) pair."),
        ("", ""),
        ("Total Cost hypotheses",
            "H0: mean(GNN_cost) = mean(Baseline_cost).  "
            "H1: mean(GNN_cost) < mean(Baseline_cost)  (one-tailed, lower cost = better)."),
        ("Fill Rate hypotheses",
            "H0: mean(GNN_fill) = mean(Baseline_fill).  "
            "H1: mean(GNN_fill) > mean(Baseline_fill)  (one-tailed, higher fill = better)."),
        ("", ""),
        ("Significance level", f"α = {alpha}"),
        ("p-value (two-tailed)", "Returned directly by scipy.stats.ttest_rel."),
        ("p-value (one-tailed)",
            "p_one = p_two / 2  if t-stat is in the hypothesized direction, "
            "else p_one = 1 - p_two / 2."),
        ("Stars", "* p < α,  ** p < 0.01,  *** p < 0.001"),
        ("", ""),
        ("Interpretation guide", ""),
        ("  Green (GNN wins)",
            "GNN-HAPPO is statistically significantly better than the baseline "
            "at the α level."),
        ("  Red (GNN loses)",
            "Baseline is statistically significantly better than GNN-HAPPO."),
        ("  Yellow (n.s.)",
            "No statistically significant difference detected. With n=5, "
            "the test has limited statistical power; a larger sample may "
            "reveal differences."),
        ("", ""),
        ("Data source",
            "evaluation_results/<model>_<net>/results_*_<net>.csv  "
            "(5 episodes, same seed=42 across all models)."),
        ("Cost Gap % in Summary",
            "mean_diff / mean_baseline × 100. Negative = GNN is cheaper."),
        ("Fill Gap (pp) in Summary",
            "mean_diff = mean(GNN_fill) - mean(Baseline_fill) in percentage points."),
        ("", ""),
        ("Limitation",
            "n=5 is a small sample. The paired design increases power by "
            "controlling for episode-level demand variation, but effect sizes "
            "must be large relative to within-pair variability to reach "
            "significance. Consider running more episodes (e.g., 30-100) for "
            "stronger statistical evidence."),
    ]
    for i, (k, v) in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=k).font = Font(bold=bool(k))
        ws.cell(row=i, column=1).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row=i, column=2, value=v).alignment = Alignment(vertical="top", wrap_text=True)
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 100


def write_excel(results: list[dict], out_path: Path, alpha: float) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    build_ttest_sheet(wb.create_sheet("Cost_TTest"), results, "Total_Cost", alpha)
    build_ttest_sheet(wb.create_sheet("FillRate_TTest"), results, "Fill_Rate", alpha)
    build_summary(wb.create_sheet("Summary"), results, alpha)
    build_methodology(wb.create_sheet("Methodology"), alpha)

    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot delete {out_path}. Close the file in Excel and re-run."
            ) from e
    wb.save(out_path)
    print(f"\n[OK] Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired t-test: GNN-HAPPO vs 3 baselines x 10 networks")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--out", type=str, default=str(OUTPUT),
                        help="Output Excel path")
    args = parser.parse_args()

    print("=" * 60)
    print("PAIRED t-TEST VALIDATION")
    print("=" * 60)
    print(f"  Proposed : {PROPOSED}")
    print(f"  Baselines: {', '.join(BASELINES)}")
    print(f"  Networks : {', '.join(NETWORKS)}")
    print(f"  α        : {args.alpha}")
    print("=" * 60)

    results = run_all_tests(args.alpha)

    # Console summary
    cost_results = [r for r in results if r["Metric"] == "Total_Cost"]
    fill_results = [r for r in results if r["Metric"] == "Fill_Rate"]

    print(f"\n{'Network':<8} {'Baseline':<14} {'Cost t':>8} {'Cost p1':>10} {'Sig':>5}  "
          f"{'Fill t':>8} {'Fill p1':>10} {'Sig':>5}")
    print("-" * 80)
    for net in NETWORKS:
        for bl in BASELINES:
            c = [x for x in cost_results
                 if x["Network"] == net and x["Baseline"] == bl][0]
            f = [x for x in fill_results
                 if x["Network"] == net and x["Baseline"] == bl][0]
            print(f"{net:<8} {bl:<14} "
                  f"{c['cost_t_stat']:>8.3f} {c['cost_p_one_tailed']:>10.6f} "
                  f"{c['cost_stars'] or 'n.s.':>5}  "
                  f"{f['fill_t_stat']:>8.3f} {f['fill_p_one_tailed']:>10.6f} "
                  f"{f['fill_stars'] or 'n.s.':>5}")

    write_excel(results, Path(args.out), args.alpha)


if __name__ == "__main__":
    main()
