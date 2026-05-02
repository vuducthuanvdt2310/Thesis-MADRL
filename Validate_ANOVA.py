#!/usr/bin/env python
"""
Validate_ANOVA.py
====================
ANOVA validation script for comparing four inventory management policies:
  • GNN-HAPPO        (proposed, GNN-based multi-agent RL)
  • Standard HAPPO   (baseline, MLP-based multi-agent RL)
  • MAPPO            (Multi-Agent PPO baseline)
  • (s,S) Heuristic  (classical inventory heuristic)

Input:  results_*.csv files specified in PATHS.
        Columns: Episode_Index, Total_Cost, Fill_Rate, Lost_Sales, Avg_Inventory

Output:
  • Basic descriptive statistics
  • ANOVA Table for each metric printed to console
  • Post-hoc comparisons (Tukey HSD) if significant difference is found
  • Excel file containing all tables and stats
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ===========================================================================
# ① CONFIGURABLE FILE PATHS  (Must match Validate_Results.py)
# ===========================================================================
PATHS = {
    "GNN-HAPPO":      "evaluation_results/eval_gnn/results_gnn_happo.csv",
    "Standard-HAPPO": "evaluation_results/eval_base/results_standard_happo.csv",
    "MAPPO":          "evaluation_results/eval_mappo/results_mappo.csv",
    "S-s-Heuristic":  "evaluation_results/basestock_v1/results_ss_heuristic.csv",
}

METRICS = ["Total_Cost", "Fill_Rate", "Lost_Sales", "Avg_Inventory"]
OUT_DIR = Path("validation_output")

def load_data(paths):
    data = {}
    for model_name, path_str in paths.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] File not found for '{model_name}': {p}")
            continue
        df = pd.read_csv(p)
        data[model_name] = df
        print(f"[OK]  Loaded {len(df):3d} episodes for '{model_name}'")
    return data

def run_anova(data, metric):
    # Collect data arrays
    arrays = []
    group_names = []
    for name, df in data.items():
        if metric in df.columns:
            arr = df[metric].dropna().values
            arrays.append(arr)
            group_names.append(name)
            
    if len(arrays) < 2:
        print(f"\n[WARN] Not enough data to run ANOVA for {metric}")
        return None, None, None
    
    # Calculate ANOVA statistics manually for the table
    all_data = np.concatenate(arrays)
    grand_mean = np.mean(all_data)
    
    # Between Groups
    ss_between = sum(len(arr) * (np.mean(arr) - grand_mean)**2 for arr in arrays)
    df_between = len(arrays) - 1
    ms_between = ss_between / df_between if df_between > 0 else 0
    
    # Within Groups
    ss_within = sum(np.sum((arr - np.mean(arr))**2) for arr in arrays)
    df_within = len(all_data) - len(arrays)
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # Total
    ss_total = ss_between + ss_within
    df_total = len(all_data) - 1
    
    # Use scipy to calculate F and p-value
    F_stat, p_value = stats.f_oneway(*arrays)
    
    # --- 1. Create Descriptive Stats DataFrame ---
    desc_rows = []
    for i, arr in enumerate(arrays):
        desc_rows.append({
            "Group": group_names[i],
            "N": len(arr),
            "Mean": np.mean(arr),
            "Std Dev": np.std(arr, ddof=1),
            "Variance": np.var(arr, ddof=1)
        })
    df_desc = pd.DataFrame(desc_rows)
    
    # Print Descriptive Statistics
    print(f"\n{'='*85}")
    print(f" ANOVA Analysis for Metric: {metric.upper()}")
    print(f"{'='*85}")
    print(f"{'Descriptive Statistics':^85}")
    print(f"{'-'*85}")
    print(f"{'Group':<20} | {'N':<5} | {'Mean':<15} | {'Std Dev':<15} | {'Variance':<15}")
    print(f"{'-'*85}")
    for _, row in df_desc.iterrows():
        print(f"{row['Group']:<20} | {row['N']:<5} | {row['Mean']:<15.4f} | {row['Std Dev']:<15.4f} | {row['Variance']:<15.4f}")
    
    # --- 2. Create ANOVA Table DataFrame ---
    anova_rows = [
        {"Source": "Between Groups", "SS": ss_between, "df": df_between, "MS": ms_between, "F": F_stat, "p-value": p_value},
        {"Source": "Within Groups", "SS": ss_within, "df": df_within, "MS": ms_within, "F": np.nan, "p-value": np.nan},
        {"Source": "Total", "SS": ss_total, "df": df_total, "MS": np.nan, "F": np.nan, "p-value": np.nan}
    ]
    df_anova = pd.DataFrame(anova_rows)
    
    # Print ANOVA table
    print(f"\n{'-'*85}")
    print(f"{'ANOVA Table':^85}")
    print(f"{'-'*85}")
    print(f"{'Source':<15} | {'SS':<18} | {'df':<5} | {'MS':<18} | {'F':<10} | {'p-value'}")
    print(f"{'-'*85}")
    print(f"{'Between Groups':<15} | {ss_between:<18.4f} | {df_between:<5} | {ms_between:<18.4f} | {F_stat:<10.4f} | {p_value:.4e}")
    print(f"{'Within Groups':<15} | {ss_within:<18.4f} | {df_within:<5} | {ms_within:<18.4f} | {'-':<10} | {'-'}")
    print(f"{'Total':<15} | {ss_total:<18.4f} | {df_total:<5} | {'-':<18} | {'-':<10} | {'-'}")
    print(f"{'-'*85}")
    
    # --- 3. Create Post-Hoc DataFrame ---
    df_posthoc = pd.DataFrame()
    if p_value < 0.05:
        print(f"\n[Result] Statistically SIGNIFICANT difference found for {metric} (p < 0.05).")
        try:
            res = stats.tukey_hsd(*arrays)
            print("\n  Tukey HSD Post-hoc Test:")
            print("  Group Mappings:")
            for i, name in enumerate(group_names):
                print(f"    {i}: {name}")
            print("\n  Test Results:")
            print(res)
            
            posthoc_rows = []
            
            # Try to get q_crit for HSD threshold
            try:
                k = len(arrays)
                q_crit = stats.studentized_range.ppf(0.95, k, df_within)
            except AttributeError:
                q_crit = None
                
            for i in range(len(arrays)):
                for j in range(i + 1, len(arrays)):
                    # Absolute mean difference
                    mean_diff = np.abs(np.mean(arrays[i]) - np.mean(arrays[j]))
                    
                    pval = res.pvalue[i, j]
                    sig = "Yes" if pval < 0.05 else "No"
                    
                    if q_crit is not None:
                        hsd_thresh = q_crit * np.sqrt(ms_within / 2.0 * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j])))
                    else:
                        hsd_thresh = "N/A"
                        
                    posthoc_rows.append({
                        "Comparison Pair (Model A vs. Model B)": f"{group_names[i]} vs. {group_names[j]}",
                        "Absolute Mean Difference ($|\\bar{X}_A - \\bar{X}_B|$)": mean_diff,
                        "HSD Threshold": hsd_thresh,
                        "p-value": pval,
                        "Significance": sig
                    })
            df_posthoc = pd.DataFrame(posthoc_rows)
            
        except Exception as e:
            print(f"\n  [WARN] stats.tukey_hsd structured extraction failed: {e}")
            print("  Performing pairwise t-tests with Bonferroni correction instead:")
            posthoc_rows = []
            num_comparisons = (len(arrays) * (len(arrays) - 1)) / 2
            alpha_corrected = 0.05 / num_comparisons
            for i in range(len(arrays)):
                for j in range(i + 1, len(arrays)):
                    mean_diff = np.abs(np.mean(arrays[i]) - np.mean(arrays[j]))
                    t_stat, p_val = stats.ttest_ind(arrays[i], arrays[j])
                    sig = "Yes" if p_val < alpha_corrected else "No"
                    print(f"    {group_names[i]:<15} vs {group_names[j]:<15} | t={t_stat:+.4f} | p={p_val:.2e} | {sig} (alpha={alpha_corrected:.4f})")
                    posthoc_rows.append({
                        "Comparison Pair (Model A vs. Model B)": f"{group_names[i]} vs. {group_names[j]}",
                        "Absolute Mean Difference ($|\\bar{X}_A - \\bar{X}_B|$)": mean_diff,
                        "HSD Threshold": "N/A (Bonferroni)",
                        "p-value": p_val,
                        "Significance": sig
                    })
            df_posthoc = pd.DataFrame(posthoc_rows)
    else:
        print(f"\n[Result] NO statistically significant difference for {metric} (p >= 0.05).")
        df_posthoc = pd.DataFrame([{"Result": "No significant difference found"}])

    return df_desc, df_anova, df_posthoc


def main():
    print("\n" + "=" * 70)
    print("  ANOVA VALIDATION SCRIPT  —  Multi-Echelon Inventory Comparison")
    print("=" * 70 + "\n")
    
    data = load_data(PATHS)
    if not data:
        print("No valid CSV files found in the defined PATHS.")
        return
        
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # We will try to save to Excel. If openpyxl is missing, save to CSVs.
    excel_path = OUT_DIR / "ANOVA_results.xlsx"
    all_results = {}
    
    for metric in METRICS:
        df_desc, df_anova, df_posthoc = run_anova(data, metric)
        if df_desc is not None:
            all_results[metric] = {
                "Descriptive Stats": df_desc,
                "ANOVA Table": df_anova,
                "Post-Hoc": df_posthoc
            }
            
    print("\n" + "=" * 70)
    print("  ANOVA Analysis Complete")
    
    # Try exporting to Excel
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for metric, tables in all_results.items():
                # Write each table to the same sheet, separated by some rows
                sheet_name = metric[:31] # Excel limits sheet name to 31 chars
                start_row = 0
                
                tables["Descriptive Stats"].to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += len(tables["Descriptive Stats"]) + 3
                
                # Write title for ANOVA table
                df_title_anova = pd.DataFrame([["ANOVA Table"]])
                df_title_anova.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                start_row += 1
                tables["ANOVA Table"].to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += len(tables["ANOVA Table"]) + 3
                
                # Write title for Post-Hoc
                df_title_posthoc = pd.DataFrame([["Post-Hoc Comparisons"]])
                df_title_posthoc.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                start_row += 1
                tables["Post-Hoc"].to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        
        print(f"  Results successfully exported to Excel: {excel_path}")
    except ModuleNotFoundError:
        print(f"\n  [WARN] 'openpyxl' module not found. Saving results to CSVs instead.")
        for metric, tables in all_results.items():
            safe_metric = metric.replace(" ", "_")
            tables["Descriptive Stats"].to_csv(OUT_DIR / f"ANOVA_{safe_metric}_DescStats.csv", index=False)
            tables["ANOVA Table"].to_csv(OUT_DIR / f"ANOVA_{safe_metric}_Table.csv", index=False)
            tables["Post-Hoc"].to_csv(OUT_DIR / f"ANOVA_{safe_metric}_PostHoc.csv", index=False)
        print(f"  Results exported to CSVs in: {OUT_DIR.resolve()}")
    except Exception as e:
        print(f"\n  [ERROR] Failed to save Excel file: {e}")
        
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
