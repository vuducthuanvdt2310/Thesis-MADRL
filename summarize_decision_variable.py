import pandas as pd
import argparse
from pathlib import Path
import os

def summarize_decision_variables(excel_path):
    print(f"Reading data from: {excel_path}")
    
    try:
        # Read the trajectory data
        df = pd.read_excel(excel_path, sheet_name='Data')
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        return

    # Identify order columns (decision variables)
    order_cols = [col for col in df.columns if col.startswith('order_')]
    
    if not order_cols:
        print("No decision variable columns (order_*) found in the dataset.")
        return
    
    print("\n" + "="*70)
    print("DECISION VARIABLE (ORDER QUANTITY) SUMMARY".center(70))
    print("="*70)
    
    # 1. Average Order Quantity per Agent
    print("\n1. Average Order Quantity per Agent (Per Step):")
    print("-" * 70)
    agent_avg = df.groupby(['agent_id', 'agent'])[order_cols].mean().round(2)
    print(agent_avg.to_string())
    
    # 2. Total Order Quantity per Agent
    print("\n2. Total Order Quantity per Agent (Over Full Episode):")
    print("-" * 70)
    agent_total = df.groupby(['agent_id', 'agent'])[order_cols].sum().round(2)
    print(agent_total.to_string())

    # 3. Overall Statistics Across All Steps
    print("\n3. Overall Statistics for Ordering Quantities (Across All Steps):")
    print("-" * 70)
    overall_stats = df[order_cols].describe().round(2)
    print(overall_stats.to_string())

    # 4. Generate Thesis Report Format
    print("\n4. Generating Thesis Report Format...")
    print("-" * 70)
    
    steps = sorted(df['step'].unique())
    thesis_rows = []
    
    for step in steps:
        step_df = df[df['step'] == step].sort_values('agent_id')
        agent_strings = []
        for _, row in step_df.iterrows():
            vals = [str(round(row[col], 2)) for col in order_cols]
            agent_strings.append(f"({','.join(vals)})")
            
        full_string = " ; ".join(agent_strings)
        
        thesis_rows.append({
            'Step': f"step {step}",
            'Variable': f"qik{step}",
            'Orders (All Agents)': full_string
        })
        
    thesis_df = pd.DataFrame(thesis_rows)
    print(f"Generated thesis format for {len(steps)} steps.")

    # 5. Save summary to a new excel file to make it readable
    output_path = Path(excel_path).parent / f"summary_decision_variables.xlsx"
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            thesis_df.to_excel(writer, sheet_name="Thesis_Format", index=False)
            agent_avg.to_excel(writer, sheet_name="Avg_Order_Per_Agent")
            agent_total.to_excel(writer, sheet_name="Total_Order_Per_Agent")
            overall_stats.to_excel(writer, sheet_name="Overall_Stats")
            
            # Additional detail: Step-by-step summary
            step_summary = df.groupby('step')[order_cols].sum().round(2)
            step_summary.to_excel(writer, sheet_name="Total_Order_Per_Step")
            
        print(f"\n[OK] Excel summary successfully generated and saved to:\n     {output_path}")
        print("     (It contains multiple sheets: Thesis_Format, Avg_Order_Per_Agent, Total_Order_Per_Agent, Overall_Stats, Total_Order_Per_Step)")
    except Exception as e:
        print(f"\n[ERROR] Failed to write summary Excel file: {e}")

def find_latest_trajectory_file(base_dir="."):
    """Find the most recently created step_trajectory_ep1.xlsx file."""
    search_pattern = "step_trajectory_ep1.xlsx"
    latest_file = None
    latest_time = 0
    
    for root, dirs, files in os.walk(base_dir):
        if search_pattern in files:
            file_path = os.path.join(root, search_pattern)
            mtime = os.path.getmtime(file_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file_path
                
    return latest_file

if __name__ == "__main__":
    # =========================================================================
    # OPTION 1: HARDCODE YOUR FILE PATH HERE
    # If you want to specify a fixed path, change None to your file path string.
    # Example: hardcoded_path = r"D:\thuan\thesis\...\step_trajectory_ep1.xlsx"
    # =========================================================================
    hardcoded_path = r"D:\thuan\thesis\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management\evaluation_results\eval_gnn\step_trajectory_ep1.xlsx"

    parser = argparse.ArgumentParser(description="Summarize decision variables (order quantity) from trajectory excel.")
    parser.add_argument("--file", type=str, help="Path to the step_trajectory_ep1.xlsx file. If not provided, finds the latest one.")
    args = parser.parse_args()
    
    # OPTION 2: PASS VIA COMMAND LINE (e.g., python script.py --file "path.xlsx")
    target_file = hardcoded_path or args.file
    
    if not target_file:
        print("No file specified. Searching for the most recent step_trajectory_ep1.xlsx...")
        # First check evaluation_results folder
        target_file = find_latest_trajectory_file("evaluation_results")
        
        if not target_file:
            # Fallback to the whole current directory
            target_file = find_latest_trajectory_file(".")
            
    if target_file:
        summarize_decision_variables(target_file)
    else:
        print("Could not find any step_trajectory_ep1.xlsx files.")
        print("Please run an evaluation first or provide the path via --file argument.")
