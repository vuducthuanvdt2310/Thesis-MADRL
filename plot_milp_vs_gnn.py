import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi

# ---------------------------------------------------------
# THESIS PRESENTATION PLOTS: MILP vs GNN-HAPPO
# ---------------------------------------------------------

# Set aesthetic style for academic/presentation quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") # Larger fonts for presentations

# Colors for presentation
COLOR_MILP = '#2ECC71'   # Green (Optimal)
COLOR_GNN = '#3498DB'    # Blue (Proposed)
COLOR_BASE = '#E74C3C'   # Red (Baseline)

def load_or_mock_data():
    """
    Attempts to load the actual CSV files. If not found, uses realistic dummy data
    so you can see the charts immediately. You can replace the paths below with your actual files.
    """
    milp_path = "evaluation_results/milp_benchmark/results_milp_90d.csv"
    gnn_path = "evaluation_results/gnn_evaluation/results.csv" # Change this to your actual GNN csv path
    
    try:
        df_milp = pd.read_csv(milp_path).iloc[0]
        # df_gnn = pd.read_csv(gnn_path).iloc[0]
        # For demonstration, we will just use dummy GNN data if the path is wrong
        raise Exception("Force fallback for GNN data demo")
    except:
        print("[Info] Using example data for plotting. Update file paths to use real data.")
        # MOCK DATA (Replace with your actual GNN vs MILP results)
        df_milp = pd.Series({
            'Total_Cost': 210287.9387,
            'Fill_Rate': 100,
            'Lost_Sales': 0,
            'Avg_Inventory': 26.6,
            'Total_Holding_Cost': 5197.0,
            'Total_Backlog_Cost': 62.0,
            'Total_Ordering_Cost': 205028.0
        })
        df_gnn = pd.Series({
            'Total_Cost': 238387.0,
            'Fill_Rate': 94.5,
            'Lost_Sales': 161,
            'Avg_Inventory': 149.8,
            'Total_Holding_Cost': 64713.0,
            'Total_Backlog_Cost': 3281.0,
            'Total_Ordering_Cost': 170391.0
        })
        
    return df_milp, df_gnn

def plot_cost_breakdown(df_milp, df_gnn, save_dir):
    """
    Stacked bar chart showing HOW the costs differ.
    This tells the teacher if the GNN is holding too much safety stock or ordering too frequently.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    labels = ['MILP', 'Proposed GNN-HAPPO']
    holding = [df_milp['Total_Holding_Cost'], df_gnn['Total_Holding_Cost']]
    backlog = [df_milp['Total_Backlog_Cost'], df_gnn['Total_Backlog_Cost']]
    ordering = [df_milp['Total_Ordering_Cost'], df_gnn['Total_Ordering_Cost']]
    
    width = 0.5
    
    # Create stacked bars
    p1 = ax.bar(labels, holding, width, label='Holding Cost', color='#FFFF00')
    p2 = ax.bar(labels, ordering, width, bottom=holding, label='Ordering Cost', color='#0000FF')
    p3 = ax.bar(labels, backlog, width, bottom=np.array(holding)+np.array(ordering), label='Backlog Cost', color='#e74c3c')
    
    # Add values on top of bars
    for i, total in enumerate([df_milp['Total_Cost'], df_gnn['Total_Cost']]):
        ax.text(i, total + 50, f'Total: ${total:,.0f}', ha='center', fontweight='bold', fontsize=14)
        
    ax.set_ylabel('Total System Cost ($)')
    ax.set_title('Cost Breakdown Analysis: Optimal vs Proposed Method', pad=20, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cost_breakdown.png'), dpi=300)
    plt.close()

def plot_kpi_radar(df_milp, df_gnn, save_dir):
    """
    Spider/Radar chart comparing the models across multiple dimensions.
    Requires normalizing the metrics so 100% is best.
    """
    # Define metrics and their "Best" (Optimal) and "Worst" theoretical bounds for normalization
    metrics = ['Fill Rate', 'Cost Efficiency', 'Inventory Leaness', 'Lost Sales Avoidance']
    
    # Normalization math (Higher is better)
    # Fill Rate is already percentage
    fr_milp, fr_gnn = df_milp['Fill_Rate'], df_gnn['Fill_Rate']
    
    # Cost Efficiency (MILP = 100%, lower cost = better)
    ce_milp = 100.0
    ce_gnn = (df_milp['Total_Cost'] / df_gnn['Total_Cost']) * 100.0
    
    # Inventory Leaness (MILP = 100%)
    il_milp = 100.0
    il_gnn = (df_milp['Avg_Inventory'] / df_gnn['Avg_Inventory']) * 100.0
    
    # Lost Sales (MILP = 100%, 0 lost sales = 100%)
    max_lost = max(df_milp['Lost_Sales'], df_gnn['Lost_Sales']) * 2 # arbitrary scale
    ls_milp = 100 - (df_milp['Lost_Sales'] / max_lost * 100)
    ls_gnn = 100 - (df_gnn['Lost_Sales'] / max_lost * 100)
    
    values_milp = [fr_milp, ce_milp, il_milp, ls_milp]
    values_gnn = [fr_gnn, ce_gnn, il_gnn, ls_gnn]
    
    # Radar chart math
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    values_milp += values_milp[:1]
    values_gnn += values_gnn[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw MILP
    ax.plot(angles, values_milp, linewidth=2, linestyle='solid', label='MILP (Upper Bound)', color=COLOR_MILP)
    ax.fill(angles, values_milp, COLOR_MILP, alpha=0.1)
    
    # Draw GNN
    ax.plot(angles, values_gnn, linewidth=2, linestyle='solid', label='GNN-HAPPO', color=COLOR_GNN)
    ax.fill(angles, values_gnn, COLOR_GNN, alpha=0.25)
    
    # Labels
    plt.xticks(angles[:-1], metrics, color='grey', size=14, fontweight='bold')
    ax.set_rlabel_position(30)
    plt.yticks([20, 40, 60, 80, 100], ["20","40","60","80","100"], color="grey", size=10)
    plt.ylim(0, 110)
    
    plt.title('Performance Profiling vs Absolute Optimal Baseline', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=300)
    plt.close()

def plot_efficiency_gap(df_milp, df_gnn, save_dir):
    """
    A visual representation of the 'Optimality Gap'.
    Teachers love to see how close the RL agent gets to the theoretical maximum.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    gap_percent = ((df_gnn['Total_Cost'] - df_milp['Total_Cost']) / df_milp['Total_Cost']) * 100
    
    bars = ax.bar(['MILP', 'GNN-HAPPO Agent'], 
                  [df_milp['Total_Cost'], df_gnn['Total_Cost']],
                  color=[COLOR_MILP, COLOR_GNN],
                  width=0.6)
    
    # Draw gap line
    x1, x2 = 0, 1
    y1, y2 = df_milp['Total_Cost'], df_gnn['Total_Cost']
    
    ax.plot([x1, x2], [y1, y1], color='grey', linestyle='--', alpha=0.7)
    
    # Annotate the gap
    mid_x = (x1 + x2) / 2
    ax.annotate(f'+{gap_percent:.1f}% Optimality Gap', 
                xy=(1, y1 + (y2-y1)/2), 
                xytext=(0.5, y1 + (y2-y1)/2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2),
                ha='center', va='center', fontweight='bold', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    ax.set_ylabel('Total System Cost ($)')
    ax.set_title('The Optimality Gap (Cost Minimization)', pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimality_gap.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating Presentation Charts...")
    save_dir = "evaluation_results/presentation_charts"
    os.makedirs(save_dir, exist_ok=True)
    
    df_milp, df_gnn = load_or_mock_data()
    
    plot_cost_breakdown(df_milp, df_gnn, save_dir)
    print("1. Generated Cost Breakdown (Stacked Bar)")
    
    plot_kpi_radar(df_milp, df_gnn, save_dir)
    print("2. Generated KPI Radar Chart")
    
    plot_efficiency_gap(df_milp, df_gnn, save_dir)
    print("3. Generated Optimality Gap Chart")
    
    print(f"\n[Success] All charts saved in high resolution to: '{save_dir}'")
    print("You can copy these .png files directly into your PowerPoint/Thesis!")
