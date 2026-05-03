import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_trained_model_gnn import GNNModelEvaluator

def get_args(model_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--config_path', type=str, default='configs/multi_sku_config.yaml')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=90)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='evaluation_results_temp')
    parser.add_argument('--experiment_name', type=str, default='temp_eval')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--gnn_type', type=str, default='GAT')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--use_residual', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--critic_pooling', type=str, default='mean')
    return parser.parse_known_args([])[0]

def main():
    entropies = ["0.001", "0.01", "0.05"]
    model_dirs = {
        "0.001": "results_entropy/gnn_happo_sensitivity_entropy_0.001/run_seed_2/models",
        "0.01": "results_entropy/gnn_happo_sensitivity_entropy_0.01/run_seed_2/models",
        "0.05": "results_entropy/gnn_happo_sensitivity_entropy_0.05/run_seed_2/models"
    }

    trajectories = {}

    for ent, mdir in model_dirs.items():
        if not os.path.exists(mdir):
            print(f"Directory not found: {mdir}")
            continue
            
        print(f"Evaluating model for entropy {ent}...")
        args = get_args(mdir)
        evaluator = GNNModelEvaluator(args)
        evaluator.evaluate()
        
        trajectories[ent] = evaluator.detailed_trajectory
        
    if not trajectories:
        print("No trajectories found. Exiting.")
        return

    # Process data for plotting
    total_costs = []
    fill_rates = []
    
    for ent in entropies:
        if ent not in trajectories:
            continue
        traj = trajectories[ent]
        n_agents = len(traj['rewards'])
        n_dcs = 2 # Assuming 2 DCs
        
        # Total cost for the episode
        ep_cost = 0
        for t in range(90):
            ep_cost += sum(-traj['rewards'][aid][t] for aid in range(n_agents))
        total_costs.append(ep_cost)
            
        # Overall fill rate for the episode
        ep_placed = 0
        ep_stock = 0
        for t in range(90):
            ep_placed += sum(np.sum(traj['orders_placed'][aid][t]) for aid in range(n_dcs, n_agents))
            ep_stock += sum(np.sum(traj['orders_from_stock'][aid][t]) for aid in range(n_dcs, n_agents))
            
        fr = (ep_stock / ep_placed * 100.0) if ep_placed > 0 else 100.0
        fill_rates.append(fr)

    # Styling helper for the requested format
    def apply_custom_style(ax):
        ax.set_facecolor('white')
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='lightgray')
        ax.xaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Entropy 0.001', 'Entropy 0.01', 'Entropy 0.05']
    x_positions = np.arange(len(entropies))

    # --- Chart 1: Bar Chart for Total Cost ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(x_positions, total_costs, width=0.5, color=colors, edgecolor='black', alpha=0.85)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(total_costs)*0.01),
                 f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

    apply_custom_style(ax1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Cost for Different Entropy Values', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('entropy_cost_bar.png', dpi=300, bbox_inches='tight')
    print("Saved entropy_cost_bar.png")
    plt.close(fig1)

    # --- Chart 2: Line Chart for Fill Rate (Like the requested format) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Plotting line with markers
    # Using 'd-' to match the GNN-HAPPO style in the user's reference image
    ax2.plot(x_positions, fill_rates, marker='d', markersize=8, linestyle='-', linewidth=2, 
             color='#27ae60', label='GNN-HAPPO')
             
    # Add value labels
    for i, fr in enumerate(fill_rates):
        ax2.text(x_positions[i], fr + 0.5, f'{fr:.1f}%', ha='center', va='bottom', color='#27ae60', fontweight='bold')

    apply_custom_style(ax2)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Fill Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fill Rate for Different Entropy Values', fontsize=14, fontweight='bold', pad=15)
    
    # Target line if needed, or adjust y-limit for visual spacing
    ax2.set_ylim([min(fill_rates) - 5, min(100.5, max(fill_rates) + 5)])
    
    # Legend like the image
    ax2.legend(loc='upper left', frameon=True, edgecolor='lightgray')

    plt.tight_layout()
    plt.savefig('entropy_fill_rate_line.png', dpi=300, bbox_inches='tight')
    print("Saved entropy_fill_rate_line.png")
    plt.close(fig2)

if __name__ == "__main__":
    main()
