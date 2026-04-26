import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def replot(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    variations = df['Variation'].values
    model_names = [c for c in df.columns if c != 'Variation']
    
    # Normalization factors (episode_length * n_agents)
    # These should match the args in lead_time_comparison.py
    n_agents = 17
    norm_factors = {
        'Base Stock': 120 * n_agents,
        'HAPPO':      115 * n_agents,
        'MAPPO':      130 * n_agents,
        'GNN-HAPPO':  90 * n_agents
    }
    
    plt.figure(figsize=(8, 6))
    
    plot_configs = {
        'GNN-HAPPO': {'label': 'GNN-HAPPO', 'marker': '.', 'color': '#3A86FF', 'markersize': 8},
        'HAPPO':     {'label': 'HAPPO',    'marker': '^', 'color': '#D67D4B', 'markersize': 8},
        'MAPPO':     {'label': 'MAPPO', 'marker': 'd', 'color': '#3BB273', 'markersize': 8},
        'Base Stock':{'label': 'Base Stock', 'marker': '*', 'color': '#C04141', 'markersize': 8}
    }
    
    model_order = ['GNN-HAPPO', 'HAPPO', 'MAPPO', 'Base Stock']
    
    for model in model_order:
        if model in df.columns:
            # Normalize: Average daily cost per agent
            vals = df[model].values / norm_factors.get(model, 1.0)
            cfg = plot_configs[model]
            plt.plot(variations, vals, 
                     label=cfg['label'], 
                     marker=cfg['marker'], 
                     color=cfg['color'], 
                     markersize=cfg['markersize'],
                     linewidth=2)
    
    plt.xlabel('Lead time variation', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    
    # Direct labeling of lead time distributions
    x_labels = ["U[6, 13]", "U[7, 14]", "U[8, 15]", "U[9, 16]", "U[10, 17]"]
    plt.xticks(variations, x_labels, fontsize=10)
    
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Matching the image's style
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = Path(save_dir) / 'lead_time_cost_comparison_normalized.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Normalized comparison plot saved to: {out_path}")

if __name__ == '__main__':
    csv_path = 'evaluation_results/lead_time_comparison/lead_time_comparison_results.csv'
    save_dir = 'evaluation_results/lead_time_comparison'
    if Path(csv_path).exists():
        replot(csv_path, save_dir)
    else:
        print(f"CSV not found at {csv_path}")
