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

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_EPISODES = 10   # number of episodes per learning rate value for box-plot statistics
N_DCS        = 2    # number of DC agents (must match training config)
# ──────────────────────────────────────────────────────────────────────────────

def get_args(model_dir, num_episodes=NUM_EPISODES):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--config_path', type=str, default='configs/multi_sku_config.yaml')
    parser.add_argument('--num_episodes', type=int, default=num_episodes)
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


def extract_episode_metrics(episode_metrics, n_dcs, n_agents):
    """
    Extract per-episode true total cost and fill rate from episode_metrics list.

    True Total Cost = sum of (holding + backlog + ordering) costs across all agents.
    Fill Rate       = pooled order-count fill rate across all retailer agents (%).
    """
    total_costs = []
    fill_rates  = []

    for m in episode_metrics:
        # True total cost: sum of individual cost components (matches CSV logic)
        true_cost = (
            float(np.sum(m['holding_costs']))
            + float(np.sum(m['backlog_costs']))
            + float(np.sum(m['ordering_costs']))
        )
        total_costs.append(true_cost)

        # Pooled fill rate across all retailer agents for this episode
        ep_placed     = sum(m['_orders_placed'][aid]     for aid in range(n_dcs, n_agents))
        ep_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(n_dcs, n_agents))
        fr = (ep_from_stock / ep_placed * 100.0) if ep_placed > 0 else 100.0
        fill_rates.append(fr)

    return total_costs, fill_rates


def apply_custom_style(ax):
    """Consistent axis styling."""
    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)


def plot_boxplot(data_dict, labels, ylabel, title, out_file, colors, show_mean=True):
    """
    Draw a styled box plot.

    Parameters
    ----------
    data_dict : dict[str, list[float]]
        Ordered dict mapping label -> list of per-episode values.
    labels    : list[str]   X-axis tick labels.
    ylabel    : str
    title     : str
    out_file  : str         Output PNG filename.
    colors    : list[str]   One colour per box.
    show_mean : bool        Overlay a mean marker.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    data_list = [data_dict[k] for k in data_dict]

    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        notch=False,
        widths=0.45,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=5, alpha=0.5),
    )

    # Colour each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Match flier colour to box
    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor('black')

    # Mean markers
    if show_mean:
        for i, vals in enumerate(data_list):
            mean_val = np.mean(vals)
            ax.scatter(i + 1, mean_val, marker='D', s=60,
                       color='white', edgecolors='black', linewidths=1.5,
                       zorder=5, label='Mean' if i == 0 else '')

    apply_custom_style(ax)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    if show_mean:
        ax.legend(loc='upper right', frameon=True, edgecolor='lightgray', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved {out_file}")
    plt.close(fig)


def main():
    lrs = ["0.0001", "0.0005", "0.001"]
    model_dirs = {
        "0.0001": "results/lr_sensi/actor_lr_0.0001/run_seed_1/models",
        "0.0005": "results/lr_sensi/actor_lr_0.0005/run_seed_1/models",
        "0.001":  "results/lr_sensi/actor_lr_0.001/run_seed_1/models",
    }

    # Storage: lr -> list of per-episode values
    all_total_costs = {}
    all_fill_rates  = {}

    for lr in lrs:
        mdir = model_dirs[lr]
        if not os.path.exists(mdir):
            print(f"[SKIP] Directory not found: {mdir}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating Learning Rate={lr}  ({NUM_EPISODES} episodes) ...")
        print(f"{'='*60}")

        args      = get_args(mdir, num_episodes=NUM_EPISODES)
        evaluator = GNNModelEvaluator(args)
        evaluator.evaluate()

        n_agents = evaluator.n_agents

        ep_costs, ep_frs = extract_episode_metrics(
            evaluator.episode_metrics, N_DCS, n_agents
        )
        all_total_costs[lr] = ep_costs
        all_fill_rates[lr]  = ep_frs

        print(f"  Total Cost  — mean: {np.mean(ep_costs):,.2f}  std: {np.std(ep_costs):,.2f}")
        print(f"  Fill Rate   — mean: {np.mean(ep_frs):.2f}%  std: {np.std(ep_frs):.2f}%")

    if not all_total_costs:
        print("No data collected. Exiting.")
        return

    # Only plot learning rates that were evaluated
    eval_lrs = [lr for lr in lrs if lr in all_total_costs]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(eval_lrs)]
    labels = [f'Learning Rate\n{lr}' for lr in eval_lrs]

    # ── Box Plot 1: Total Cost ─────────────────────────────────────────────────
    plot_boxplot(
        data_dict = {lr: all_total_costs[lr] for lr in eval_lrs},
        labels    = labels,
        ylabel    = 'Total Cost (000VND)',
        title     = f'Total Cost Distribution Across {NUM_EPISODES} Episodes\n(Learning Rate Sensitivity)',
        out_file  = 'lr_cost_boxplot.png',
        colors    = colors,
        show_mean = True,
    )

    # ── Box Plot 2: Fill Rate ──────────────────────────────────────────────────
    plot_boxplot(
        data_dict = {lr: all_fill_rates[lr] for lr in eval_lrs},
        labels    = labels,
        ylabel    = 'Fill Rate (%)',
        title     = f'Fill Rate Distribution Across {NUM_EPISODES} Episodes\n(Learning Rate Sensitivity)',
        out_file  = 'lr_fill_rate_boxplot.png',
        colors    = colors,
        show_mean = True,
    )

    # ── Summary Table ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Learning Rate':<15} {'Cost Mean':>12} {'Cost Std':>10} {'FR Mean':>10} {'FR Std':>8}")
    print("-"*65)
    for lr in eval_lrs:
        print(
            f"{lr:<15}"
            f"{np.mean(all_total_costs[lr]):>12,.2f}"
            f"{np.std(all_total_costs[lr]):>10,.2f}"
            f"{np.mean(all_fill_rates[lr]):>10.2f}%"
            f"{np.std(all_fill_rates[lr]):>8.2f}%"
        )
    print("="*65)


if __name__ == "__main__":
    main()
