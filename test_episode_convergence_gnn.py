#!/usr/bin/env python
"""
Episode Convergence Analysis for GNN-HAPPO Models
====================================================

This script determines the minimum number of evaluation episodes needed
for stable (converged) cost statistics.  It re-uses the GNNModelEvaluator
from ``test_trained_model_gnn.py`` and runs evaluations at:

    num_episodes = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

For **each** setting the script records:
  - mean total cost
  - std  total cost
  - min / max total cost
  - coefficient of variation (CV = std / |mean|)

All results are saved to a single Excel file with:
  - Sheet "Summary"   : one row per num_episodes setting
  - Sheet "Episode_Costs" : raw per-episode cost for the largest run (150 eps)
  - Sheet "Rolling_Std"   : rolling std computed as episodes accumulate (1→150)

A convergence plot (std vs. num_episodes) is also saved.

Usage
-----
    python test_episode_convergence_gnn.py \
        --model_dir results/5Mar_1_gnn/run_seed_1/models \
        --episode_length 90 \
        --seed 42 \
        --save_dir convergence_results

NOTE: The script runs the LARGEST num_episodes value (150) ONCE and then
sub-samples the first N episodes to compute statistics for each N, so total
wall-clock time ≈ one run of 150 episodes (not 13 separate runs).
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Re-use the GNN evaluator (import from the existing file)
# ---------------------------------------------------------------------------
from test_trained_model_gnn import GNNModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Episode Convergence Analysis for GNN-HAPPO'
    )

    # Required
    parser.add_argument('--model_dir', type=str, required=False, default='results/14Apr_gnn_kaggle_vari/run_seed_1/models',
                        help='Path to saved model directory')
    parser.add_argument('--config_path', type=str,
                        default='configs/multi_sku_config.yaml',
                        help='Env config path (kept for compatibility)')

    # Episode settings
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Length of each episode in days')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Convergence sweep
    parser.add_argument('--min_episodes', type=int, default=10,
                        help='Minimum num_episodes to test (default: 30)')
    parser.add_argument('--max_episodes', type=int, default=150,
                        help='Maximum num_episodes to test (default: 150)')
    parser.add_argument('--step_episodes', type=int, default=5,
                        help='Step between num_episodes values (default: 10)')

    # Output
    parser.add_argument('--save_dir', type=str, default='convergence_results',
                        help='Directory to save convergence results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this run (default: auto timestamp)')

    # Hardware
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA if available')

    # GNN architecture (must match training config)
    parser.add_argument('--gnn_type', type=str, default='GAT')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--use_residual',
                        type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--critic_pooling', type=str, default='mean')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_convergence_analysis(args):
    """Run the full convergence sweep."""

    # Determine num_episodes values to test
    ep_values = list(range(args.min_episodes,
                           args.max_episodes + 1,
                           args.step_episodes))
    max_ep = max(ep_values)

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f'convergence_gnn_{timestamp}'
    save_dir = Path(args.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('Episode Convergence Analysis — GNN-HAPPO')
    print('=' * 70)
    print(f'  Episode values : {ep_values}')
    print(f'  Max episodes   : {max_ep}  (run once, sub-sample)')
    print(f'  Episode length : {args.episode_length} days')
    print(f'  Seed           : {args.seed}')
    print(f'  Output dir     : {save_dir}')
    print('=' * 70 + '\n')

    # ------------------------------------------------------------------
    # Step 1: Run the evaluator ONCE with the largest num_episodes value
    # ------------------------------------------------------------------
    # Create a copy of args with num_episodes = max_ep
    eval_args = argparse.Namespace(**vars(args))
    eval_args.num_episodes = max_ep
    eval_args.experiment_name = f'{exp_name}_full_run'
    eval_args.save_dir = str(save_dir / '_full_run')

    evaluator = GNNModelEvaluator(eval_args)
    evaluator.evaluate()

    # Extract per-episode total costs (holding + backlog + ordering)
    all_costs = []
    for m in evaluator.episode_metrics:
        total_holding = float(np.sum(m['holding_costs']))
        total_backlog = float(np.sum(m['backlog_costs']))
        total_ordering = float(np.sum(m['ordering_costs']))
        true_total_cost = total_holding + total_backlog + total_ordering
        all_costs.append(true_total_cost)

    all_costs = np.array(all_costs)
    print(f'\n[OK] Collected {len(all_costs)} episode costs.\n')

    # ------------------------------------------------------------------
    # Step 2: Compute statistics for each sub-sample size
    # ------------------------------------------------------------------
    summary_rows = []
    for n_ep in ep_values:
        subset = all_costs[:n_ep]
        mean_c = float(np.mean(subset))
        std_c = float(np.std(subset))
        min_c = float(np.min(subset))
        max_c = float(np.max(subset))
        cv = std_c / abs(mean_c) if abs(mean_c) > 1e-9 else 0.0
        ci_95 = 1.65 * std_c / np.sqrt(n_ep)  # 95% CI half-width

        summary_rows.append({
            'num_episodes': n_ep,
            'mean_cost': round(mean_c, 2),
            'std_cost': round(std_c, 2),
            'min_cost': round(min_c, 2),
            'max_cost': round(max_c, 2),
            'cv_pct': round(cv * 100, 4),
            'ci_95_half_width': round(ci_95, 2),
            'ci_95_pct_of_mean': round(ci_95 / abs(mean_c) * 100, 4) if abs(mean_c) > 1e-9 else 0.0,
        })

    df_summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Step 3: Rolling std (cumulative as episodes grow 1 → max_ep)
    # ------------------------------------------------------------------
    rolling_rows = []
    for i in range(2, len(all_costs) + 1):  # need ≥2 for std
        subset = all_costs[:i]
        rolling_rows.append({
            'episodes_so_far': i,
            'rolling_mean': round(float(np.mean(subset)), 2),
            'rolling_std': round(float(np.std(subset)), 2),
            'rolling_cv_pct': round(
                float(np.std(subset)) / abs(float(np.mean(subset))) * 100, 4
            ) if abs(float(np.mean(subset))) > 1e-9 else 0.0,
        })
    df_rolling = pd.DataFrame(rolling_rows)

    # ------------------------------------------------------------------
    # Step 4: Raw episode costs
    # ------------------------------------------------------------------
    df_episode = pd.DataFrame({
        'episode': np.arange(1, len(all_costs) + 1),
        'total_cost': np.round(all_costs, 2),
    })

    # ------------------------------------------------------------------
    # Step 5: Save to Excel
    # ------------------------------------------------------------------
    excel_path = save_dir / 'episode_convergence_analysis.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_episode.to_excel(writer, sheet_name='Episode_Costs', index=False)
        df_rolling.to_excel(writer, sheet_name='Rolling_Std', index=False)
    print(f'[OK] Saved convergence Excel: {excel_path}\n')

    # ------------------------------------------------------------------
    # Step 6: Convergence plot
    # ------------------------------------------------------------------
    _plot_convergence(df_summary, df_rolling, save_dir)

    # ------------------------------------------------------------------
    # Step 7: Print summary table
    # ------------------------------------------------------------------
    _print_summary_table(df_summary)

    print(f'\n[DONE] All results saved to: {save_dir}')


def _plot_convergence(df_summary, df_rolling, save_dir):
    """Create a multi-panel convergence plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Episode Convergence Analysis — GNN-HAPPO',
                 fontsize=16, fontweight='bold')

    # --- Panel 1: Std vs num_episodes ---
    ax = axes[0, 0]
    ax.plot(df_summary['num_episodes'], df_summary['std_cost'],
            'o-', color='#E74C3C', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Std of Total Cost', fontsize=12)
    ax.set_title('Cost Std vs. Num Episodes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Annotate each point
    for _, row in df_summary.iterrows():
        ax.annotate(f'{row["std_cost"]:.0f}',
                    (row['num_episodes'], row['std_cost']),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8)

    # --- Panel 2: CV% vs num_episodes ---
    ax = axes[0, 1]
    ax.plot(df_summary['num_episodes'], df_summary['cv_pct'],
            's-', color='#3498DB', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax.set_title('CV% vs. Num Episodes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Rolling std (cumulative) ---
    ax = axes[1, 0]
    ax.plot(df_rolling['episodes_so_far'], df_rolling['rolling_std'],
            '-', color='#2ECC71', linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Episodes Accumulated', fontsize=12)
    ax.set_ylabel('Rolling Std of Total Cost', fontsize=12)
    ax.set_title('Rolling Std (Cumulative)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Mark the sweep points
    for _, row in df_summary.iterrows():
        n = row['num_episodes']
        match = df_rolling[df_rolling['episodes_so_far'] == n]
        if not match.empty:
            ax.plot(n, match['rolling_std'].values[0], 'ro', markersize=7)

    # --- Panel 4: 95% CI half-width as % of mean ---
    ax = axes[1, 1]
    ax.plot(df_summary['num_episodes'], df_summary['ci_95_pct_of_mean'],
            'D-', color='#9B59B6', linewidth=2, markersize=8)
    ax.axhline(y=5, color='red', linestyle='--', linewidth=1.2,
               label='5% threshold')
    ax.axhline(y=2, color='orange', linestyle='--', linewidth=1.2,
               label='2% threshold')
    ax.set_xlabel('Number of Episodes', fontsize=12)
    ax.set_ylabel('95% CI / Mean (%)', fontsize=12)
    ax.set_title('Precision: 95% CI as % of Mean', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / 'convergence_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved convergence plot: {plot_path}')


def _print_summary_table(df_summary):
    """Pretty-print the convergence summary."""
    print('\n' + '=' * 90)
    print('Episode Convergence Summary')
    print('=' * 90)
    print(f'{"Num_Eps":>8}  {"Mean Cost":>12}  {"Std Cost":>10}  '
          f'{"CV%":>8}  {"95%CI±":>10}  {"CI/Mean%":>10}  '
          f'{"Min Cost":>10}  {"Max Cost":>10}')
    print('-' * 90)
    for _, r in df_summary.iterrows():
        print(f'{r["num_episodes"]:>8}  {r["mean_cost"]:>12.2f}  '
              f'{r["std_cost"]:>10.2f}  {r["cv_pct"]:>8.2f}  '
              f'{r["ci_95_half_width"]:>10.2f}  '
              f'{r["ci_95_pct_of_mean"]:>10.4f}  '
              f'{r["min_cost"]:>10.2f}  {r["max_cost"]:>10.2f}')
    print('=' * 90)

    # Recommend the smallest N where CI/Mean < 5%
    converged = df_summary[df_summary['ci_95_pct_of_mean'] < 5.0]
    if not converged.empty:
        rec = int(converged.iloc[0]['num_episodes'])
        print(f'\n✓ RECOMMENDATION: {rec} episodes is sufficient '
              f'(95% CI < 5% of mean cost).')
    else:
        print('\n⚠ 95% CI is still > 5% of mean even at '
              f'{int(df_summary.iloc[-1]["num_episodes"])} episodes. '
              f'Consider increasing max_episodes.')

    # Tighter threshold
    tight = df_summary[df_summary['ci_95_pct_of_mean'] < 2.0]
    if not tight.empty:
        rec2 = int(tight.iloc[0]['num_episodes'])
        print(f'  For tighter precision (CI < 2% of mean): '
              f'{rec2} episodes.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_convergence_analysis(args)


if __name__ == '__main__':
    main()
