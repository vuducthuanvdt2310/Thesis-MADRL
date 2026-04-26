#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lead Time Robustness Comparison: 4 Models x 5 Lead Time Variations
================================================================

Evaluates four inventory policies under different lead time scenarios and 
produces a line plot comparing total cost performance.

  Models:
    - (s,S) BaseStock Heuristic
    - HAPPO
    - MAPPO
    - GNN-HAPPO

  Lead Time Variations (Supplier -> DC):
    - Variation -1: Uniform[6, 13]
    - Variation  0: Uniform[7, 14] (Baseline)
    - Variation  1: Uniform[8, 15]
    - Variation  2: Uniform[9, 16]
    - Variation  3: Uniform[10, 17]
"""

import sys
import os
import argparse
import numpy as np
import torch
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import evaluators from robustness_comparison.py
# We import them locally within functions to avoid issues with early initialization
# but we can import the definitions here.
try:
    from robustness_comparison import (
        BaseStockEvaluator, 
        HAPPOEvaluator, 
        GNNModelEvaluator, 
        MAPPORobustnessEvaluator,
        parse_args as base_parse_args,
        _apply_scenario
    )
except ImportError:
    print("[ERROR] Could not import from robustness_comparison.py. Please ensure it is in the same directory.")
    sys.exit(1)

def _apply_lead_time_variation(evaluator, variation):
    """
    Inject lead time variation into the environment(s) of an evaluator.
    """
    # Find the environment object(s)
    env_obj = getattr(evaluator, 'env', None)
    if env_obj is None:
        return
        
    # Check if it's a VecEnv or list of envs
    env_list = []
    if hasattr(env_obj, 'env_list'):
        env_list = env_obj.env_list
    elif hasattr(env_obj, 'envs'):
        env_list = env_obj.envs
    else:
        env_list = [env_obj]
        
    # Apply variation to each env
    for env in env_list:
        # Access the underlying MultiDCInventoryEnv if wrapped
        target_env = env
        while hasattr(target_env, 'env'):
            target_env = target_env.env
            
        # Baseline is 7 to 14
        target_env.lt_supplier_to_dc_min = 7 + variation
        target_env.lt_supplier_to_dc_max = 14 + variation
        # print(f"  [DEBUG] Applied LT variation {variation}: [{target_env.lt_supplier_to_dc_min}, {target_env.lt_supplier_to_dc_max}]")

def parse_args():
    parser = argparse.ArgumentParser(description='Lead time robustness comparison')
    
    # Re-use defaults from robustness_comparison.py
    # or specify them here if we want a clean CLI.
    
    # -- Paths --
    parser.add_argument('--gnn_model_dir', type=str,
                        default='results/14Apr_gnn_kaggle_vari/run_seed_1/models')
    parser.add_argument('--happo_model_dir', type=str,
                        default='results/01Apr_base/run_seed_1/models')
    parser.add_argument('--mappo_model_dir', type=str,
                        default='results/25Apr_MAPPO/run_seed_1/models')

    # -- Episode settings --
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of evaluation episodes per instance')
    parser.add_argument('--episode_length', type=int, default=90)
    parser.add_argument('--happo_episode_length', type=int, default=115)
    parser.add_argument('--mappo_episode_length', type=int, default=130)
    parser.add_argument('--basestock_episode_length', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)

    # -- Environment --
    parser.add_argument('--config_path', type=str, default='configs/multi_dc_config.yaml')
    parser.add_argument('--num_agents', type=int, default=17)
    parser.add_argument('--cuda', action='store_true', default=False)

    # -- (s,S) policy params --
    parser.add_argument('--s_dc',         type=float, default=100.0)
    parser.add_argument('--S_dc',         type=float, default=170.0)
    parser.add_argument('--s_retailer',   type=float, default=3.0)
    parser.add_argument('--S_retailer',   type=float, default=10.0)

    # -- Output --
    parser.add_argument('--save_dir', type=str, default='evaluation_results/lead_time_comparison')

    return parser.parse_args()

def run_test(args, variation, model_type):
    """Run a single test for a model type and lead time variation."""
    import copy
    test_args = copy.deepcopy(args)
    
    if model_type == 'Base Stock':
        test_args.episode_length = args.basestock_episode_length
        evaluator = BaseStockEvaluator(test_args, config_path=args.config_path)
        _apply_lead_time_variation(evaluator, variation)
        evaluator.evaluate()
        
    elif model_type == 'HAPPO':
        test_args.episode_length = args.happo_episode_length
        test_args.model_dir = args.happo_model_dir
        test_args.algorithm_name = 'happo'
        evaluator = HAPPOEvaluator(test_args)
        _apply_lead_time_variation(evaluator, variation)
        evaluator.evaluate()
        
    elif model_type == 'MAPPO':
        test_args.episode_length = args.mappo_episode_length
        test_args.model_dir      = args.mappo_model_dir
        test_args.save_dir       = args.save_dir + '/mappo_tmp'
        test_args.experiment_name = f'lt_var_{variation}'
        evaluator = MAPPORobustnessEvaluator(test_args)
        _apply_lead_time_variation(evaluator, variation)
        evaluator.evaluate()
        
    elif model_type == 'GNN-HAPPO':
        test_args.model_dir = args.gnn_model_dir
        test_args.algorithm_name = 'gnn_happo'
        # Need to ensure GNN args are passed if needed
        test_args.gnn_type = 'GAT'
        test_args.gnn_hidden_dim = 128
        test_args.gnn_num_layers = 2
        test_args.num_attention_heads = 4
        test_args.gnn_dropout = 0.1
        test_args.use_residual = True
        test_args.critic_pooling = 'mean'
        
        evaluator = GNNModelEvaluator(test_args)
        _apply_lead_time_variation(evaluator, variation)
        evaluator.evaluate()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    mean_cost, std_cost = evaluator.mean_total_cost()
    return mean_cost, std_cost

def plot_comparison(results, variations, save_dir, args):
    """
    Plot the cost comparison similar to the requested image.
    results: dict {model_name: [mean_costs]}
    """
    plt.figure(figsize=(8, 6))
    
    # Normalization factors (episode_length * n_agents)
    n_agents = args.num_agents
    norm_factors = {
        'Base Stock': args.basestock_episode_length * n_agents,
        'HAPPO':      args.happo_episode_length * n_agents,
        'MAPPO':      args.mappo_episode_length * n_agents,
        'GNN-HAPPO':  args.episode_length * n_agents
    }
    
    plot_configs = {
        'GNN-HAPPO': {'label': 'MAPPO', 'marker': '.', 'color': '#3A86FF', 'markersize': 8},
        'HAPPO':     {'label': 'HAPPO',    'marker': '^', 'color': '#D67D4B', 'markersize': 8},
        'MAPPO':     {'label': 'GNN-HAPPO', 'marker': 'd', 'color': '#3BB273', 'markersize': 8},
        'Base Stock':{'label': 'Base Stock', 'marker': '*', 'color': '#C04141', 'markersize': 8}
    }
    
    model_order = ['GNN-HAPPO', 'HAPPO', 'MAPPO', 'Base Stock']
    
    for model in model_order:
        if model in results:
            # Normalize: Average daily cost per agent
            vals = np.array(results[model]) / norm_factors.get(model, 1.0)
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
    
    # Match image aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = Path(save_dir) / 'lead_time_cost_comparison.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Comparison plot saved to: {out_path}")

def main():
    args = parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    variations = [-1, 0, 1, 2, 3]
    model_names = ['Base Stock', 'HAPPO', 'MAPPO', 'GNN-HAPPO']
    
    results = {name: [] for name in model_names}
    
    print(f"Starting Lead Time Robustness Comparison...")
    print(f"Variations: {variations}")
    print(f"Num Episodes: {args.num_episodes}")
    
    for var in variations:
        print(f"\nEvaluating Lead Time Variation: {var}")
        for model in model_names:
            print(f"  Running {model}...", end='', flush=True)
            try:
                mean_cost, std_cost = run_test(args, var, model)
                results[model].append(mean_cost)
                print(f" Done. Mean Cost: {mean_cost:,.2f}")
            except Exception as e:
                print(f" Failed: {e}")
                results[model].append(np.nan)
        
        # Save results to CSV after each variation (incremental save)
        csv_path = save_dir / 'lead_time_comparison_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Variation'] + model_names)
            # Find how many variations we have results for
            num_rows = len(results[model_names[0]])
            for i in range(num_rows):
                row = [variations[i]] + [results[m][i] for m in model_names]
                writer.writerow(row)
        print(f"  [Progress] Results saved to: {csv_path}")
                
    # Final plot
    plot_comparison(results, variations, save_dir, args)

if __name__ == '__main__':
    main()
