#!/usr/bin/env python
"""
Test/Evaluation Script for Trained MAPPO Models

Usage:
    python test_trained_model_mappo.py \
        --model_dir results/mappo_training/run_seed_1/models \
        --episode_length 90 \
        --num_episodes 10 \
        --experiment_name "eval_mappo"
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Re-use the full evaluator from the baseline script — it already supports
# loading any HAPPO-compatible policy (same actor/critic architecture).
# MAPPO uses the exact same Actor/Critic networks; only the training update differs.
from test_trained_model import ModelEvaluator, NumpyEncoder
from algorithms.mappo_policy import MAPPO_Policy
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Trained MAPPO Model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory (e.g., results/.../models)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Length of each episode in days')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this evaluation run (default: timestamp)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA if available')
    parser.add_argument('--num_agents', type=int, default=17,
                        help='Number of agents (2 DCs + 15 Retailers)')

    return parser.parse_args()


class MAPPOEvaluator(ModelEvaluator):
    """
    Thin subclass of ModelEvaluator that loads MAPPO_Policy weights instead of
    HAPPO_Policy weights.  All evaluation logic (episode loop, cost breakdown,
    plots, CSV/JSON output) is inherited unchanged.
    """

    def _create_env(self):
        """Create evaluation environment with algorithm_name='mappo'."""
        print('Creating evaluation environment...')
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon',
            num_agents=self.args.num_agents,
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name='mappo',
        )
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)

        if hasattr(env, 'num_agent'):
            self.args.num_agents = env.num_agent

        print(f'[OK] Environment created')
        print(f'     Agents     : {self.args.num_agents} (2 DCs + {self.args.num_agents - 2} Retailers)')
        obs_dims = [env.observation_space[i].shape[0] for i in range(self.args.num_agents)]
        print(f'     Obs dims   : DC={obs_dims[0]}D, Retailer={obs_dims[2]}D')
        print(f'     Action dim : {env.action_space[0].shape[0]}D\n')
        return env

    def _load_models(self):
        """Load MAPPO actor weights for all agents."""
        print('Loading MAPPO models...')
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f'Model directory not found: {model_dir}')

        # Build args for policy construction
        cfg_parser = get_config()
        cfg_parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon',
            num_agents=self.args.num_agents,
            use_centralized_V=True,
            algorithm_name='mappo',
            hidden_size=128,
            layer_N=2,
            use_ReLU=True,
            use_orthogonal=True,
            gain=0.01,
            recurrent_N=2,
            use_naive_recurrent_policy=True,
        )
        all_args = cfg_parser.parse_known_args([])[0]

        policies = []
        for agent_id in range(self.args.num_agents):
            obs_space = self.env.observation_space[agent_id]
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]

            # Find best checkpoint
            agent_files = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
            if not agent_files:
                raise FileNotFoundError(f'No actor file for agent {agent_id} in {model_dir}')

            reward_files = []
            for f in agent_files:
                if f.name == f'actor_agent{agent_id}.pt':
                    continue
                try:
                    rv = float(f.name.split('_reward_')[1].replace('.pt', ''))
                    reward_files.append((rv, f))
                except (IndexError, ValueError):
                    pass

            if reward_files:
                reward_files.sort(key=lambda x: x[0], reverse=True)
                best_rv, best_file = reward_files[0]
                print(f'  Agent {agent_id:2d}: best reward {best_rv:.2f}  <- {best_file.name}')
            else:
                best_file = model_dir / f'actor_agent{agent_id}.pt'
                if not best_file.exists():
                    best_file = agent_files[0]
                print(f'  Agent {agent_id:2d}: {best_file.name}')

            # Detect actual obs dim from saved weights (handles dim mismatch)
            sd = torch.load(str(best_file), map_location=self.device)
            if 'base.mlp.fc1.0.weight' in sd:
                saved_dim = sd['base.mlp.fc1.0.weight'].shape[1]
                if saved_dim != obs_space.shape[0]:
                    from gymnasium import spaces as gym_spaces
                    obs_space = gym_spaces.Box(
                        low=-np.inf, high=np.inf, shape=(saved_dim,), dtype=np.float32
                    )

            policy = MAPPO_Policy(all_args, obs_space, share_obs_space, act_space,
                                  device=self.device)
            policy.actor.load_state_dict(sd)
            policy.actor.eval()
            policies.append(policy)

        print(f'\n[OK] All {self.args.num_agents} MAPPO agent models loaded!\n')
        return policies

    # Override just the header print and CSV/JSON labels
    def _print_header(self):
        print('=' * 70)
        print('MAPPO Model Evaluation')
        print('=' * 70)
        print(f'Model directory : {self.args.model_dir}')
        print(f'Num episodes    : {self.args.num_episodes}')
        print(f'Episode length  : {self.args.episode_length} days')
        print(f'Results dir     : {self.save_dir}')
        print(f'Device          : {self.device}')
        print('=' * 70 + '\n')

    def _save_metrics_csv(self):
        import csv
        results_path = self.save_dir / 'results_mappo.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'
        for path in (results_path, compat_path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales',
                                 'Avg_Inventory', 'Total_Holding_Cost',
                                 'Total_Backlog_Cost', 'Total_Ordering_Cost'])
                for ep_num, m in enumerate(self.episode_metrics):
                    fill_rate = float(np.mean(m['service_level']))
                    total_placed     = sum(m['_orders_placed'][a]     for a in range(2, self.args.num_agents))
                    total_from_stock = sum(m['_orders_from_stock'][a] for a in range(2, self.args.num_agents))
                    lost_sales = total_placed - total_from_stock
                    writer.writerow([
                        ep_num + 1,
                        round(float(np.sum(m['holding_costs'])) + float(np.sum(m['backlog_costs'])) + float(np.sum(m['ordering_costs'])), 4),
                        round(fill_rate, 4),
                        round(lost_sales, 4),
                        round(float(np.mean(m['avg_inventory'])), 4),
                        round(float(np.sum(m['holding_costs'])), 4),
                        round(float(np.sum(m['backlog_costs'])), 4),
                        round(float(np.sum(m['ordering_costs'])), 4),
                    ])
        print(f'[OK] Saved CSV: {results_path.name}')

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('MAPPO Evaluation Summary')
        print('=' * 70)
        print(f"Episodes      : {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print(f"Avg reward    : {stats['total_reward']['mean']:>14.2f} (+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost      : {stats['total_cost']['mean']:>14.2f} (+/-{stats['total_cost']['std']:.2f})")
        print(f"Best reward   : {stats['total_reward']['max']:>14.2f}")
        print(f"Worst reward  : {stats['total_reward']['min']:>14.2f}")
        total_placed_all     = sum(m['_orders_placed'][a]     for m in self.episode_metrics for a in range(2, self.args.num_agents))
        total_from_stock_all = sum(m['_orders_from_stock'][a] for m in self.episode_metrics for a in range(2, self.args.num_agents))
        sys_fr = (total_from_stock_all / total_placed_all * 100.0) if total_placed_all > 0 else 100.0
        print(f"Retailer Fill Rate (system): {sys_fr:>6.1f}%   <- KPI")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Reward':>12} {'Avg Cost':>12} {'Holding':>12} {'Backlog':>12} {'Svc%':>8}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            print(f"{agent:<14} {data['avg_reward']:>12.1f} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}%")
        print('=' * 70)


def main():
    args = parse_args()

    # Pass algorithm_name so parent class skips GNN graph build
    args.algorithm_name = 'mappo'

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name if args.experiment_name else f'eval_mappo_{timestamp}'

    evaluator = MAPPOEvaluator(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
