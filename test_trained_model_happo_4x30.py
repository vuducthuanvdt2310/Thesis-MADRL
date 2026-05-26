#!/usr/bin/env python
"""
Test/Evaluation Script for Trained HAPPO Models -- 4 DC x 30 Retailer Scale

Subclasses HAPPOEvaluator2x30 to reuse its topology-aware episode runner.

Usage:
    python test_trained_model_happo_4x30.py \
        --model_dir results/happo_4x30/run_seed_1/models \
        --num_episodes 100 --episode_length 365
"""

import numpy as np
import torch

from test_trained_model_happo_2x30 import HAPPOEvaluator2x30
from test_trained_model_happo_2x15 import parse_test_args
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC

N_DCS = 4
N_RETAILERS = 30
N_AGENTS = N_DCS + N_RETAILERS  # 34
CONFIG_PATH = 'configs/multi_dc_4x30_config.yaml'


class HAPPOEvaluator4x30(HAPPOEvaluator2x30):
    """HAPPO evaluator for the 4 DC x 30 Retailer environment."""

    def __init__(self, args):
        self.n_dcs = N_DCS
        self.policy_class_name = 'happo'
        from test_trained_model_happo_2x15 import ModelEvaluator
        ModelEvaluator.__init__(self, args)

    def _create_env(self):
        print('Creating evaluation environment (4x30 scale)...')
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon_4x30',
            num_agents=N_AGENTS,
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name='happo',
        )
        all_args = parser.parse_known_args([])[0]
        all_args.env_config_path = CONFIG_PATH
        env = DummyVecEnvMultiDC(all_args)

        self.args.num_agents = env.num_agent if hasattr(env, 'num_agent') else N_AGENTS
        self.n_dcs = N_DCS
        print(f'[OK] Environment: {self.args.num_agents} agents '
              f'({N_DCS} DCs + {N_RETAILERS} Retailers)')
        obs_dims = [env.observation_space[i].shape[0] for i in range(self.args.num_agents)]
        print(f'     Obs: DC={obs_dims[0]}D, Retailer={obs_dims[N_DCS]}D | '
              f'Action: {env.action_space[0].shape[0]}D\n')
        return env

    def _save_metrics_csv(self):
        import csv

        results_path = self.save_dir / 'results_standard_happo_4x30.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'

        for csv_path in (results_path, compat_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                                 'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])

                for ep_num, metrics in enumerate(self.episode_metrics):
                    fill_rate = float(np.mean(metrics['service_level']))
                    total_placed     = sum(metrics['_orders_placed'][aid]     for aid in range(self.n_dcs, self.args.num_agents))
                    total_from_stock = sum(metrics['_orders_from_stock'][aid] for aid in range(self.n_dcs, self.args.num_agents))
                    lost_sales = total_placed - total_from_stock
                    avg_inventory = float(np.mean(metrics['avg_inventory']))
                    total_holding = float(np.sum(metrics['holding_costs']))
                    total_backlog = float(np.sum(metrics['backlog_costs']))
                    total_ordering = float(np.sum(metrics['ordering_costs']))
                    true_total_cost = total_holding + total_backlog + total_ordering
                    writer.writerow([
                        ep_num + 1,
                        round(true_total_cost, 4),
                        round(fill_rate, 4),
                        round(lost_sales, 4),
                        round(avg_inventory, 4),
                        round(total_holding, 4),
                        round(total_backlog, 4),
                        round(total_ordering, 4),
                    ])

        print(f"  Saved metrics CSV: {results_path.name}  (also {compat_path.name})")

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('HAPPO Evaluation Summary -- 4x30 Scale')
        print('=' * 70)
        print(f"Episodes: {stats['num_episodes']} | Length: {stats['episode_length']} days")
        print(f"Avg reward: {stats['total_reward']['mean']:.2f} (+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost  : {stats['total_cost']['mean']:.2f} (+/-{stats['total_cost']['std']:.2f})")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Cost':>12} {'Holding':>12} {'Backlog':>12} {'Svc%':>8}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            print(f"{agent:<14} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}%")
        print('=' * 70)


def main():
    args = parse_test_args()
    evaluator = HAPPOEvaluator4x30(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
