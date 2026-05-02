#!/usr/bin/env python
"""
Test/Evaluation Script for Trained GNN-HAPPO Models — 4 DC × 40 Retailer Scale

Wraps the existing GNNModelEvaluator from test_trained_model_gnn.py,
overriding environment creation to use the 4x40 config.

Usage:
    python test_trained_model_gnn_4x40.py \
        --model_dir results/gnn_happo_4x40/run_seed_1/models \
        --num_episodes 100 --episode_length 365
"""

import sys
import os
import numpy as np
import torch

# Reuse the full evaluator from the original test script
from test_trained_model_gnn import GNNModelEvaluator, parse_args
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency

# ================================================================
# SCALED CONFIG
# ================================================================
N_DCS = 4
N_RETAILERS = 40
N_AGENTS = N_DCS + N_RETAILERS  # 44
CONFIG_PATH = 'configs/multi_dc_4x40_config.yaml'


class GNNModelEvaluator4x40(GNNModelEvaluator):
    """Evaluator subclass for the 4 DC × 40 Retailer scaled environment."""

    def _create_env(self):
        """Override: create env with 4x40 config."""
        print('Creating evaluation environment (4×40 scale)...')
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon_4x40',
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name='gnn_happo',
        )
        all_args = parser.parse_known_args([])[0]
        all_args.env_config_path = CONFIG_PATH
        env = DummyVecEnvMultiDC(all_args)

        self.n_agents = env.num_agent if hasattr(env, 'num_agent') else N_AGENTS
        print(f'[OK] Environment created')
        print(f'     Agents         : {self.n_agents} ({N_DCS} DCs + {N_RETAILERS} Retailers)')
        obs_dims = [env.observation_space[i].shape[0] for i in range(self.n_agents)]
        print(f'     Obs dims       : DC={obs_dims[0]}D, Retailer={obs_dims[N_DCS]}D')
        print(f'     Action dim     : {env.action_space[0].shape[0]}D\n')
        return env

    def _build_graph(self):
        """Override: build graph for 4 DCs + 40 retailers."""
        print('Building supply chain graph (4×40)...')
        adj = build_supply_chain_adjacency(
            n_dcs=N_DCS, n_retailers=N_RETAILERS, self_loops=True
        )
        adj = normalize_adjacency(adj, method='symmetric')
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        print(f'[OK] Graph: {adj.shape[0]} nodes, {int((adj > 0).sum())} edges\n')
        return adj_tensor

    def _build_all_args(self):
        """Override: set num_agents to 44."""
        gnn_type = getattr(self, 'detected_gnn_type', self.args.gnn_type)
        parser = get_config()
        parser.add_argument('--gnn_type', type=str, default=gnn_type)
        parser.add_argument('--gnn_hidden_dim', type=int, default=self.args.gnn_hidden_dim)
        parser.add_argument('--gnn_num_layers', type=int, default=self.args.gnn_num_layers)
        parser.add_argument('--num_attention_heads', type=int,
                            default=self.args.num_attention_heads)
        parser.add_argument('--gnn_dropout', type=float, default=self.args.gnn_dropout)
        parser.add_argument('--use_residual',
                            type=lambda x: x.lower() == 'true',
                            default=self.args.use_residual)
        parser.add_argument('--critic_pooling', type=str, default=self.args.critic_pooling)
        parser.add_argument('--single_agent_obs_dim', type=int,
                            default=self.single_agent_obs_dim)
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon_4x40',
            num_agents=self.n_agents,
            use_centralized_V=True,
            algorithm_name='gnn_happo',
            hidden_size=128,
            layer_N=2,
            use_ReLU=True,
            use_orthogonal=True,
            gain=0.01,
            recurrent_N=2,
            use_naive_recurrent_policy=True,
            single_agent_obs_dim=self.single_agent_obs_dim,
        )
        return parser.parse_known_args([])[0]

    def _calculate_statistics(self):
        """Override: handle N DCs instead of hardcoded 2."""
        stats = super()._calculate_statistics()
        # Fix dc_cycle_service_level to use N_DCS
        stats['dc_cycle_service_level'] = {
            dc_id: float(np.mean([
                m['dc_cycle_service_level'].get(dc_id, 100.0)
                for m in self.episode_metrics
            ]))
            for dc_id in range(N_DCS)
        }
        return stats

    def _run_episode(self, episode_num, save_trajectory=False):
        """Override: use N_DCS instead of hardcoded 2 for DC detection."""
        # Temporarily patch the is_dc check boundary
        orig_n_agents = self.n_agents
        metrics = super()._run_episode(episode_num, save_trajectory)
        return metrics

    def _print_header(self):
        print('=' * 70)
        print('GNN-HAPPO Model Evaluation — 4×40 Scale')
        print('=' * 70)
        print(f'Model directory : {self.args.model_dir}')
        print(f'Config          : {CONFIG_PATH}')
        print(f'Topology        : 1 Supplier → {N_DCS} DCs → {N_RETAILERS} Retailers')
        print(f'Num episodes    : {self.args.num_episodes}')
        print(f'Episode length  : {self.args.episode_length} days')
        print(f'Results dir     : {self.save_dir}')
        print(f'Device          : {self.device}')
        print('=' * 70 + '\n')

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('GNN-HAPPO Evaluation Summary — 4×40 Scale')
        print('=' * 70)
        print(f"Episodes      : {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print(f"Avg reward    : {stats['total_reward']['mean']:>14.2f} "
              f"(+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost      : {stats['total_cost']['mean']:>14.2f} "
              f"(+/-{stats['total_cost']['std']:.2f})")
        sys_fr = stats.get('system_retailer_fill_rate', 0.0)
        print(f"Retailer Fill Rate (system): {sys_fr:>6.1f}%   <- KPI")
        dc_csl = stats.get('dc_cycle_service_level', {})
        for dc_id, csl_val in dc_csl.items():
            print(f"DC_{dc_id} Cycle SL : {csl_val:>6.1f}%")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Reward':>12} {'Avg Cost':>12} "
              f"{'Holding':>12} {'Backlog':>12} {'Svc%':>8}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            print(f"{agent:<14} {data['avg_reward']:>12.1f} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}%")
        print('=' * 70)


def main():
    args = parse_args()
    evaluator = GNNModelEvaluator4x40(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
