#!/usr/bin/env python
"""
Test/Evaluation Script for Trained GNN-HAPPO Models — Fixed Demand Variant
===========================================================================

This script is identical to test_trained_model_gnn.py, with ONE key difference:

    Each episode is run with a **fixed, deterministic demand seed**.
    Episode i uses seed = (base_seed + i), so:
      - Demand sequences are reproducible across runs.
      - Every episode sees a DIFFERENT but fully deterministic demand sequence.
      - All policies tested with this script face EXACTLY the same demand
        sequences, making cross-policy comparisons fair.

This eliminates demand randomness as a confounding variable, so the observed
variation in (holding cost, service level) across episodes reflects policy
behaviour under controlled demand — not stochastic noise.

Usage:
    python test_trained_model_gnn_fixed_demand.py \\
        --model_dir results/07Apr_gnn_kaggle/run_seed_1/models \\
        --episode_length 365 \\
        --num_episodes 10 \\
        --seed 100 \\
        --experiment_name "eval_gnn_fixed_demand"
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
import json
import pandas as pd

from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from algorithms.gnn_happo_policy import GNN_HAPPO_Policy
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Trained GNN-HAPPO Model — Fixed Demand per Episode'
    )

    # Required
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory (e.g., results/.../models)')
    parser.add_argument('--config_path', type=str,
                        default='configs/multi_sku_config.yaml',
                        help='Path to environment config file (unused but kept for compatibility)')

    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Length of each episode in days')
    parser.add_argument('--seed', type=int, default=100,
                        help='Base random seed. Episode i uses seed+i (default: 100)')

    # Output
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this evaluation run (default: timestamp)')

    # Hardware
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA if available')

    # GNN architecture (must match training config)
    parser.add_argument('--gnn_type', type=str, default='GAT',
                        help='GNN type used during training (GAT, GCN, ...)')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN hidden dimension used during training')
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                        help='Number of GNN layers used during training')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='Number of attention heads (for GAT)')
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--use_residual', type=lambda x: x.lower() == 'true',
                        default=True)
    parser.add_argument('--critic_pooling', type=str, default='mean')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# GNN Evaluator — Fixed Demand
# ---------------------------------------------------------------------------

class GNNModelEvaluatorFixedDemand:
    """
    Evaluates trained GNN-HAPPO models with deterministic, per-episode demand.

    Key behaviour:
      - Before env.reset() for episode i, we set numpy + torch seeds to
        (base_seed + i).  This fixes the demand sequence drawn by the env's
        np.random calls (demand sampling, lead-time sampling, price noise).
      - The policy RNN state is freshly zeroed for each episode (no state
        leakage across episodes).
      - The model itself is deterministic (deterministic=True in act()).
        This isolates cost variation to demand alone, showing the true
        trade-off curve when you compare policies.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
        )

        # Output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = args.experiment_name if args.experiment_name else f'eval_gnn_fixed_{timestamp}'
        self.save_dir = Path(args.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._print_header()

        # 1. Create environment
        self.env = self._create_env()

        # Infer number of SKUs
        base_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        if base_env_list:
            base_env = base_env_list[0]
            self.n_skus = getattr(base_env, 'n_skus', 3)
        else:
            self.n_skus = 3

        # 2. Build graph adjacency
        self.adj_tensor = self._build_graph()

        # 3. Detect obs dim AND gnn_type from saved model
        self.single_agent_obs_dim = self._detect_model_config()

        # 4. Load GNN policies
        self.policies = self._load_models()

        # 5. Storage
        self.episode_metrics = []
        self.detailed_trajectory = None

    # ------------------------------------------------------------------
    # Setup helpers  (identical to base script)
    # ------------------------------------------------------------------

    def _print_header(self):
        print('=' * 70)
        print('GNN-HAPPO Model Evaluation — FIXED DEMAND PER EPISODE')
        print('=' * 70)
        print(f'Model directory : {self.args.model_dir}')
        print(f'Num episodes    : {self.args.num_episodes}')
        print(f'Episode length  : {self.args.episode_length} days')
        print(f'Base seed       : {self.args.seed}  (ep i uses seed+i)')
        print(f'Results dir     : {self.save_dir}')
        print(f'Device          : {self.device}')
        print('Demand          : FIXED per episode (reproducible across policies)')
        print('=' * 70 + '\n')

    def _create_env(self):
        print('Creating evaluation environment...')
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon',
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name='gnn_happo',
        )
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)

        self.n_agents = env.num_agent if hasattr(env, 'num_agent') else 17
        print(f'[OK] Environment created')
        print(f'     Agents         : {self.n_agents} (2 DCs + {self.n_agents - 2} Retailers)')
        obs_dims = [env.observation_space[i].shape[0] for i in range(self.n_agents)]
        print(f'     Obs dims       : DC={obs_dims[0]}D, Retailer={obs_dims[2]}D')
        print(f'     Action dim     : {env.action_space[0].shape[0]}D\n')
        return env

    def _build_graph(self):
        print('Building supply chain graph...')
        adj = build_supply_chain_adjacency(
            n_dcs=2, n_retailers=self.n_agents - 2, self_loops=True
        )
        adj = normalize_adjacency(adj, method='symmetric')
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        print(f'[OK] Graph: {adj.shape[0]} nodes, {int((adj > 0).sum())} edges\n')
        return adj_tensor

    def _detect_model_config(self):
        """Read obs dim AND gnn_type from agent-0 saved model state dict."""
        model_dir = Path(self.args.model_dir)
        agent0_files = sorted(model_dir.glob('actor_agent0*.pt'))
        if not agent0_files:
            raise FileNotFoundError(f'No actor_agent0 .pt file found in {model_dir}')

        sd = torch.load(str(agent0_files[0]), map_location='cpu')
        keys = list(sd.keys())

        if any('gnn_base.layers.0.weight' in k for k in keys):
            detected_gnn_type = 'GCN'
        elif any('gnn_base.layers.0.W' in k for k in keys):
            detected_gnn_type = 'GAT'
        else:
            detected_gnn_type = self.args.gnn_type

        gcn_key = 'gnn_base.layers.0.weight'
        gat_key = 'gnn_base.layers.0.W'
        if gcn_key in sd:
            obs_dim = sd[gcn_key].shape[0]
        elif gat_key in sd:
            obs_dim = sd[gat_key].shape[1]
        else:
            obs_dim = max(
                self.env.observation_space[i].shape[0] for i in range(self.n_agents)
            )

        print(f'[Auto] Detected GNN type     : {detected_gnn_type}')
        print(f'[Auto] Detected obs dim      : {obs_dim}D')
        if detected_gnn_type != self.args.gnn_type:
            print(f'       (overrides --gnn_type {self.args.gnn_type})')

        self.detected_gnn_type = detected_gnn_type
        return obs_dim

    def _build_all_args(self):
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
            scenario_name='inventory_2echelon',
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

    def _load_models(self):
        print('Loading GNN models...')
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f'Model directory not found: {model_dir}')

        all_args = self._build_all_args()

        from gymnasium import spaces as gym_spaces
        padded_obs_space = gym_spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.single_agent_obs_dim,), dtype=np.float32
        )

        policies = []
        for agent_id in range(self.n_agents):
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]

            best_file = self._find_best_model(model_dir, agent_id)

            policy = GNN_HAPPO_Policy(
                all_args,
                padded_obs_space,
                share_obs_space,
                act_space,
                n_agents=self.n_agents,
                agent_id=agent_id,
                device=self.device,
            )

            state_dict = torch.load(str(best_file), map_location=self.device)
            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()

            policies.append(policy)
            print(f'  [OK] Agent {agent_id:2d} loaded  <- {best_file.name}')

        print(f'\n[OK] All {self.n_agents} GNN agent models loaded successfully!\n')
        return policies

    def _find_best_model(self, model_dir: Path, agent_id: int) -> Path:
        all_files = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
        if not all_files:
            raise FileNotFoundError(
                f'No model file found for agent {agent_id} in {model_dir}'
            )

        reward_files = []
        for f in all_files:
            if f'actor_agent{agent_id}.pt' == f.name:
                continue
            try:
                reward_val = float(f.name.split('_reward_')[1].replace('.pt', ''))
                reward_files.append((reward_val, f))
            except (IndexError, ValueError):
                pass

        if reward_files:
            reward_files.sort(key=lambda x: x[0], reverse=True)
            best_reward, best_file = reward_files[0]
            print(f'  Agent {agent_id:2d}: best reward {best_reward:.2f}')
            return best_file

        plain = model_dir / f'actor_agent{agent_id}.pt'
        return plain if plain.exists() else all_files[0]

    # ------------------------------------------------------------------
    # Evaluation loop — KEY CHANGE: seed per episode
    # ------------------------------------------------------------------

    def evaluate(self):
        base_seed = getattr(self.args, 'seed', 100)

        print('=' * 70)
        print(f'Starting GNN Fixed-Demand Evaluation: {self.args.num_episodes} episode(s)')
        print(f'Base seed = {base_seed}  |  Episode i uses numpy/torch seed = {base_seed} + i')
        print('=' * 70 + '\n')

        for ep in range(self.args.num_episodes):
            # ── Per-episode deterministic seed ──────────────────────────────
            ep_seed = base_seed + ep
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(ep_seed)
            # ────────────────────────────────────────────────────────────────

            metrics = self._run_episode(ep, ep_seed, save_trajectory=(ep == 0))
            self.episode_metrics.append(metrics)

            avg_r = np.mean([m['total_reward'] for m in self.episode_metrics])
            print(f'Episode {ep + 1:>2}/{self.args.num_episodes} '
                  f'[seed={ep_seed}] '
                  f'| Total reward: {metrics["total_reward"]:>14.2f} '
                  f'| Running avg: {avg_r:>14.2f}')

        print('\n[OK] Evaluation complete!\n')

    def _run_episode(self, episode_num: int, ep_seed: int,
                     save_trajectory: bool = False) -> dict:
        """Run one evaluation episode with a fixed demand seed."""
        # Reset env — demand RNG is now fixed to ep_seed set before this call
        obs, _ = self.env.reset()
        max_obs_dim = self.single_agent_obs_dim

        # Fresh RNN states for each episode (no cross-episode state leakage)
        rnn_states = np.zeros(
            (1, self.n_agents, 2, 128), dtype=np.float32
        )
        masks = np.ones((1, self.n_agents, 1), dtype=np.float32)

        ep_data = {
            'episode_seed': ep_seed,
            'total_reward': 0.0,
            'total_cost': 0.0,
            'agent_rewards': [0.0] * self.n_agents,
            'agent_costs': [0.0] * self.n_agents,
            'holding_costs': [0.0] * self.n_agents,
            'backlog_costs': [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            'avg_inventory': [0.0] * self.n_agents,
            'avg_backlog': [0.0] * self.n_agents,
            'service_level': [0.0] * self.n_agents,
            '_orders_placed':     [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
            'dc_cycle_service_level': {},
            'final_inventory': None,
            'final_backlog': None,
        }

        if save_trajectory:
            traj = {
                'inventory': [[] for _ in range(self.n_agents)],
                'inventory_skus': [[] for _ in range(self.n_agents)],
                'backlog': [[] for _ in range(self.n_agents)],
                'actions': [[] for _ in range(self.n_agents)],
                'rewards': [[] for _ in range(self.n_agents)],
                'demand': [[] for _ in range(self.n_agents)],
                'norm_demand': [[] for _ in range(self.n_agents)],
                'norm_inventory': [[] for _ in range(self.n_agents)],
                'norm_order': [[] for _ in range(self.n_agents)],
                'orders_placed': [[] for _ in range(self.n_agents)],
                'orders_from_stock': [[] for _ in range(self.n_agents)],
                'norm_scales': None,
            }

        for step in range(self.args.episode_length):
            _pre_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = _pre_env_list[0].market_prices.copy() if _pre_env_list else None

            obs_structured = np.zeros(
                (1, self.n_agents, max_obs_dim), dtype=np.float32
            )
            for aid in range(self.n_agents):
                raw = np.stack(obs[:, aid])
                d = raw.shape[1]
                obs_structured[0, aid, :d] = raw[0]

            actions_env = []
            raw_actions = {}
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_structured,
                        self.adj_tensor,
                        agent_id,
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,   # deterministic policy for clean comparison
                    )

                rnn_states[:, agent_id] = (
                    rnn_state.cpu().numpy()
                    if isinstance(rnn_state, torch.Tensor)
                    else rnn_state
                )
                action_np = (
                    action.cpu().numpy()
                    if isinstance(action, torch.Tensor)
                    else action
                )
                raw_action = action_np[0]
                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list',
                               getattr(self.env, 'envs', None))
            if env_list:
                env_state = env_list[0]

                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.n_agents):
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward
                    ep_data['agent_rewards'][agent_id] += reward
                    ep_data['agent_costs'][agent_id] += cost
                    ep_data['total_reward'] += reward
                    ep_data['total_cost'] += cost

                    h_cost = b_cost = o_cost = 0.0
                    is_dc = agent_id < 2
                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(3):
                            h_cost += (env_state.inventory[agent_id][sku]
                                       * env_state.H_dc[dc_idx][sku])
                            dc_owed_sku = sum(
                                env_state.dc_retailer_backlog[agent_id][r_id][sku]
                                for r_id in env_state.dc_assignments[agent_id]
                            )
                            b_cost += dc_owed_sku * env_state.B_dc[dc_idx][sku]
                            if executed_actions[agent_id][sku] > 0:
                                price = pre_step_prices[sku] if pre_step_prices is not None else env_state.market_prices[sku]
                                o_cost += (env_state.C_fixed_dc[dc_idx][sku]
                                           + price * executed_actions[agent_id][sku])
                    else:
                        r_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(3):
                            h_cost += (env_state.inventory[agent_id][sku]
                                       * env_state.H_retailer[r_idx][sku])
                            b_cost += (env_state.backlog[agent_id][sku]
                                       * env_state.B_retailer[r_idx][sku])
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                o_cost += (env_state.C_fixed_retailer[r_idx][sku]
                                           + env_state.C_var_retailer[r_idx][assigned_dc][sku] * order_qty)

                        for sku in range(3):
                            placed = env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            from_stock = env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                            ep_data['_orders_placed'][agent_id]     += placed
                            ep_data['_orders_from_stock'][agent_id] += from_stock

                    ep_data['holding_costs'][agent_id] += h_cost
                    ep_data['backlog_costs'][agent_id] += b_cost
                    ep_data['ordering_costs'][agent_id] += o_cost

                    inv_vec = env_state.inventory[agent_id]
                    inv = inv_vec.sum()

                    if is_dc:
                        bl = sum(
                            sum(env_state.dc_retailer_backlog[agent_id][r_id].values())
                            for r_id in env_state.dc_assignments[agent_id]
                        )
                    else:
                        bl = env_state.backlog[agent_id].sum()

                    ep_data['avg_inventory'][agent_id] += inv
                    ep_data['avg_backlog'][agent_id] += bl

                    if save_trajectory:
                        traj['inventory'][agent_id].append(float(inv))
                        traj['inventory_skus'][agent_id].append(
                            np.array(inv_vec, dtype=float).copy()
                        )
                        traj['backlog'][agent_id].append(float(bl))
                        traj['rewards'][agent_id].append(reward)
                        traj['actions'][agent_id].append(executed_actions[agent_id].copy())
                        if agent_id < 2:
                            demand_vec = np.zeros(self.n_skus, dtype=float)
                            for r_id in env_state.dc_assignments[agent_id]:
                                op = env_state.step_orders_placed.get(r_id, {})
                                for s in range(self.n_skus):
                                    demand_vec[s] += op.get(s, 0.0)
                        else:
                            demand_vec = np.array(
                                env_state.step_demand.get(agent_id, np.zeros(self.n_skus, dtype=float)),
                                dtype=float,
                            )
                        traj['demand'][agent_id].append(demand_vec.copy())

                        if agent_id < 2:
                            traj['norm_demand'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                            traj['norm_inventory'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                            traj['norm_order'][agent_id].append(np.zeros(self.n_skus, dtype=float))
                        else:
                            dm = getattr(env_state, 'demand_mean', np.ones(self.n_skus) * 1.5)
                            ds = getattr(env_state, 'demand_std', np.ones(self.n_skus) * 1.0)
                            demand_cap = np.maximum(dm + 3.0 * ds, 1e-6)
                            norm_d = (demand_vec / demand_cap).astype(float)
                            norm_inv = (np.array(inv_vec, dtype=float) / 150.0)
                            act = executed_actions[agent_id]
                            norm_ord = np.clip((np.array(act, dtype=float) - 20.0) / 50.0, 0.0, 1.0)
                            traj['norm_demand'][agent_id].append(norm_d)
                            traj['norm_inventory'][agent_id].append(norm_inv)
                            traj['norm_order'][agent_id].append(norm_ord)

                        op = [env_state.step_orders_placed.get(agent_id, {}).get(s, 0) for s in range(self.n_skus)]
                        ofs = [env_state.step_orders_from_stock.get(agent_id, {}).get(s, 0) for s in range(self.n_skus)]
                        traj['orders_placed'][agent_id].append(np.array(op, dtype=float))
                        traj['orders_from_stock'][agent_id].append(np.array(ofs, dtype=float))

                        if traj['norm_scales'] is None and hasattr(env_state, 'demand_mean'):
                            dm = np.array(env_state.demand_mean, dtype=float).flatten()
                            ds = np.array(env_state.demand_std, dtype=float).flatten()
                            traj['norm_scales'] = {
                                'demand_mean_0': float(dm[0]) if len(dm) > 0 else 0,
                                'demand_mean_1': float(dm[1]) if len(dm) > 1 else 0,
                                'demand_mean_2': float(dm[2]) if len(dm) > 2 else 0,
                                'demand_std_0': float(ds[0]) if len(ds) > 0 else 0,
                                'demand_std_1': float(ds[1]) if len(ds) > 1 else 0,
                                'demand_std_2': float(ds[2]) if len(ds) > 2 else 0,
                                'demand_cap_0': float(dm[0] + 3 * ds[0]) if len(dm) > 0 else 0,
                                'demand_cap_1': float(dm[1] + 3 * ds[1]) if len(dm) > 1 else 0,
                                'demand_cap_2': float(dm[2] + 3 * ds[2]) if len(dm) > 2 else 0,
                                'inv_scale_retailer': 150.0,
                                'backlog_scale_retailer': 100.0,
                                'order_min_retailer': 0,
                                'order_max_retailer': 10.0,
                            }

                if step == self.args.episode_length - 1:
                    ep_data['final_inventory'] = [
                        float(env_state.inventory[i].sum())
                        for i in range(self.n_agents)
                    ]
                    ep_data['final_backlog'] = [
                        float(env_state.backlog[i].sum())
                        for i in range(self.n_agents)
                    ]

            info_dc_sl = infos[0][0].get('dc_cycle_service_level', {}) if infos and infos[0] else {}
            ep_data['dc_cycle_service_level'] = {
                int(dc_id): float(sl_val)
                for dc_id, sl_val in info_dc_sl.items()
            }

        # Normalise
        T = self.args.episode_length
        for agent_id in range(self.n_agents):
            ep_data['avg_inventory'][agent_id] /= T
            ep_data['avg_backlog'][agent_id] /= T
            if agent_id >= 2:
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                dc_sl_map = ep_data.get('dc_cycle_service_level', {})
                ep_data['service_level'][agent_id] = dc_sl_map.get(agent_id, 100.0)

        if save_trajectory:
            self.detailed_trajectory = traj

        return ep_data

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self):
        print('Generating evaluation report...')
        stats = self._calculate_statistics()
        self._save_metrics_json(stats)
        self._save_metrics_csv()
        self._create_visualizations(stats)
        self._print_summary(stats)
        print(f'\n[OK] Report saved to: {self.save_dir}\n')

    def _calculate_statistics(self) -> dict:
        def arr(key):
            return [m[key] for m in self.episode_metrics]

        stats = {
            'num_episodes': len(self.episode_metrics),
            'episode_length': self.args.episode_length,
            'base_seed': self.args.seed,
            'demand_mode': 'fixed_per_episode',
            'total_reward': {
                'mean': float(np.mean(arr('total_reward'))),
                'std': float(np.std(arr('total_reward'))),
                'min': float(np.min(arr('total_reward'))),
                'max': float(np.max(arr('total_reward'))),
            },
            'total_cost': {
                'mean': float(np.mean(arr('total_cost'))),
                'std': float(np.std(arr('total_cost'))),
                'min': float(np.min(arr('total_cost'))),
                'max': float(np.max(arr('total_cost'))),
            },
            'per_agent': {},
            'dc_cycle_service_level': {
                dc_id: float(np.mean([
                    m['dc_cycle_service_level'].get(dc_id, 100.0)
                    for m in self.episode_metrics
                ]))
                for dc_id in range(2)
            },
        }

        for aid in range(self.n_agents):
            label = f'{"DC" if aid < 2 else "Retailer"}_{aid}'

            if aid >= 2:
                total_placed     = sum(m['_orders_placed'][aid]     for m in self.episode_metrics)
                total_from_stock = sum(m['_orders_from_stock'][aid] for m in self.episode_metrics)
                pooled_sl = (total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0
            else:
                pooled_sl = float(np.mean([m['service_level'][aid] for m in self.episode_metrics]))

            stats['per_agent'][label] = {
                'avg_reward': float(np.mean([m['agent_rewards'][aid] for m in self.episode_metrics])),
                'avg_cost': float(np.mean([m['agent_costs'][aid] for m in self.episode_metrics])),
                'avg_holding_cost': float(np.mean([m['holding_costs'][aid] for m in self.episode_metrics])),
                'avg_backlog_cost': float(np.mean([m['backlog_costs'][aid] for m in self.episode_metrics])),
                'avg_ordering_cost': float(np.mean([m['ordering_costs'][aid] for m in self.episode_metrics])),
                'avg_inventory': float(np.mean([m['avg_inventory'][aid] for m in self.episode_metrics])),
                'avg_backlog': float(np.mean([m['avg_backlog'][aid] for m in self.episode_metrics])),
                'service_level': pooled_sl,
            }

        total_placed_all     = sum(m['_orders_placed'][aid]
                                   for m in self.episode_metrics
                                   for aid in range(2, self.n_agents))
        total_from_stock_all = sum(m['_orders_from_stock'][aid]
                                   for m in self.episode_metrics
                                   for aid in range(2, self.n_agents))
        stats['system_retailer_fill_rate'] = (
            (total_from_stock_all / total_placed_all * 100.0)
            if total_placed_all > 0 else 100.0
        )

        return stats

    def _save_metrics_json(self, stats):
        path = self.save_dir / 'evaluation_metrics.json'
        output = {
            'metadata': {
                'model_dir': str(self.args.model_dir),
                'algorithm': 'GNN-HAPPO',
                'num_episodes': self.args.num_episodes,
                'episode_length': self.args.episode_length,
                'base_seed': self.args.seed,
                'demand_mode': 'fixed_per_episode (seed+i per episode)',
                'gnn_type': self.args.gnn_type,
                'gnn_hidden_dim': self.args.gnn_hidden_dim,
                'gnn_num_layers': self.args.gnn_num_layers,
                'single_agent_obs_dim': self.single_agent_obs_dim,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'statistics': stats,
            'episode_data': self.episode_metrics,
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f'[OK] Saved metrics JSON: {path.name}')

    def _save_metrics_csv(self):
        import csv
        results_path = self.save_dir / 'results_gnn_happo_fixed_demand.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'
        for path in (results_path, compat_path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Episode_Index', 'Episode_Seed',
                    'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                    'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'
                ])
                for ep_num, m in enumerate(self.episode_metrics):
                    fill_rate = float(np.mean(m['service_level']))
                    total_placed     = sum(m['_orders_placed'][aid]     for aid in range(2, self.n_agents))
                    total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(2, self.n_agents))
                    lost_sales = total_placed - total_from_stock
                    avg_inventory = float(np.mean(m['avg_inventory']))
                    total_holding = float(np.sum(m['holding_costs']))
                    total_backlog = float(np.sum(m['backlog_costs']))
                    total_ordering = float(np.sum(m['ordering_costs']))
                    true_total_cost = total_holding + total_backlog + total_ordering
                    writer.writerow([
                        ep_num + 1,
                        m.get('episode_seed', self.args.seed + ep_num),
                        round(true_total_cost, 4),
                        round(fill_rate, 4),
                        round(lost_sales, 4),
                        round(avg_inventory, 4),
                        round(total_holding, 4),
                        round(total_backlog, 4),
                        round(total_ordering, 4),
                    ])
        print(f'[OK] Saved metrics CSV : {results_path.name}  (also {compat_path.name})')

    def _create_visualizations(self, stats):
        print('Creating visualizations...')
        self._plot_episode_costs()
        self._plot_holding_vs_service_level()
        self._plot_cost_breakdown(stats)
        self._plot_service_levels(stats)
        self._plot_performance_distribution()
        print('[OK] All visualizations created')

    def _plot_episode_costs(self):
        """Plot holding cost, backlog cost, and fill rate per episode side-by-side."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        episodes = range(1, len(self.episode_metrics) + 1)

        holding = [float(np.sum(m['holding_costs'])) for m in self.episode_metrics]
        backlog  = [float(np.sum(m['backlog_costs'])) for m in self.episode_metrics]
        fill_rate = [float(np.mean(m['service_level'])) for m in self.episode_metrics]

        axes[0].bar(episodes, holding, color='#4ECDC4', alpha=0.85, edgecolor='black')
        axes[0].set_ylabel('Total Holding Cost', fontsize=11)
        axes[0].set_title('Holding Cost per Episode (Fixed Demand Seeds)', fontsize=12, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.3)

        axes[1].bar(episodes, backlog, color='#FF6B6B', alpha=0.85, edgecolor='black')
        axes[1].set_ylabel('Total Backlog Cost', fontsize=11)
        axes[1].set_title('Backlog Cost per Episode', fontsize=12, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.3)

        axes[2].plot(episodes, fill_rate, marker='o', color='#2E86AB', linewidth=2)
        axes[2].axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Target 95%')
        axes[2].set_ylabel('Fill Rate (%)', fontsize=11)
        axes[2].set_title('Fill Rate per Episode', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Episode', fontsize=11)
        axes[2].set_ylim([0, 110])
        axes[2].legend()
        axes[2].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'episode_costs_fixed_demand.png', dpi=300)
        plt.close()
        print('[OK] Saved episode_costs_fixed_demand.png')

    def _plot_holding_vs_service_level(self):
        """
        Scatter plot: Total Holding Cost vs Fill Rate per episode.
        This is the key trade-off plot — with fixed demand, any correlation
        here reflects the policy's ordering behaviour, not demand noise.
        """
        holding   = [float(np.sum(m['holding_costs'])) for m in self.episode_metrics]
        fill_rate = [float(np.mean(m['service_level'])) for m in self.episode_metrics]
        episodes  = list(range(1, len(self.episode_metrics) + 1))

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(holding, fill_rate, c=episodes, cmap='viridis',
                             s=100, edgecolors='black', zorder=5)
        for i, ep in enumerate(episodes):
            ax.annotate(f'Ep{ep}', (holding[i], fill_rate[i]),
                        textcoords='offset points', xytext=(6, 4), fontsize=9)

        # Correlation
        corr = np.corrcoef(holding, fill_rate)[0, 1]
        ax.set_xlabel('Total Holding Cost', fontsize=12)
        ax.set_ylabel('Fill Rate (%)', fontsize=12)
        ax.set_title(
            f'Holding Cost vs. Service Level (Fixed Demand)\n'
            f'Pearson r = {corr:.3f}  '
            f'{"✓ trade-off visible" if corr > 0.3 else "— policy stable across episodes"}',
            fontsize=12, fontweight='bold'
        )
        ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Target 95%', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Episode number')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / 'holding_vs_service_level.png', dpi=300)
        plt.close()
        print('[OK] Saved holding_vs_service_level.png')

    def _plot_cost_breakdown(self, stats):
        fig, ax = plt.subplots(figsize=(14, 7))
        agents = list(stats['per_agent'].keys())
        hc = [stats['per_agent'][a]['avg_holding_cost'] for a in agents]
        bc = [stats['per_agent'][a]['avg_backlog_cost'] for a in agents]
        oc = [stats['per_agent'][a]['avg_ordering_cost'] for a in agents]
        x = np.arange(len(agents))
        w = 0.6
        ax.bar(x, hc, w, label='Holding', color='#4ECDC4', edgecolor='black')
        ax.bar(x, bc, w, bottom=hc, label='Backlog', color='#FF6B6B', edgecolor='black')
        ax.bar(x, oc, w, bottom=np.array(hc) + np.array(bc),
               label='Ordering', color='#FFD700', edgecolor='black')
        for i in range(len(agents)):
            total = hc[i] + bc[i] + oc[i]
            ax.text(i, total, f'{total:.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_ylabel('Avg Cost / Episode', fontsize=12)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_title('GNN-HAPPO Cost Breakdown by Agent (Fixed Demand)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'cost_breakdown.png', dpi=300)
        plt.close()

    def _plot_service_levels(self, stats):
        fig, ax = plt.subplots(figsize=(12, 5))
        agents = list(stats['per_agent'].keys())
        svc = [stats['per_agent'][a]['service_level'] for a in agents]
        colors = ['#2E86AB'] * 2 + ['#A23B72'] * (len(agents) - 2)
        bars = ax.bar(agents, svc, color=colors, alpha=0.85, edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target 95%')
        ax.set_ylabel('Service Level (%)', fontsize=12)
        ax.set_title('Service Level by Agent — Fixed Demand per Episode',
                     fontsize=14, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()

    def _plot_performance_distribution(self):
        if len(self.episode_metrics) < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        rewards = [m['total_reward'] for m in self.episode_metrics]
        ax.hist(rewards, bins=min(20, len(rewards)), edgecolor='black',
                color='#3498db', alpha=0.8)
        ax.axvline(np.mean(rewards), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(rewards):.0f}')
        ax.set_xlabel('Total Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('GNN-HAPPO Reward Distribution (Fixed Demand)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_distribution.png', dpi=300)
        plt.close()

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('GNN-HAPPO Fixed-Demand Evaluation Summary')
        print('=' * 70)
        print(f"Episodes      : {stats['num_episodes']}  (base seed={stats['base_seed']}, ep_i uses seed+i)")
        print(f"Episode length: {stats['episode_length']} days")
        print(f"Demand        : FIXED per episode (reproducible)")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    evaluator = GNNModelEvaluatorFixedDemand(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
