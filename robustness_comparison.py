#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness Comparison: 3 Models x 3 Demand Scenarios
=====================================================

Evaluates three inventory policies under three demand scenarios and produces
a grouped bar chart comparing total cost and fill rate.

  Models:
    - (s,S) BaseStock Heuristic
    - HAPPO  (results/01Apr_base/run_seed_1/models)
    - GNN-HAPPO (results/14Apr_gnn_kaggle_vari/run_seed_1/models)

  Scenarios (per-SKU mean & std override):
    S1 - Balanced      : low-stress, matches training distribution
    S2 - High Demand   : elevated means, mixed stress
    S3 - Extreme Stress: original means + high volatility

All other evaluation parameters are kept identical to the originals:
  episode_length = 150  (from Test_baseline_basestock.py)
  num_episodes   = 100
  seed           = 42
"""

# ---------------------------------------------------------------------------
# Standard library / third-party imports
# ---------------------------------------------------------------------------
import sys
import os
import argparse
import numpy as np
import torch
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from envs.multi_dc_env import MultiDCInventoryEnv
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from algorithms.happo_policy import HAPPO_Policy
from algorithms.gnn_happo_policy import GNN_HAPPO_Policy
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency


# ---------------------------------------------------------------------------
# Demand scenario definitions
# Each entry overrides demand_mean and demand_std in the environment.
# SKU order must match environment SKU indexing (0, 1, 2).
# ---------------------------------------------------------------------------

DEMAND_SCENARIOS = {
    "Scenario_1_Balanced": {
        "SKU_0": {"mean": 1.41, "std": 1.99},
        "SKU_1": {"mean": 1.06, "std": 1.28},
        "SKU_2": {"mean": 0.77, "std": 1.06},
        "label": "S1: Balanced\n(Low Stress)",
        "short": "S1-Balanced",
    },
    "Scenario_2_HighDemand": {
        "SKU_0": {"mean": 1.81, "std": 1.99},
        "SKU_1": {"mean": 1.378, "std": 1.28},
        "SKU_2": {"mean": 1.001, "std": 1.06},
        "label": "S2: High Demand\n(Mixed Stress)",
        "short": "S2-High",
    },
    "Scenario_3_Extreme_Stress": {
        "SKU_0": {"mean": 1.41, "std": 2.0},
        "SKU_1": {"mean": 1.06, "std": 1.4},
        "SKU_2": {"mean": 0.77, "std": 1.2},
        "label": "S3: Extreme Stress\n(High Volatility)",
        "short": "S3-Extreme",
    },
}


def _apply_scenario(base_env, scenario: dict):
    """Inject per-SKU mean/std into a MultiDCInventoryEnv instance."""
    n_skus = getattr(base_env, 'n_skus', 3)
    new_mean = np.array(base_env.demand_mean, dtype=float).copy() if hasattr(base_env, 'demand_mean') else np.zeros(n_skus)
    new_std  = np.array(base_env.demand_std,  dtype=float).copy() if hasattr(base_env, 'demand_std')  else np.ones(n_skus)
    for sku_idx in range(n_skus):
        key = f'SKU_{sku_idx}'
        if key in scenario:
            new_mean[sku_idx] = scenario[key]['mean']
            new_std[sku_idx]  = scenario[key]['std']
    base_env.demand_mean = new_mean
    base_env.demand_std  = new_std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
#  SECTION 1 — (s,S) BASE-STOCK EVALUATOR
# ============================================================================

class SsPolicy:
    """
    (s, S) Inventory Policy — reorder only when Inventory Position drops at or
    below the reorder point `s`; then order up to the order-up-to level `S`.
    """
    def __init__(self, s_dc, S_dc, s_retailer, S_retailer, n_dcs, n_agents, n_skus):
        assert S_dc > s_dc,             f'S_dc ({S_dc}) must be > s_dc ({s_dc})'
        assert S_retailer > s_retailer, f'S_retailer ({S_retailer}) must be > s_retailer ({s_retailer})'
        self.s_dc = s_dc
        self.S_dc = S_dc
        self.s_retailer = s_retailer
        self.S_retailer = S_retailer
        self.n_dcs = n_dcs
        self.n_agents = n_agents
        self.n_skus = n_skus

    def get_actions(self, env: MultiDCInventoryEnv) -> dict:
        actions = {}
        for agent_id in range(self.n_agents):
            order = np.zeros(self.n_skus, dtype=np.float32)
            for sku in range(self.n_skus):
                on_hand = float(env.inventory[agent_id][sku])
                pipeline_qty = sum(
                    o['qty'] for o in env.pipeline[agent_id] if o['sku'] == sku
                )
                if agent_id < self.n_dcs:
                    owed = sum(
                        env.dc_retailer_backlog[agent_id][r_id][sku]
                        for r_id in env.dc_assignments[agent_id]
                    )
                    ip = on_hand - owed + pipeline_qty
                    s, S = self.s_dc, self.S_dc
                else:
                    backlog = float(env.backlog[agent_id][sku])
                    ip = on_hand - backlog + pipeline_qty
                    s, S = self.s_retailer, self.S_retailer
                if ip <= s:
                    order[sku] = max(0.0, S)
            actions[agent_id] = order
        return actions


BaseStockPolicy = SsPolicy  # alias


class BaseStockEvaluator:
    """
    Evaluates the (s,S) policy on the multi-DC environment.
    """

    def __init__(self, args, demand_scenario=None, config_path='configs/multi_dc_config.yaml'):
        self.args = args
        self.demand_scenario = demand_scenario  # dict with per-SKU mean/std (or None)
        self._config_path = config_path

        # Create environment
        self.env = self._create_env()
        self.n_agents = self.env.n_agents
        self.n_dcs    = self.env.n_dcs
        self.n_skus   = self.env.n_skus

        # Apply demand scenario override
        if demand_scenario is not None:
            _apply_scenario(self.env, demand_scenario)

        # (s,S) policy
        self.policy = SsPolicy(
            s_dc=args.s_dc, S_dc=args.S_dc,
            s_retailer=args.s_retailer, S_retailer=args.S_retailer,
            n_dcs=self.n_dcs, n_agents=self.n_agents, n_skus=self.n_skus,
        )

        self.episode_metrics = []

    # ------------------------------------------------------------------
    def _create_env(self):
        env = MultiDCInventoryEnv(config_path=self._config_path)
        env.max_days = self.args.episode_length
        return env

    # ------------------------------------------------------------------
    def evaluate(self):
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)

        # Patch clip so (s,S) quantities are not capped at 10
        _original_clip = self.env._clip_actions

        def _ss_clip_actions(acts):
            clipped = {}
            for aid, act in acts.items():
                if aid in self.env.dc_ids:
                    clipped[aid] = np.clip(act, 0, 5000)
                else:
                    clipped[aid] = np.maximum(0.0, np.array(act, dtype=np.float32))
            return clipped

        self.env._clip_actions = _ss_clip_actions

        for ep in range(self.args.num_episodes):
            metrics = self._run_episode(ep)
            self.episode_metrics.append(metrics)

        self.env._clip_actions = _original_clip

    # ------------------------------------------------------------------
    def _run_episode(self, episode_num):
        obs = self.env.reset()

        ep_data = {
            'total_reward':   0.0,
            'total_cost':     0.0,
            'agent_rewards':  [0.0] * self.n_agents,
            'agent_costs':    [0.0] * self.n_agents,
            'holding_costs':  [0.0] * self.n_agents,
            'backlog_costs':  [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            'avg_inventory':  [0.0] * self.n_agents,
            'avg_backlog':    [0.0] * self.n_agents,
            'service_level':  [0.0] * self.n_agents,
            '_orders_placed':     [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
            'dc_cycle_service_level': {},
            'final_inventory': None,
            'final_backlog':   None,
            # Trajectory: step-level retail demand & orders (list of arrays, one per step)
            'traj_demand': [np.zeros(self.n_skus) for _ in range(self.args.episode_length)],
            'traj_orders': [np.zeros(self.n_skus) for _ in range(self.args.episode_length)],
        }

        for step in range(self.args.episode_length):
            pre_step_prices = self.env.market_prices.copy()
            actions = self.policy.get_actions(self.env)

            try:
                obs, rewards, dones, infos = self.env.step(actions)
            except Exception as exc:
                print(f'[WARNING] env.step() raised an exception at step {step}: {exc}')
                break

            executed_actions = {
                aid: np.maximum(0.0, np.array(a, dtype=np.float32))
                for aid, a in actions.items()
            }

            for agent_id in range(self.n_agents):
                reward = float(rewards[agent_id])
                cost   = -reward
                ep_data['agent_rewards'][agent_id] += reward
                ep_data['agent_costs'][agent_id]   += cost
                ep_data['total_reward']             += reward
                ep_data['total_cost']               += cost

                h_cost = b_cost = o_cost = 0.0
                is_dc = agent_id < self.n_dcs

                if is_dc:
                    dc_idx = agent_id
                    for sku in range(self.n_skus):
                        h_cost += self.env.inventory[agent_id][sku] * self.env.H_dc[dc_idx][sku]
                        dc_owed_sku = sum(
                            self.env.dc_retailer_backlog[agent_id][r_id][sku]
                            for r_id in self.env.dc_assignments[agent_id]
                        )
                        b_cost += dc_owed_sku * self.env.B_dc[dc_idx][sku]
                        act_sku = float(executed_actions[agent_id][sku])
                        if act_sku > 0:
                            price = pre_step_prices[sku]
                            o_cost += self.env.C_fixed_dc[dc_idx][sku] + price * act_sku
                else:
                    r_idx = agent_id - self.n_dcs
                    assigned_dc = self.env.retailer_to_dc[agent_id]
                    for sku in range(self.n_skus):
                        h_cost += self.env.inventory[agent_id][sku] * self.env.H_retailer[r_idx][sku]
                        b_cost += self.env.backlog[agent_id][sku]   * self.env.B_retailer[r_idx][sku]
                        order_qty = float(executed_actions[agent_id][sku])
                        if order_qty > 0:
                            o_cost += (
                                self.env.C_fixed_retailer[r_idx][sku]
                                + self.env.C_var_retailer[r_idx][assigned_dc][sku] * order_qty
                            )

                    for sku in range(self.n_skus):
                        placed     = self.env.step_orders_placed.get(agent_id, {}).get(sku, 0)
                        from_stock = self.env.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                        ep_data['_orders_placed'][agent_id]     += placed
                        ep_data['_orders_from_stock'][agent_id] += from_stock
                        # Trajectory: accumulate retail demand & orders per SKU
                        ep_data['traj_demand'][step][sku] += float(
                            self.env.step_orders_placed.get(agent_id, {}).get(sku, 0)
                        )
                        ep_data['traj_orders'][step][sku] += float(
                            executed_actions[agent_id][sku]
                        )

                ep_data['holding_costs'][agent_id]  += h_cost
                ep_data['backlog_costs'][agent_id]   += b_cost
                ep_data['ordering_costs'][agent_id]  += o_cost

                inv_vec = self.env.inventory[agent_id]
                inv     = float(inv_vec.sum())
                if is_dc:
                    bl = sum(
                        sum(self.env.dc_retailer_backlog[agent_id][r_id].values())
                        for r_id in self.env.dc_assignments[agent_id]
                    )
                else:
                    bl = float(self.env.backlog[agent_id].sum())
                ep_data['avg_inventory'][agent_id] += inv
                ep_data['avg_backlog'][agent_id]   += bl

            if step == self.args.episode_length - 1:
                ep_data['final_inventory'] = [
                    float(self.env.inventory[i].sum()) for i in range(self.n_agents)
                ]
                ep_data['final_backlog'] = [
                    float(self.env.backlog[i].sum()) for i in range(self.n_agents)
                ]

            info_dc_sl = {}
            if isinstance(infos, dict):
                for v in infos.values():
                    if isinstance(v, dict) and 'dc_cycle_service_level' in v:
                        info_dc_sl = v['dc_cycle_service_level']
                        break
            ep_data['dc_cycle_service_level'] = {
                int(dc_id): float(sl_val) for dc_id, sl_val in info_dc_sl.items()
            }

        T = self.args.episode_length
        for agent_id in range(self.n_agents):
            ep_data['avg_inventory'][agent_id] /= T
            ep_data['avg_backlog'][agent_id]   /= T
            if agent_id >= self.n_dcs:
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                dc_sl_map = ep_data.get('dc_cycle_service_level', {})
                ep_data['service_level'][agent_id] = dc_sl_map.get(agent_id, 100.0)

        return ep_data

    # ------------------------------------------------------------------
    def mean_total_cost(self):
        costs = []
        for m in self.episode_metrics:
            total_holding  = float(np.sum(m['holding_costs']))
            total_backlog  = float(np.sum(m['backlog_costs']))
            total_ordering = float(np.sum(m['ordering_costs']))
            costs.append(total_holding + total_backlog + total_ordering)
        return float(np.mean(costs)), float(np.std(costs))

    def mean_fill_rate(self):
        rates = []
        for m in self.episode_metrics:
            total_placed     = sum(m['_orders_placed'][aid]     for aid in range(self.n_dcs, self.n_agents))
            total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(self.n_dcs, self.n_agents))
            rates.append((total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0)
        return float(np.mean(rates))

    def get_trajectory(self):
        """Returns step-level demand and orders for the first episode."""
        if not self.episode_metrics: return None, None
        return self.episode_metrics[0]['traj_demand'], self.episode_metrics[0]['traj_orders']


# ============================================================================
#  SECTION 2 — HAPPO EVALUATOR
# ============================================================================

class HAPPOEvaluator:
    """
    Evaluates trained HAPPO models and generates comprehensive metrics.
    """

    def __init__(self, args, demand_scenario=None):
        self.args = args
        self.demand_scenario = demand_scenario  # dict with per-SKU mean/std (or None)
        self.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

        # Create environment
        self.env = self._create_env()

        # Build adjacency matrix for GNN (if needed)
        self.adj_tensor = None
        if getattr(args, 'algorithm_name', 'happo') == 'gnn_happo':
            from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
            adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True)
            adj = normalize_adjacency(adj, method='symmetric')
            self.adj_tensor = torch.FloatTensor(adj).to(self.device)

        # Load trained models
        self.policies = self._load_models()

        # Metrics storage
        self.episode_metrics = []
        self.detailed_trajectory = None

    def _create_env(self):
        parser = get_config()
        parser.set_defaults(
            env_name="MultiDC",
            scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents,
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name="happo"
        )
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)

        if hasattr(env, 'num_agent'):
            self.args.num_agents = env.num_agent

        # Apply demand scenario override
        if self.demand_scenario is not None:
            env_list = getattr(env, 'env_list', getattr(env, 'envs', None))
            if env_list:
                _apply_scenario(env_list[0], self.demand_scenario)

        return env

    def _load_models(self):
        algorithm_name = getattr(self.args, 'algorithm_name', 'happo')
        is_gnn = (algorithm_name == 'gnn_happo')

        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        policies = []

        parser = get_config()
        defaults = dict(
            env_name="MultiDC",
            scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents,
            use_centralized_V=True,
            algorithm_name=algorithm_name,
            hidden_size=128,
            layer_N=2,
            use_ReLU=True,
            use_orthogonal=True,
            gain=0.01,
            recurrent_N=2,
            use_naive_recurrent_policy=True
        )
        if is_gnn:
            parser.add_argument('--gnn_type', type=str, default='GAT')
            parser.add_argument('--gnn_hidden_dim', type=int, default=128)
            parser.add_argument('--gnn_num_layers', type=int, default=2)
            parser.add_argument('--num_attention_heads', type=int, default=4)
            parser.add_argument('--gnn_dropout', type=float, default=0.1)
            parser.add_argument('--use_residual', type=lambda x: (str(x).lower() == 'true'), default=True)
            parser.add_argument('--critic_pooling', type=str, default='mean')
            parser.add_argument('--single_agent_obs_dim', type=int, default=36)
            defaults['single_agent_obs_dim'] = 36
        parser.set_defaults(**defaults)
        all_args = parser.parse_known_args([])[0]

        max_obs_dim = max([self.env.observation_space[i].shape[0] for i in range(self.args.num_agents)])

        for agent_id in range(self.args.num_agents):
            obs_space       = self.env.observation_space[agent_id]
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space       = self.env.action_space[agent_id]

            agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
            if not agent_files:
                raise FileNotFoundError(f"No model found for agent {agent_id} in {model_dir}")

            best_file = None
            suffixed_files = []
            for f in agent_files:
                if f.name == f"actor_agent{agent_id}.pt":
                    continue
                try:
                    parts = f.name.split('_reward_')
                    if len(parts) == 2:
                        reward_val = float(parts[1].replace('.pt', ''))
                        suffixed_files.append((reward_val, f))
                except ValueError:
                    continue

            if suffixed_files:
                suffixed_files.sort(key=lambda x: x[0], reverse=True)
                _, best_file = suffixed_files[0]
            else:
                simple_path = model_dir / f"actor_agent{agent_id}.pt"
                best_file = simple_path if simple_path.exists() else agent_files[0]

            state_dict = torch.load(str(best_file), map_location=self.device)

            if is_gnn:
                from gymnasium import spaces as gym_spaces
                padded_obs_space = gym_spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_obs_dim,), dtype=np.float32
                )
                policy = GNN_HAPPO_Policy(
                    all_args, padded_obs_space, share_obs_space, act_space,
                    n_agents=self.args.num_agents, agent_id=agent_id, device=self.device
                )
            else:
                saved_input_dim = None
                if 'base.mlp.fc1.0.weight' in state_dict:
                    saved_input_dim = state_dict['base.mlp.fc1.0.weight'].shape[1]
                elif 'base.cnn.cnn.0.weight' in state_dict:
                    saved_input_dim = state_dict['base.cnn.cnn.0.weight'].shape[1]

                current_obs_dim = obs_space.shape[0]
                if saved_input_dim is not None and saved_input_dim != current_obs_dim:
                    from gymnasium import spaces as gym_spaces
                    obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(saved_input_dim,), dtype=np.float32)

                policy = HAPPO_Policy(all_args, obs_space, share_obs_space, act_space, device=self.device)

            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()
            policies.append(policy)

        return policies

    def evaluate(self):
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for episode in range(self.args.num_episodes):
            metrics = self._run_episode(episode, save_trajectory=(episode == 0))
            self.episode_metrics.append(metrics)

    def _run_episode(self, episode_num, save_trajectory=False):
        obs, _ = self.env.reset()
        is_gnn = (getattr(self.args, 'algorithm_name', 'happo') == 'gnn_happo')
        rnn_states = np.zeros((1, self.args.num_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.args.num_agents, 1), dtype=np.float32)

        if is_gnn:
            max_obs_dim = max([self.env.observation_space[i].shape[0] for i in range(self.args.num_agents)])

        def _build_gnn_obs(obs_array):
            structured = np.zeros((1, self.args.num_agents, max_obs_dim), dtype=np.float32)
            for aid in range(self.args.num_agents):
                agent_obs = np.stack(obs_array[:, aid])
                obs_dim = agent_obs.shape[1]
                structured[0, aid, :obs_dim] = agent_obs[0]
            return structured

        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        episode_data = {
            'total_reward': 0,
            'total_cost': 0,
            'agent_rewards': [0] * self.args.num_agents,
            'agent_costs': [0] * self.args.num_agents,
            'holding_costs': [0] * self.args.num_agents,
            'backlog_costs': [0] * self.args.num_agents,
            'ordering_costs': [0] * self.args.num_agents,
            'final_inventory': None,
            'final_backlog': None,
            'avg_inventory': [0] * self.args.num_agents,
            'avg_backlog': [0] * self.args.num_agents,
            'service_level': [0] * self.args.num_agents,
            '_demand_total': [0.0] * self.args.num_agents,
            '_demand_met':   [0.0] * self.args.num_agents,
            '_orders_placed':     [0] * self.args.num_agents,
            '_orders_from_stock': [0] * self.args.num_agents,
            # Trajectory: step-level retail demand & orders per SKU
            'traj_demand': [np.zeros(n_skus) for _ in range(self.args.episode_length)],
            'traj_orders': [np.zeros(n_skus) for _ in range(self.args.episode_length)],
        }


        for step in range(self.args.episode_length):
            env_list_pre = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = None
            if env_list_pre:
                pre_step_prices = env_list_pre[0].market_prices.copy()

            actions_env = []
            raw_actions = {}

            if is_gnn:
                obs_structured = _build_gnn_obs(obs)

            for agent_id in range(self.args.num_agents):
                self.policies[agent_id].actor.eval()

                with torch.no_grad():
                    if is_gnn:
                        action, rnn_state = self.policies[agent_id].act(
                            obs_structured, self.adj_tensor, agent_id,
                            rnn_states[:, agent_id], masks[:, agent_id],
                            None, deterministic=True
                        )
                    else:
                        obs_agent = np.stack(obs[:, agent_id])
                        policy_input_dim = self.policies[agent_id].obs_space.shape[0]
                        current_obs_dim = obs_agent.shape[1]
                        if current_obs_dim < policy_input_dim:
                            diff = policy_input_dim - current_obs_dim
                            padding = np.zeros((obs_agent.shape[0], diff), dtype=np.float32)
                            obs_agent = np.concatenate([obs_agent, padding], axis=1)

                        action, rnn_state = self.policies[agent_id].act(
                            obs_agent, rnn_states[:, agent_id], masks[:, agent_id],
                            deterministic=True, agent_id=agent_id
                        )

                rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                raw_action = action_np[0]

                # DC IP-sufficiency guard (inference-time)
                if agent_id < 2 and env_list_pre:
                    _env = env_list_pre[0]
                    _z     = 1.65
                    _lt    = 14
                    _n_ret = len(_env.dc_assignments[agent_id])
                    _zero_action = True
                    for _sku in range(n_skus):
                        _mu    = float(_env.demand_mean[_sku]) * _n_ret
                        _sigma = float(_env.demand_std[_sku])  * _n_ret
                        _out_level = _mu * _lt + _z * _sigma * float(np.sqrt(_lt))
                        _on_hand   = float(_env.inventory[agent_id][_sku])
                        _owed      = sum(
                            _env.dc_retailer_backlog[agent_id][r_id][_sku]
                            for r_id in _env.dc_assignments[agent_id]
                        )
                        _pipeline  = sum(
                            o['qty'] for o in _env.pipeline[agent_id] if o['sku'] == _sku
                        )
                        _ip = _on_hand - _owed + _pipeline
                        if _ip < _out_level:
                            _zero_action = False
                            break
                    if _zero_action:
                        raw_action = np.zeros_like(raw_action)

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list and len(env_list) > 0:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)
            else:
                executed_actions = {aid: raw_actions[aid] for aid in range(self.args.num_agents)}

            if env_list and len(env_list) > 0:
                for agent_id in range(self.args.num_agents):
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward
                    episode_data['agent_rewards'][agent_id] += reward
                    episode_data['agent_costs'][agent_id]   += cost
                    episode_data['total_reward'] += reward
                    episode_data['total_cost']   += cost

                    holding_cost_step = backlog_cost_step = ordering_cost_step = 0
                    is_dc = agent_id < 2

                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_dc[dc_idx][sku]
                            dc_owed_sku = sum(
                                env_state.dc_retailer_backlog[agent_id][r_id][sku]
                                for r_id in env_state.dc_assignments[agent_id]
                            )
                            backlog_cost_step += dc_owed_sku * env_state.B_dc[dc_idx][sku]
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                price = pre_step_prices[sku] if pre_step_prices is not None else env_state.market_prices[sku]
                                ordering_cost_step += env_state.C_fixed_dc[dc_idx][sku] + price * order_qty
                    else:
                        retailer_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_retailer[retailer_idx][sku]
                            backlog_cost_step  += env_state.backlog[agent_id][sku]  * env_state.B_retailer[retailer_idx][sku]
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                var_cost = env_state.C_var_retailer[retailer_idx][assigned_dc][sku]
                                ordering_cost_step += env_state.C_fixed_retailer[retailer_idx][sku] + var_cost * order_qty

                        for sku in range(n_skus):
                            placed     = env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            from_stock = env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                            episode_data['_orders_placed'][agent_id]     += placed
                            episode_data['_orders_from_stock'][agent_id] += from_stock
                            actual_demand = 0.0
                            if (sku < len(env_state.demand_history) and
                                    retailer_idx < len(env_state.demand_history[sku]) and
                                    len(env_state.demand_history[sku][retailer_idx]) > 0):
                                actual_demand = float(env_state.demand_history[sku][retailer_idx][-1])
                            episode_data['_demand_total'][agent_id] += actual_demand
                            episode_data['_demand_met'][agent_id]   += float(from_stock)
                            # Trajectory
                            episode_data['traj_demand'][step][sku] += actual_demand
                            episode_data['traj_orders'][step][sku] += float(executed_actions[agent_id][sku])

                    episode_data['holding_costs'][agent_id]  += holding_cost_step
                    episode_data['backlog_costs'][agent_id]  += backlog_cost_step
                    episode_data['ordering_costs'][agent_id] += ordering_cost_step

                    inv = env_state.inventory[agent_id].sum()
                    bl  = env_state.backlog[agent_id].sum()
                    episode_data['avg_inventory'][agent_id] += inv
                    episode_data['avg_backlog'][agent_id]   += bl

                if step == self.args.episode_length - 1:
                    episode_data['final_inventory'] = [
                        env_state.inventory[i].sum() for i in range(self.args.num_agents)
                    ]
                    episode_data['final_backlog'] = [
                        env_state.backlog[i].sum() for i in range(self.args.num_agents)
                    ]

        T = self.args.episode_length
        for agent_id in range(self.args.num_agents):
            episode_data['avg_inventory'][agent_id] /= T
            episode_data['avg_backlog'][agent_id]   /= T
            if agent_id >= 2:
                placed     = episode_data['_orders_placed'][agent_id]
                from_stock = episode_data['_orders_from_stock'][agent_id]
                episode_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                episode_data['service_level'][agent_id] = 100.0

        return episode_data

    # ------------------------------------------------------------------
    def mean_total_cost(self):
        costs = []
        for m in self.episode_metrics:
            total_holding  = float(np.sum(m['holding_costs']))
            total_backlog  = float(np.sum(m['backlog_costs']))
            total_ordering = float(np.sum(m['ordering_costs']))
            costs.append(total_holding + total_backlog + total_ordering)
        return float(np.mean(costs)), float(np.std(costs))

    def mean_fill_rate(self):
        n_dcs = 2
        rates = []
        for m in self.episode_metrics:
            total_placed     = sum(m['_orders_placed'][aid]     for aid in range(n_dcs, self.args.num_agents))
            total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(n_dcs, self.args.num_agents))
            rates.append((total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0)
        return float(np.mean(rates))

    def get_trajectory(self):
        """Returns step-level demand and orders for the first episode."""
        if not self.episode_metrics: return None, None
        return self.episode_metrics[0]['traj_demand'], self.episode_metrics[0]['traj_orders']


# ============================================================================
#  SECTION 3 — GNN-HAPPO EVALUATOR
# ============================================================================

class GNNModelEvaluator:
    """
    Evaluates trained GNN-HAPPO models on the multi-DC inventory environment.
    """

    def __init__(self, args, demand_scenario=None):
        self.args = args
        self.demand_scenario = demand_scenario  # dict with per-SKU mean/std (or None)
        self.device = torch.device(
            'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
        )

        # 1. Create environment
        self.env = self._create_env()

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

    def _create_env(self):
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

        # Apply demand scenario override
        if self.demand_scenario is not None:
            env_list = getattr(env, 'env_list', getattr(env, 'envs', None))
            if env_list:
                _apply_scenario(env_list[0], self.demand_scenario)

        return env

    def _build_graph(self):
        adj = build_supply_chain_adjacency(
            n_dcs=2, n_retailers=self.n_agents - 2, self_loops=True
        )
        adj = normalize_adjacency(adj, method='symmetric')
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        return adj_tensor

    def _detect_model_config(self):
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
            obs_dim = max(self.env.observation_space[i].shape[0] for i in range(self.n_agents))

        self.detected_gnn_type = detected_gnn_type
        return obs_dim

    def _build_all_args(self):
        gnn_type = getattr(self, 'detected_gnn_type', self.args.gnn_type)
        parser = get_config()
        parser.add_argument('--gnn_type', type=str, default=gnn_type)
        parser.add_argument('--gnn_hidden_dim', type=int, default=self.args.gnn_hidden_dim)
        parser.add_argument('--gnn_num_layers', type=int, default=self.args.gnn_num_layers)
        parser.add_argument('--num_attention_heads', type=int, default=self.args.num_attention_heads)
        parser.add_argument('--gnn_dropout', type=float, default=self.args.gnn_dropout)
        parser.add_argument('--use_residual', type=lambda x: x.lower() == 'true', default=self.args.use_residual)
        parser.add_argument('--critic_pooling', type=str, default=self.args.critic_pooling)
        parser.add_argument('--single_agent_obs_dim', type=int, default=self.single_agent_obs_dim)
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
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f'Model directory not found: {model_dir}')

        all_args = self._build_all_args()

        from gymnasium import spaces as gym_spaces
        padded_obs_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.single_agent_obs_dim,), dtype=np.float32
        )

        policies = []
        for agent_id in range(self.n_agents):
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]
            best_file = self._find_best_model(model_dir, agent_id)

            policy = GNN_HAPPO_Policy(
                all_args, padded_obs_space, share_obs_space, act_space,
                n_agents=self.n_agents, agent_id=agent_id, device=self.device,
            )

            state_dict = torch.load(str(best_file), map_location=self.device)
            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()
            policies.append(policy)

        return policies

    def _find_best_model(self, model_dir, agent_id):
        all_files = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
        if not all_files:
            raise FileNotFoundError(f'No model file found for agent {agent_id} in {model_dir}')

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
            _, best_file = reward_files[0]
            return best_file

        plain = model_dir / f'actor_agent{agent_id}.pt'
        return plain if plain.exists() else all_files[0]

    def evaluate(self):
        import torch as _torch
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

        for ep in range(self.args.num_episodes):
            metrics = self._run_episode(ep, save_trajectory=(ep == 0))
            self.episode_metrics.append(metrics)

    def _run_episode(self, episode_num, save_trajectory=False):
        obs, _ = self.env.reset()
        max_obs_dim = self.single_agent_obs_dim

        rnn_states = np.zeros((1, self.n_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.n_agents, 1), dtype=np.float32)

        ep_data = {
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
            # Trajectory: step-level retail demand & orders per SKU
            'traj_demand': [np.zeros(3) for _ in range(self.args.episode_length)],
            'traj_orders': [np.zeros(3) for _ in range(self.args.episode_length)],
        }

        for step in range(self.args.episode_length):
            _pre_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = _pre_env_list[0].market_prices.copy() if _pre_env_list else None

            obs_structured = np.zeros((1, self.n_agents, max_obs_dim), dtype=np.float32)
            for aid in range(self.n_agents):
                raw = np.stack(obs[:, aid])
                d = raw.shape[1]
                obs_structured[0, aid, :d] = raw[0]

            actions_env = []
            raw_actions = {}
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_structured, self.adj_tensor, agent_id,
                        rnn_states[:, agent_id], masks[:, agent_id],
                        deterministic=False,
                    )

                rnn_states[:, agent_id] = (
                    rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                )
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                raw_action = action_np[0]

                # DC IP-sufficiency guard
                if agent_id < 2 and _pre_env_list:
                    _env = _pre_env_list[0]
                    _z     = 1.4
                    _lt    = 7
                    _n_ret = len(_env.dc_assignments[agent_id])
                    _zero_action = True
                    for _sku in range(self.n_skus):
                        _mu    = float(_env.demand_mean[_sku]) * _n_ret
                        _sigma = float(_env.demand_std[_sku])  * _n_ret
                        _out_level = _mu * _lt + _z * _sigma * float(np.sqrt(_lt))
                        _on_hand   = float(_env.inventory[agent_id][_sku])
                        _owed      = sum(
                            _env.dc_retailer_backlog[agent_id][r_id][_sku]
                            for r_id in _env.dc_assignments[agent_id]
                        )
                        _pipeline  = sum(
                            o['qty'] for o in _env.pipeline[agent_id] if o['sku'] == _sku
                        )
                        _ip = _on_hand - _owed + _pipeline
                        if _ip < _out_level:
                            _zero_action = False
                            break
                    if _zero_action:
                        raw_action = np.zeros_like(raw_action)

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.n_agents):
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward
                    ep_data['agent_rewards'][agent_id] += reward
                    ep_data['agent_costs'][agent_id]   += cost
                    ep_data['total_reward'] += reward
                    ep_data['total_cost']   += cost

                    h_cost = b_cost = o_cost = 0.0
                    is_dc = agent_id < 2
                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(3):
                            h_cost += env_state.inventory[agent_id][sku] * env_state.H_dc[dc_idx][sku]
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
                            h_cost += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                            b_cost += env_state.backlog[agent_id][sku]   * env_state.B_retailer[r_idx][sku]
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                o_cost += (env_state.C_fixed_retailer[r_idx][sku]
                                           + env_state.C_var_retailer[r_idx][assigned_dc][sku] * order_qty)

                        for sku in range(3):
                            placed     = env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            from_stock = env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                            ep_data['_orders_placed'][agent_id]     += placed
                            ep_data['_orders_from_stock'][agent_id] += from_stock
                            
                            actual_demand = 0.0
                            if (sku < len(env_state.demand_history) and
                                    r_idx < len(env_state.demand_history[sku]) and
                                    len(env_state.demand_history[sku][r_idx]) > 0):
                                actual_demand = float(env_state.demand_history[sku][r_idx][-1])
                            
                            # Trajectory
                            ep_data['traj_demand'][step][sku] += actual_demand
                            ep_data['traj_orders'][step][sku] += float(executed_actions[agent_id][sku])

                    ep_data['holding_costs'][agent_id]  += h_cost
                    ep_data['backlog_costs'][agent_id]  += b_cost
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
                    ep_data['avg_backlog'][agent_id]   += bl

                if step == self.args.episode_length - 1:
                    ep_data['final_inventory'] = [
                        float(env_state.inventory[i].sum()) for i in range(self.n_agents)
                    ]
                    ep_data['final_backlog'] = [
                        float(env_state.backlog[i].sum()) for i in range(self.n_agents)
                    ]

            info_dc_sl = infos[0][0].get('dc_cycle_service_level', {}) if infos and infos[0] else {}
            ep_data['dc_cycle_service_level'] = {
                int(dc_id): float(sl_val) for dc_id, sl_val in info_dc_sl.items()
            }

        T = self.args.episode_length
        for agent_id in range(self.n_agents):
            ep_data['avg_inventory'][agent_id] /= T
            ep_data['avg_backlog'][agent_id]   /= T
            if agent_id >= 2:
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                dc_sl_map = ep_data.get('dc_cycle_service_level', {})
                ep_data['service_level'][agent_id] = dc_sl_map.get(agent_id, 100.0)

        return ep_data

    # ------------------------------------------------------------------
    def mean_total_cost(self):
        costs = []
        for m in self.episode_metrics:
            total_holding  = float(np.sum(m['holding_costs']))
            total_backlog  = float(np.sum(m['backlog_costs']))
            total_ordering = float(np.sum(m['ordering_costs']))
            costs.append(total_holding + total_backlog + total_ordering)
        return float(np.mean(costs)), float(np.std(costs))

    def mean_fill_rate(self):
        rates = []
        for m in self.episode_metrics:
            total_placed     = sum(m['_orders_placed'][aid]     for aid in range(2, self.n_agents))
            total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(2, self.n_agents))
            rates.append((total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0)
        return float(np.mean(rates))

    def get_trajectory(self):
        """Returns step-level demand and orders for the first episode."""
        if not self.episode_metrics: return None, None
        return self.episode_metrics[0]['traj_demand'], self.episode_metrics[0]['traj_orders']


# ============================================================================
#  SECTION 4 — Argument parsing & main orchestration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Robustness comparison: 3 models × 3 demand scenarios'
    )

    # -- Paths (hard-coded defaults) --
    parser.add_argument('--gnn_model_dir', type=str,
                        default='results/14Apr_gnn_kaggle_vari/run_seed_1/models',
                        help='GNN-HAPPO model directory')
    parser.add_argument('--happo_model_dir', type=str,
                        default='results/01Apr_base/run_seed_1/models',
                        help='HAPPO model directory')

    # -- Episode settings --
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of evaluation episodes')
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Episode length for GNN-HAPPO (days)')
    parser.add_argument('--happo_episode_length', type=int, default=115,
                        help='Episode length for HAPPO (days)')
    parser.add_argument('--basestock_episode_length', type=int, default=120,
                        help='Episode length used exclusively for the (s,S) BaseStock evaluator')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # -- Environment --
    parser.add_argument('--config_path', type=str, default='configs/multi_dc_config.yaml',
                        help='Path to environment YAML config (for BaseStock env)')
    parser.add_argument('--num_agents', type=int, default=17,
                        help='Number of agents (2 DCs + 15 Retailers)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA if available')

    # -- (s,S) policy params --
    parser.add_argument('--s_dc',         type=float, default=100.0)
    parser.add_argument('--S_dc',         type=float, default=170.0)
    parser.add_argument('--s_retailer',   type=float, default=3.0)
    parser.add_argument('--S_retailer',   type=float, default=10.0)

    # -- GNN architecture --
    parser.add_argument('--gnn_type',             type=str,   default='GAT')
    parser.add_argument('--gnn_hidden_dim',       type=int,   default=128)
    parser.add_argument('--gnn_num_layers',       type=int,   default=2)
    parser.add_argument('--num_attention_heads',  type=int,   default=4)
    parser.add_argument('--gnn_dropout',          type=float, default=0.1)
    parser.add_argument('--use_residual',         type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--critic_pooling',       type=str,   default='mean')

    # -- Output --
    parser.add_argument('--save_dir', type=str, default='evaluation_results/robustness_comparison',
                        help='Directory to save results')

    return parser.parse_args()


def run_basestock(args, scenario: dict, scenario_label: str):
    print(f'\n{"="*60}')
    print(f'  Running (s,S) BaseStock -- {scenario_label}')
    print(f'{"="*60}')
    import copy
    bs_args = copy.deepcopy(args)
    bs_args.episode_length = args.basestock_episode_length  # use dedicated episode length
    evaluator = BaseStockEvaluator(
        bs_args, demand_scenario=scenario,
        config_path=args.config_path
    )
    evaluator.evaluate()
    mean_cost, std_cost = evaluator.mean_total_cost()
    fill_rate = evaluator.mean_fill_rate()
    traj_d, traj_o = evaluator.get_trajectory()
    print(f'  --> Mean total cost: {mean_cost:,.2f}  (+/-{std_cost:,.2f})')
    print(f'  --> Mean fill rate:  {fill_rate:.1f}%')
    return mean_cost, std_cost, fill_rate, traj_d, traj_o


def run_happo(args, scenario: dict, scenario_label: str):
    print(f'\n{"="*60}')
    print(f'  Running HAPPO -- {scenario_label}')
    print(f'{"="*60}')
    import copy
    happo_args = copy.deepcopy(args)
    happo_args.episode_length = args.happo_episode_length
    happo_args.model_dir = args.happo_model_dir
    happo_args.algorithm_name = 'happo'
    evaluator = HAPPOEvaluator(happo_args, demand_scenario=scenario)
    evaluator.evaluate()
    mean_cost, std_cost = evaluator.mean_total_cost()
    fill_rate = evaluator.mean_fill_rate()
    traj_d, traj_o = evaluator.get_trajectory()
    print(f'  --> Mean total cost: {mean_cost:,.2f}  (+/-{std_cost:,.2f})')
    print(f'  --> Mean fill rate:  {fill_rate:.1f}%')
    return mean_cost, std_cost, fill_rate, traj_d, traj_o


def run_gnn(args, scenario: dict, scenario_label: str):
    print(f'\n{"="*60}')
    print(f'  Running GNN-HAPPO -- {scenario_label}')
    print(f'{"="*60}')
    import copy
    gnn_args = copy.deepcopy(args)
    gnn_args.model_dir = args.gnn_model_dir
    evaluator = GNNModelEvaluator(gnn_args, demand_scenario=scenario)
    evaluator.evaluate()
    mean_cost, std_cost = evaluator.mean_total_cost()
    fill_rate = evaluator.mean_fill_rate()
    traj_d, traj_o = evaluator.get_trajectory()
    print(f'  --> Mean total cost: {mean_cost:,.2f}  (+/-{std_cost:,.2f})')
    print(f'  --> Mean fill rate:  {fill_rate:.1f}%')
    return mean_cost, std_cost, fill_rate, traj_d, traj_o


def plot_total_cost(results, scenario_labels, save_dir):
    """Save a standalone bar chart of Mean Total Cost per scenario."""
    model_names = ['(s,S) BaseStock', 'HAPPO', 'GNN-HAPPO']
    colors      = ['#4ECDC4', '#FF6B6B', '#3A86FF']

    n_scenarios = len(scenario_labels)
    n_models    = len(model_names)

    means = np.array([[results[s][m][0] for m in range(n_models)] for s in range(n_scenarios)])
    stds  = np.array([[results[s][m][1] for m in range(n_models)] for s in range(n_scenarios)])

    x      = np.arange(n_scenarios)
    width  = 0.22
    offsets = np.array([-width, 0, width])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title('Total Cost Comparison\n3 Models x 3 Demand Scenarios',
                 fontsize=14, fontweight='bold', pad=12)

    for i, (name, color, offset) in enumerate(zip(model_names, colors, offsets)):
        bars = ax.bar(
            x + offset, means[:, i], width,
            label=name, color=color, alpha=0.88,
            edgecolor='black', linewidth=0.8,
            yerr=stds[:, i], capsize=5, error_kw=dict(elinewidth=1.2)
        )
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + stds[bar_idx, i] + max(means.max() * 0.005, 10),
                f'{h:,.0f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_ylabel('Mean Total Cost (per episode)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path = Path(save_dir) / 'robustness_total_cost.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Total cost chart saved to: {out_path}')


def plot_fill_rate(results, scenario_labels, save_dir):
    """Save a standalone bar chart of Mean Fill Rate per scenario."""
    model_names = ['(s,S) BaseStock', 'HAPPO', 'GNN-HAPPO']
    colors      = ['#4ECDC4', '#FF6B6B', '#3A86FF']

    n_scenarios = len(scenario_labels)
    n_models    = len(model_names)

    fills = np.array([[results[s][m][2] for m in range(n_models)] for s in range(n_scenarios)])

    x      = np.arange(n_scenarios)
    width  = 0.22
    offsets = np.array([-width, 0, width])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title('Fill Rate Comparison\n3 Models x 3 Demand Scenarios',
                 fontsize=14, fontweight='bold', pad=12)

    for i, (name, color, offset) in enumerate(zip(model_names, colors, offsets)):
        bars = ax.bar(
            x + offset, fills[:, i], width,
            label=name, color=color, alpha=0.88,
            edgecolor='black', linewidth=0.8
        )
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f'{h:.1f}%',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold'
            )

    ax.axhline(y=95, color='red', linestyle='--', linewidth=1.8,
               alpha=0.8, label='Target 95%')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_ylabel('Mean Fill Rate (%)', fontsize=12)
    ax.set_ylim([0, 110])
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path = Path(save_dir) / 'robustness_fill_rate.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Fill rate chart saved to: {out_path}')


def plot_demand_vs_orders(trajectories, model_names, scenario_labels, save_dir):
    """
    Saves a 3-row (scenarios) x 3-column (SKUs) line chart for each model
    showing step-level demand vs. orders.
    trajectories[s_idx][m_idx] = (traj_d, traj_o)
    """
    import re
    for m_idx, m_name in enumerate(model_names):
        fig, axes = plt.subplots(len(scenario_labels), 3, figsize=(15, 3.5 * len(scenario_labels)))
        fig.suptitle(f'Demand vs. Orders: {m_name}', fontsize=16, fontweight='bold', y=1.02)
        
        for s_idx, s_label in enumerate(scenario_labels):
            traj_d, traj_o = trajectories[s_idx][m_idx]
            
            # If a model failed to record trajectories, skip
            if traj_d is None or traj_o is None:
                continue
                
            d_arr = np.array(traj_d)
            o_arr = np.array(traj_o)
            
            for sku in range(3):
                ax = axes[s_idx, sku] if len(scenario_labels) > 1 else axes[sku]
                steps = np.arange(len(d_arr))
                
                ax.plot(steps, d_arr[:, sku], label='Retail Demand', color='#FF6B6B', alpha=0.85, linewidth=1.5)
                ax.plot(steps, o_arr[:, sku], label='Retail Orders', color='#3A86FF', alpha=0.85, linestyle='--', linewidth=1.5)
                
                ax.set_title(f'Scenario: {s_label.strip()} | SKU {sku}', fontsize=11)
                ax.set_xlabel('Time Step (Days)', fontsize=10)
                ax.set_ylabel('Quantity', fontsize=10)
                if s_idx == 0 and sku == 0:
                    ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', m_name.replace('(s,S)', 'Ss').strip('_'))
        out_path = Path(save_dir) / f'demand_order_traj_{safe_name}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[OK] Trajectory chart saved to: {out_path}')


def save_summary_csv(results, save_dir, scenario_labels, model_names):
    out_path = Path(save_dir) / 'robustness_summary.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Model', 'Mean_Total_Cost', 'Std_Total_Cost', 'Mean_Fill_Rate_%'])
        for s_idx, s_label in enumerate(scenario_labels):
            for m_idx, m_name in enumerate(model_names):
                mean_cost, std_cost, fill_rate = results[s_idx][m_idx]
                writer.writerow([s_label, m_name,
                                  round(mean_cost, 2), round(std_cost, 2), round(fill_rate, 2)])
    print(f'[OK] Summary CSV saved to: {out_path}')


def print_summary_table(results, scenario_labels, model_names):
    print('\n' + '=' * 80)
    print('ROBUSTNESS COMPARISON SUMMARY')
    print('=' * 80)
    header = f"{'Scenario':<30} {'Model':<18} {'Mean Cost':>14} {'Std Cost':>12} {'Fill Rate':>10}"
    print(header)
    print('-' * 80)
    for s_idx, s_label in enumerate(scenario_labels):
        for m_idx, m_name in enumerate(model_names):
            mean_cost, std_cost, fill_rate = results[s_idx][m_idx]
            print(f'{s_label:<30} {m_name:<18} {mean_cost:>14,.2f} {std_cost:>12,.2f} {fill_rate:>9.1f}%')
        print('-' * 80)
    print('=' * 80)


def main():
    args = parse_args()

    model_names = ['(s,S) BaseStock', 'HAPPO', 'GNN-HAPPO']

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 70)
    print('  ROBUSTNESS COMPARISON: 3 Models x 3 Demand Scenarios')
    print('=' * 70)
    print(f'  GNN-HAPPO model : {args.gnn_model_dir}')
    print(f'  HAPPO model     : {args.happo_model_dir}')
    print(f'  Num episodes    : {args.num_episodes}')
    print(f'  GNN ep. length  : {args.episode_length} days  (GNN-HAPPO)')
    print(f'  HAPPO ep. length: {args.happo_episode_length} days  (HAPPO)')
    print(f'  BaseStock ep.   : {args.basestock_episode_length} days  ((s,S) only)')
    print(f'  Seed            : {args.seed}')
    print('  Scenarios:')
    for sc_key, sc_val in DEMAND_SCENARIOS.items():
        print(f'    [{sc_val["short"]}]  ', end='')
        for sku in range(3):
            k = f'SKU_{sku}'
            if k in sc_val:
                print(f'SKU{sku}(mu={sc_val[k]["mean"]}, s={sc_val[k]["std"]})  ', end='')
        print()
    print(f'  Save dir        : {save_dir}')
    print('=' * 70 + '\n')

    # results[scenario_idx][model_idx] = (mean_cost, std_cost, fill_rate)
    # trajectories[scenario_idx][model_idx] = (traj_d, traj_o)
    results = []
    trajectories = []
    scenario_labels = []   # short labels for chart x-axis
    scenario_full_labels = []  # full labels for CSV / summary table

    for sc_key, sc_val in DEMAND_SCENARIOS.items():
        scenario_label = sc_val['short']
        chart_label    = sc_val['label']   # multiline label for plot
        scenario_labels.append(chart_label)
        scenario_full_labels.append(scenario_label)

        scenario_results = []
        scenario_trajs = []

        # 1. (s,S) BaseStock
        mean_cost, std_cost, fill_rate, traj_d, traj_o = run_basestock(args, sc_val, scenario_label)
        scenario_results.append((mean_cost, std_cost, fill_rate))
        scenario_trajs.append((traj_d, traj_o))

        # 2. HAPPO
        mean_cost, std_cost, fill_rate, traj_d, traj_o = run_happo(args, sc_val, scenario_label)
        scenario_results.append((mean_cost, std_cost, fill_rate))
        scenario_trajs.append((traj_d, traj_o))

        # 3. GNN-HAPPO
        mean_cost, std_cost, fill_rate, traj_d, traj_o = run_gnn(args, sc_val, scenario_label)
        scenario_results.append((mean_cost, std_cost, fill_rate))
        scenario_trajs.append((traj_d, traj_o))

        results.append(scenario_results)
        trajectories.append(scenario_trajs)

    # Print summary table
    print_summary_table(results, scenario_full_labels, model_names)

    # Save CSV
    save_summary_csv(results, save_dir, scenario_full_labels, model_names)

    # Generate separate bar charts
    plot_total_cost(results, scenario_labels, save_dir)
    plot_fill_rate(results, scenario_labels, save_dir)
    plot_demand_vs_orders(trajectories, model_names, scenario_full_labels, save_dir)

    print(f'\n[DONE] All results saved to: {save_dir}\n')


if __name__ == '__main__':
    main()
