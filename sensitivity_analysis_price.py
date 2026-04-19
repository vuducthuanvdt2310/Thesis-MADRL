#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis: Price Scenarios
=====================================================

Evaluates three inventory policies across different price market scenarios
and produces grouped bar charts comparing total cost and fill rate.

  Models:
    - (s,S) BaseStock Heuristic
    - HAPPO
    - GNN-HAPPO
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

PRICE_SCENARIOS = {
    "Scenario_1_Balanced": {
        "base_price": [1.0, 0.8, 1.5],
        "min_price": [0.6, 0.4, 1.0],
        "max_price": [1.8, 1.4, 2.5],
        "volatility": 0.15,
        "label": "Balanced Price",
        "short": "S1-Balanced"
    },
    "Scenario_2_HighPrice": {
        "base_price": [2.0, 1.6, 3.0],
        "min_price": [1.2, 0.8, 2.0],
        "max_price": [3.6, 2.8, 5.0],
        "volatility": 0.15,
        "label": "High Price",
        "short": "S2-HighPrice"
    },
    "Scenario_3_HighVariant": {
        "base_price": [1.0, 0.8, 1.5],
        "min_price": [0.2, 0.1, 0.3],
        "max_price": [3.0, 2.5, 4.5],
        "volatility": 0.50,
        "label": "High Variant",
        "short": "S3-HighVariant"
    }
}


def _apply_price_scenario(base_env, scenario: dict):
    """Inject price scenario parameters into a MultiDCInventoryEnv instance."""
    base_env.base_market_price = np.array(scenario['base_price'], dtype=np.float32)
    base_env.price_volatility = scenario['volatility']
    base_env.price_bounds = {
        'min': np.array(scenario['min_price'], dtype=np.float32),
        'max': np.array(scenario['max_price'], dtype=np.float32)
    }
    
    # Also update base_env.market_prices to reflect new base prices initially
    base_env.market_prices = base_env.base_market_price.copy()
    
    # update config
    if hasattr(base_env, 'config'):
        if 'pricing' not in base_env.config:
            base_env.config['pricing'] = {}
        base_env.config['pricing']['base_price'] = scenario['base_price']
        base_env.config['pricing']['min_price'] = scenario['min_price']
        base_env.config['pricing']['max_price'] = scenario['max_price']
        base_env.config['pricing']['volatility'] = scenario['volatility']


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
    def __init__(self, s_dc, S_dc, s_retailer, S_retailer, n_dcs, n_agents, n_skus):
        assert S_dc > s_dc, f'S_dc ({S_dc}) must be > s_dc ({s_dc})'
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
                pipeline_qty = sum(o['qty'] for o in env.pipeline[agent_id] if o['sku'] == sku)
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

BaseStockPolicy = SsPolicy


class BaseStockEvaluator:
    def __init__(self, args, scenario, config_path='configs/multi_dc_config.yaml'):
        self.args = args
        self.scenario = scenario
        self._config_path = config_path

        self.env = self._create_env()
        self.n_agents = self.env.n_agents
        self.n_dcs    = self.env.n_dcs
        self.n_skus   = self.env.n_skus

        _apply_price_scenario(self.env, self.scenario)

        self.policy = SsPolicy(
            s_dc=args.s_dc, S_dc=args.S_dc,
            s_retailer=args.s_retailer, S_retailer=args.S_retailer,
            n_dcs=self.n_dcs, n_agents=self.n_agents, n_skus=self.n_skus,
        )

        self.episode_metrics = []

    def _create_env(self):
        env = MultiDCInventoryEnv(config_path=self._config_path)
        env.max_days = self.args.episode_length
        return env

    def evaluate(self):
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)

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

    def _run_episode(self, episode_num):
        obs = self.env.reset()

        ep_data = {
            'holding_costs':  [0.0] * self.n_agents,
            'backlog_costs':  [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            '_orders_placed':     [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
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

                ep_data['holding_costs'][agent_id]  += h_cost
                ep_data['backlog_costs'][agent_id]   += b_cost
                ep_data['ordering_costs'][agent_id]  += o_cost

        return ep_data

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


# ============================================================================
#  SECTION 2 — HAPPO EVALUATOR
# ============================================================================

class HAPPOEvaluator:
    def __init__(self, args, scenario):
        self.args = args
        self.scenario = scenario
        self.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
        self.env = self._create_env()

        self.adj_tensor = None
        if getattr(args, 'algorithm_name', 'happo') == 'gnn_happo': # Just in case it's set
            pass

        self.policies = self._load_models()
        self.episode_metrics = []

    def _create_env(self):
        parser = get_config()
        parser.set_defaults(
            env_name="MultiDC", scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents, episode_length=self.args.episode_length,
            n_eval_rollout_threads=1, use_centralized_V=True, algorithm_name="happo"
        )
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)

        if hasattr(env, 'num_agent'):
            self.args.num_agents = env.num_agent

        env_list = getattr(env, 'env_list', getattr(env, 'envs', None))
        if env_list:
            for e in env_list:
                _apply_price_scenario(e, self.scenario)

        return env

    def _load_models(self):
        algorithm_name = getattr(self.args, 'algorithm_name', 'happo')
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        policies = []
        parser = get_config()
        defaults = dict(
            env_name="MultiDC", scenario_name="inventory_2echelon",
            num_agents=self.args.num_agents, use_centralized_V=True,
            algorithm_name=algorithm_name, hidden_size=128, layer_N=2,
            use_ReLU=True, use_orthogonal=True, gain=0.01, recurrent_N=2,
            use_naive_recurrent_policy=True
        )
        parser.set_defaults(**defaults)
        all_args = parser.parse_known_args([])[0]

        for agent_id in range(self.args.num_agents):
            obs_space       = self.env.observation_space[agent_id]
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space       = self.env.action_space[agent_id]

            agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
            if not agent_files:
                raise FileNotFoundError(f"No model found for agent {agent_id} in {model_dir}")

            suffixed_files = []
            for f in agent_files:
                if f.name == f"actor_agent{agent_id}.pt": continue
                try:
                    parts = f.name.split('_reward_')
                    if len(parts) == 2:
                        suffixed_files.append((float(parts[1].replace('.pt', '')), f))
                except ValueError:
                    continue

            if suffixed_files:
                suffixed_files.sort(key=lambda x: x[0], reverse=True)
                _, best_file = suffixed_files[0]
            else:
                simple_path = model_dir / f"actor_agent{agent_id}.pt"
                best_file = simple_path if simple_path.exists() else agent_files[0]

            state_dict = torch.load(str(best_file), map_location=self.device)

            saved_input_dim = None
            if 'base.mlp.fc1.0.weight' in state_dict:
                saved_input_dim = state_dict['base.mlp.fc1.0.weight'].shape[1]

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
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        for episode in range(self.args.num_episodes):
            metrics = self._run_episode()
            self.episode_metrics.append(metrics)

    def _run_episode(self):
        obs, _ = self.env.reset()
        rnn_states = np.zeros((1, self.args.num_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.args.num_agents, 1), dtype=np.float32)

        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        episode_data = {
            'holding_costs': [0] * self.args.num_agents,
            'backlog_costs': [0] * self.args.num_agents,
            'ordering_costs': [0] * self.args.num_agents,
            '_orders_placed':     [0] * self.args.num_agents,
            '_orders_from_stock': [0] * self.args.num_agents,
        }

        for step in range(self.args.episode_length):
            env_list_pre = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = env_list_pre[0].market_prices.copy() if env_list_pre else None

            actions_env, raw_actions = [], {}

            for agent_id in range(self.args.num_agents):
                with torch.no_grad():
                    obs_agent = np.stack(obs[:, agent_id])
                    policy_input_dim = self.policies[agent_id].obs_space.shape[0]
                    current_obs_dim = obs_agent.shape[1]
                    if current_obs_dim < policy_input_dim:
                        obs_agent = np.concatenate([obs_agent, np.zeros((obs_agent.shape[0], policy_input_dim - current_obs_dim), dtype=np.float32)], axis=1)

                    action, rnn_state = self.policies[agent_id].act(
                        obs_agent, rnn_states[:, agent_id], masks[:, agent_id],
                        deterministic=True, agent_id=agent_id
                    )

                rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                raw_action = action_np[0]

                if agent_id < 2 and env_list_pre:
                    _env = env_list_pre[0]
                    _z, _lt, _n_ret = 1.65, 14, len(_env.dc_assignments[agent_id])
                    _zero_action = True
                    for _sku in range(n_skus):
                        _mu, _sigma = float(_env.demand_mean[_sku]) * _n_ret, float(_env.demand_std[_sku]) * _n_ret
                        _out_level = _mu * _lt + _z * _sigma * float(np.sqrt(_lt))
                        _ip = float(_env.inventory[agent_id][_sku]) - sum(_env.dc_retailer_backlog[agent_id][r_id][_sku] for r_id in _env.dc_assignments[agent_id]) + sum(o['qty'] for o in _env.pipeline[agent_id] if o['sku'] == _sku)
                        if _ip < _out_level:
                            _zero_action = False
                            break
                    if _zero_action: raw_action = np.zeros_like(raw_action)

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list and len(env_list) > 0:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.args.num_agents):
                    holding_cost_step = backlog_cost_step = ordering_cost_step = 0
                    if agent_id < 2:
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_dc[agent_id][sku]
                            backlog_cost_step += sum(env_state.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env_state.dc_assignments[agent_id]) * env_state.B_dc[agent_id][sku]
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0: ordering_cost_step += env_state.C_fixed_dc[agent_id][sku] + (pre_step_prices[sku] if pre_step_prices is not None else env_state.market_prices[sku]) * order_qty
                    else:
                        r_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                            backlog_cost_step += env_state.backlog[agent_id][sku] * env_state.B_retailer[r_idx][sku]
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0: ordering_cost_step += env_state.C_fixed_retailer[r_idx][sku] + env_state.C_var_retailer[r_idx][assigned_dc][sku] * order_qty
                        for sku in range(n_skus):
                            episode_data['_orders_placed'][agent_id] += env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            episode_data['_orders_from_stock'][agent_id] += env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)

                    episode_data['holding_costs'][agent_id] += holding_cost_step
                    episode_data['backlog_costs'][agent_id] += backlog_cost_step
                    episode_data['ordering_costs'][agent_id] += ordering_cost_step

        return episode_data

    def mean_total_cost(self):
        costs = []
        for m in self.episode_metrics:
            costs.append(float(np.sum(m['holding_costs'])) + float(np.sum(m['backlog_costs'])) + float(np.sum(m['ordering_costs'])))
        return float(np.mean(costs)), float(np.std(costs))

    def mean_fill_rate(self):
        rates = []
        for m in self.episode_metrics:
            total_placed = sum(m['_orders_placed'][aid] for aid in range(2, self.args.num_agents))
            total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(2, self.args.num_agents))
            rates.append((total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0)
        return float(np.mean(rates))


# ============================================================================
#  SECTION 3 — GNN-HAPPO EVALUATOR
# ============================================================================

class GNNModelEvaluator:
    def __init__(self, args, scenario):
        self.args = args
        self.scenario = scenario
        self.device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

        self.env = self._create_env()
        base_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        self.n_skus = getattr(base_env_list[0], 'n_skus', 3) if base_env_list else 3
        
        self.adj_tensor = self._build_graph()
        self.single_agent_obs_dim = self._detect_model_config()
        self.policies = self._load_models()
        self.episode_metrics = []

    def _create_env(self):
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC', scenario_name='inventory_2echelon',
            episode_length=self.args.episode_length, n_eval_rollout_threads=1,
            use_centralized_V=True, algorithm_name='gnn_happo'
        )
        all_args = parser.parse_known_args([])[0]
        env = DummyVecEnvMultiDC(all_args)
        self.n_agents = env.num_agent if hasattr(env, 'num_agent') else 17

        env_list = getattr(env, 'env_list', getattr(env, 'envs', None))
        if env_list:
            for e in env_list:
                _apply_price_scenario(e, self.scenario)

        return env

    def _build_graph(self):
        adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=self.n_agents - 2, self_loops=True)
        adj = normalize_adjacency(adj, method='symmetric')
        return torch.FloatTensor(adj).to(self.device)

    def _detect_model_config(self):
        model_dir = Path(self.args.model_dir)
        agent0_files = sorted(model_dir.glob('actor_agent0*.pt'))
        if not agent0_files: raise FileNotFoundError(f'No actor_agent0 .pt file found in {model_dir}')

        sd = torch.load(str(agent0_files[0]), map_location='cpu')
        keys = list(sd.keys())

        if any('gnn_base.layers.0.weight' in k for k in keys):
            self.detected_gnn_type = 'GCN'
            return sd['gnn_base.layers.0.weight'].shape[0]
        elif any('gnn_base.layers.0.W' in k for k in keys):
            self.detected_gnn_type = 'GAT'
            return sd['gnn_base.layers.0.W'].shape[1]
        
        self.detected_gnn_type = self.args.gnn_type
        return max(self.env.observation_space[i].shape[0] for i in range(self.n_agents))

    def _build_all_args(self):
        parser = get_config()
        parser.add_argument('--gnn_type', type=str, default=getattr(self, 'detected_gnn_type', self.args.gnn_type))
        parser.add_argument('--gnn_hidden_dim', type=int, default=self.args.gnn_hidden_dim)
        parser.add_argument('--gnn_num_layers', type=int, default=self.args.gnn_num_layers)
        parser.add_argument('--num_attention_heads', type=int, default=self.args.num_attention_heads)
        parser.add_argument('--gnn_dropout', type=float, default=self.args.gnn_dropout)
        parser.add_argument('--use_residual', type=lambda x: str(x).lower() == 'true', default=self.args.use_residual)
        parser.add_argument('--critic_pooling', type=str, default=self.args.critic_pooling)
        parser.add_argument('--single_agent_obs_dim', type=int, default=self.single_agent_obs_dim)
        parser.set_defaults(
            env_name='MultiDC', scenario_name='inventory_2echelon', num_agents=self.n_agents,
            use_centralized_V=True, algorithm_name='gnn_happo', hidden_size=128, layer_N=2,
            use_ReLU=True, use_orthogonal=True, gain=0.01, recurrent_N=2, use_naive_recurrent_policy=True,
            single_agent_obs_dim=self.single_agent_obs_dim
        )
        return parser.parse_known_args([])[0]

    def _load_models(self):
        model_dir = Path(self.args.model_dir)
        all_args = self._build_all_args()
        from gymnasium import spaces as gym_spaces
        padded_obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(self.single_agent_obs_dim,), dtype=np.float32)

        policies = []
        for agent_id in range(self.n_agents):
            best_file = self._find_best_model(model_dir, agent_id)
            policy = GNN_HAPPO_Policy(
                all_args, padded_obs_space, self.env.share_observation_space[agent_id], self.env.action_space[agent_id],
                n_agents=self.n_agents, agent_id=agent_id, device=self.device
            )
            policy.actor.load_state_dict(torch.load(str(best_file), map_location=self.device))
            policy.actor.eval()
            policies.append(policy)
        return policies

    def _find_best_model(self, model_dir, agent_id):
        all_files = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
        reward_files = []
        for f in all_files:
            if f'actor_agent{agent_id}.pt' == f.name: continue
            try: reward_files.append((float(f.name.split('_reward_')[1].replace('.pt', '')), f))
            except (IndexError, ValueError): pass
        if reward_files:
            reward_files.sort(key=lambda x: x[0], reverse=True)
            return reward_files[0][1]
        plain = model_dir / f'actor_agent{agent_id}.pt'
        return plain if plain.exists() else all_files[0]

    def evaluate(self):
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        for ep in range(self.args.num_episodes):
            self.episode_metrics.append(self._run_episode())

    def _run_episode(self):
        obs, _ = self.env.reset()
        rnn_states = np.zeros((1, self.n_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.n_agents, 1), dtype=np.float32)

        ep_data = {
            'holding_costs': [0.0] * self.n_agents,
            'backlog_costs': [0.0] * self.n_agents,
            'ordering_costs': [0.0] * self.n_agents,
            '_orders_placed': [0] * self.n_agents,
            '_orders_from_stock': [0] * self.n_agents,
        }

        for step in range(self.args.episode_length):
            _pre_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = _pre_env_list[0].market_prices.copy() if _pre_env_list else None

            obs_structured = np.zeros((1, self.n_agents, self.single_agent_obs_dim), dtype=np.float32)
            for aid in range(self.n_agents):
                raw = np.stack(obs[:, aid])
                obs_structured[0, aid, :raw.shape[1]] = raw[0]

            actions_env, raw_actions = [], {}
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_structured, self.adj_tensor, agent_id,
                        rnn_states[:, agent_id], masks[:, agent_id], deterministic=False
                    )

                rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                raw_action = action_np[0]

                if agent_id < 2 and _pre_env_list:
                    _env = _pre_env_list[0]
                    _z, _lt, _n_ret = 1.4, 7, len(_env.dc_assignments[agent_id])
                    _zero_action = True
                    for _sku in range(self.n_skus):
                        _mu, _sigma = float(_env.demand_mean[_sku]) * _n_ret, float(_env.demand_std[_sku]) * _n_ret
                        _out_level = _mu * _lt + _z * _sigma * float(np.sqrt(_lt))
                        _ip = float(_env.inventory[agent_id][_sku]) - sum(_env.dc_retailer_backlog[agent_id][r_id][_sku] for r_id in _env.dc_assignments[agent_id]) + sum(o['qty'] for o in _env.pipeline[agent_id] if o['sku'] == _sku)
                        if _ip < _out_level:
                            _zero_action = False
                            break
                    if _zero_action: raw_action = np.zeros_like(raw_action)

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            obs, rewards, dones, infos = self.env.step([actions_env])

            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.n_agents):
                    h_cost = b_cost = o_cost = 0.0
                    is_dc = agent_id < 2
                    if is_dc:
                        for sku in range(3):
                            h_cost += env_state.inventory[agent_id][sku] * env_state.H_dc[agent_id][sku]
                            b_cost += sum(env_state.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env_state.dc_assignments[agent_id]) * env_state.B_dc[agent_id][sku]
                            if executed_actions[agent_id][sku] > 0:
                                o_cost += env_state.C_fixed_dc[agent_id][sku] + (pre_step_prices[sku] if pre_step_prices is not None else env_state.market_prices[sku]) * executed_actions[agent_id][sku]
                    else:
                        r_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(3):
                            h_cost += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                            b_cost += env_state.backlog[agent_id][sku] * env_state.B_retailer[r_idx][sku]
                            if executed_actions[agent_id][sku] > 0:
                                o_cost += env_state.C_fixed_retailer[r_idx][sku] + env_state.C_var_retailer[r_idx][assigned_dc][sku] * executed_actions[agent_id][sku]
                        for sku in range(3):
                            ep_data['_orders_placed'][agent_id] += env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            ep_data['_orders_from_stock'][agent_id] += env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)

                    ep_data['holding_costs'][agent_id] += h_cost
                    ep_data['backlog_costs'][agent_id] += b_cost
                    ep_data['ordering_costs'][agent_id] += o_cost

        return ep_data

    def mean_total_cost(self):
        costs = []
        for m in self.episode_metrics:
            costs.append(float(np.sum(m['holding_costs'])) + float(np.sum(m['backlog_costs'])) + float(np.sum(m['ordering_costs'])))
        return float(np.mean(costs)), float(np.std(costs))

    def mean_fill_rate(self):
        rates = []
        for m in self.episode_metrics:
            total_placed = sum(m['_orders_placed'][aid] for aid in range(2, self.n_agents))
            total_from_stock = sum(m['_orders_from_stock'][aid] for aid in range(2, self.n_agents))
            rates.append((total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0)
        return float(np.mean(rates))


# ============================================================================
#  SECTION 4 — Argument parsing & main orchestration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_model_dir', type=str, default='results/14Apr_gnn_kaggle_vari/run_seed_1/models')
    parser.add_argument('--happo_model_dir', type=str, default='results/01Apr_base/run_seed_1/models')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=90)
    parser.add_argument('--basestock_episode_length', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_path', type=str, default='configs/multi_dc_config.yaml')
    parser.add_argument('--num_agents', type=int, default=17)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--s_dc', type=float, default=100.0)
    parser.add_argument('--S_dc', type=float, default=170.0)
    parser.add_argument('--s_retailer', type=float, default=3.0)
    parser.add_argument('--S_retailer', type=float, default=10.0)
    parser.add_argument('--gnn_type', type=str, default='GAT')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--use_residual', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--critic_pooling', type=str, default='mean')
    parser.add_argument('--save_dir', type=str, default='evaluation_results/sensitivity_price_scenario')
    return parser.parse_args()


def plot_metrics(results, scenarios, save_dir):
    model_names = ['(s,S) BaseStock', 'HAPPO', 'GNN-HAPPO']
    colors = ['#4ECDC4', '#FF6B6B', '#3A86FF']
    n_scenarios = len(scenarios)
    n_models = len(model_names)

    labels = [sc['label'] for sc in scenarios]

    # Plot Total Cost
    means_cost = np.array([[results[s][m][0] for m in range(n_models)] for s in range(n_scenarios)])
    stds_cost  = np.array([[results[s][m][1] for m in range(n_models)] for s in range(n_scenarios)])
    x = np.arange(n_scenarios)
    width = 0.22
    offsets = np.array([-width, 0, width])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title('Total Cost across Price Scenarios', fontsize=14, fontweight='bold', pad=12)
    for i, (name, color, offset) in enumerate(zip(model_names, colors, offsets)):
        bars = ax.bar(x + offset, means_cost[:, i], width, label=name, color=color, alpha=0.88, edgecolor='black', linewidth=0.8, yerr=stds_cost[:, i], capsize=5)
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + stds_cost[bar_idx, i] + max(means_cost.max() * 0.005, 10), f'{h:,.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Price Scenario', fontsize=12)
    ax.set_ylabel('Mean Total Cost', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'price_scenario_total_cost.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Fill Rate
    fills = np.array([[results[s][m][2] for m in range(n_models)] for s in range(n_scenarios)])
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title('Fill Rate across Price Scenarios', fontsize=14, fontweight='bold', pad=12)
    for i, (name, color, offset) in enumerate(zip(model_names, colors, offsets)):
        bars = ax.bar(x + offset, fills[:, i], width, label=name, color=color, alpha=0.88, edgecolor='black', linewidth=0.8)
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.axhline(y=95, color='red', linestyle='--', linewidth=1.8, alpha=0.8, label='Target 95%')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('Price Scenario', fontsize=12)
    ax.set_ylabel('Mean Fill Rate (%)', fontsize=12)
    ax.set_ylim([0, 110])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'price_scenario_fill_rate.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_csv(results, scenarios, save_dir, model_names):
    out_path = Path(save_dir) / 'price_scenario_summary.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Model', 'Mean_Total_Cost', 'Std_Total_Cost', 'Mean_Fill_Rate_%'])
        for s_idx, sc in enumerate(scenarios):
            for m_idx, m_name in enumerate(model_names):
                mean_cost, std_cost, fill_rate = results[s_idx][m_idx]
                writer.writerow([sc['short'], m_name, round(mean_cost, 2), round(std_cost, 2), round(fill_rate, 2)])
    print(f'[OK] CSV saved to: {out_path}')


def print_summary(results, scenarios, model_names):
    print('\n' + '=' * 80)
    header = f"{'Scenario':<18} {'Model':<18} {'Mean Cost':>14} {'Std Cost':>12} {'Fill Rate':>10}"
    print(header)
    print('-' * 80)
    for s_idx, sc in enumerate(scenarios):
        for m_idx, m_name in enumerate(model_names):
            mean_cost, std_cost, fill_rate = results[s_idx][m_idx]
            print(f"{sc['short']:<18} {m_name:<18} {mean_cost:>14,.2f} {std_cost:>12,.2f} {fill_rate:>9.1f}%")
        print('-' * 80)
    print('=' * 80)


def main():
    args = parse_args()
    model_names = ['(s,S) BaseStock', 'HAPPO', 'GNN-HAPPO']
    scenarios = list(PRICE_SCENARIOS.values())

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = [] # [s_idx][m_idx] = (cost, std, fill)

    for sc in scenarios:
        sc_results = []
        print(f"\nEvaluating Scenario: {sc['label']}")

        # BaseStock
        import copy
        bs_args = copy.deepcopy(args)
        bs_args.episode_length = args.basestock_episode_length
        evaluator_bs = BaseStockEvaluator(bs_args, scenario=sc, config_path=args.config_path)
        evaluator_bs.evaluate()
        sc_results.append((*evaluator_bs.mean_total_cost(), evaluator_bs.mean_fill_rate()))

        # HAPPO
        happo_args = copy.deepcopy(args)
        happo_args.algorithm_name = 'happo'
        happo_args.model_dir = args.happo_model_dir
        evaluator_happo = HAPPOEvaluator(happo_args, scenario=sc)
        evaluator_happo.evaluate()
        sc_results.append((*evaluator_happo.mean_total_cost(), evaluator_happo.mean_fill_rate()))

        # GNN-HAPPO
        gnn_args = copy.deepcopy(args)
        gnn_args.model_dir = args.gnn_model_dir
        evaluator_gnn = GNNModelEvaluator(gnn_args, scenario=sc)
        evaluator_gnn.evaluate()
        sc_results.append((*evaluator_gnn.mean_total_cost(), evaluator_gnn.mean_fill_rate()))

        results.append(sc_results)

    print_summary(results, scenarios, model_names)
    save_csv(results, scenarios, save_dir, model_names)
    plot_metrics(results, scenarios, save_dir)
    print(f'\n[DONE] All results saved to: {save_dir}\n')


if __name__ == '__main__':
    main()
