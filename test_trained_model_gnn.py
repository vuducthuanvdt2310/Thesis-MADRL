#!/usr/bin/env python
"""
Test/Evaluation Script for Trained GNN-HAPPO Models
=====================================================

This script evaluates a trained GNN-HAPPO model on the multi-DC inventory
environment. It is DEDICATED to GNN models only — use test_trained_model.py
for the baseline (standard MLP-HAPPO) models.

Usage:
    python test_trained_model_gnn.py \
        --model_dir results/5Mar_1_gnn/run_seed_1/models \
        --episode_length 365 \
        --num_episodes 5 \
        --experiment_name "eval_gnn_365"
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
    parser = argparse.ArgumentParser(description='Evaluate Trained GNN-HAPPO Model')

    # Required
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory (e.g., results/.../models)')
    parser.add_argument('--config_path', type=str,
                        default='configs/multi_sku_config.yaml',
                        help='Path to environment config file (unused but kept for compatibility)')

    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100 for validation)')
    parser.add_argument('--episode_length', type=int, default=90,
                        help='Length of each episode in days')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

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
# GNN Evaluator
# ---------------------------------------------------------------------------

class GNNModelEvaluator:
    """Evaluates trained GNN-HAPPO models on the multi-DC inventory environment."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
        )

        # Output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = args.experiment_name if args.experiment_name else f'eval_gnn_{timestamp}'
        self.save_dir = Path(args.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._print_header()

        # 1. Create environment
        self.env = self._create_env()

        # Infer number of SKUs from underlying MultiDC env (for logging demand etc.)
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
    # Setup helpers
    # ------------------------------------------------------------------

    def _print_header(self):
        print('=' * 70)
        print('GNN-HAPPO Model Evaluation')
        print('=' * 70)
        print(f'Model directory : {self.args.model_dir}')
        print(f'Num episodes    : {self.args.num_episodes}')
        print(f'Episode length  : {self.args.episode_length} days')
        print(f'Results dir     : {self.save_dir}')
        print(f'Device          : {self.device}')
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

        # --- Detect GNN type ---
        # GCN layers store: gnn_base.layers.N.weight  /  .bias
        # GAT layers store: gnn_base.layers.N.W  /  .a
        if any('gnn_base.layers.0.weight' in k for k in keys):
            detected_gnn_type = 'GCN'
        elif any('gnn_base.layers.0.W' in k for k in keys):
            detected_gnn_type = 'GAT'
        else:
            detected_gnn_type = self.args.gnn_type  # fallback to user arg

        # --- Detect obs dim ---
        gcn_key = 'gnn_base.layers.0.weight'
        gat_key = 'gnn_base.layers.0.W'
        if gcn_key in sd:
            obs_dim = sd[gcn_key].shape[0]  # GCN weight: [in_features, out_features]
        elif gat_key in sd:
            obs_dim = sd[gat_key].shape[1]  # GAT W: [num_heads, in_features, head_dim]
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
        """Build full args namespace for constructing GNN_HAPPO_Policy."""
        # Use auto-detected gnn_type (from model keys) if available
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
        """Load GNN actor weights for all agents."""
        print('Loading GNN models...')
        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f'Model directory not found: {model_dir}')

        all_args = self._build_all_args()

        # Observation space padded to single_agent_obs_dim (what GNN was trained with)
        from gymnasium import spaces as gym_spaces
        padded_obs_space = gym_spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.single_agent_obs_dim,), dtype=np.float32
        )

        policies = []
        for agent_id in range(self.n_agents):
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]

            # Find best checkpoint for this agent
            best_file = self._find_best_model(model_dir, agent_id)

            # Build GNN policy
            policy = GNN_HAPPO_Policy(
                all_args,
                padded_obs_space,
                share_obs_space,
                act_space,
                n_agents=self.n_agents,
                agent_id=agent_id,
                device=self.device,
            )

            # Load actor weights
            state_dict = torch.load(str(best_file), map_location=self.device)
            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()

            policies.append(policy)
            print(f'  [OK] Agent {agent_id:2d} loaded  <- {best_file.name}')

        print(f'\n[OK] All {self.n_agents} GNN agent models loaded successfully!\n')
        return policies

    def _find_best_model(self, model_dir: Path, agent_id: int) -> Path:
        """Return the best-reward checkpoint file for a given agent."""
        all_files = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
        if not all_files:
            raise FileNotFoundError(
                f'No model file found for agent {agent_id} in {model_dir}'
            )

        # Prefer reward-suffixed files — pick the one with highest reward
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

        # Fall back: plain actor file or first available
        plain = model_dir / f'actor_agent{agent_id}.pt'
        return plain if plain.exists() else all_files[0]

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self):
        # ── Reproducibility ────────────────────────────────────────────────
        import torch as _torch
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        # ───────────────────────────────────────────────────────────────────

        print('=' * 70)
        print(f'Starting GNN Evaluation: {self.args.num_episodes} episode(s)  [seed={seed}]')
        print('=' * 70 + '\n')

        for ep in range(self.args.num_episodes):
            metrics = self._run_episode(ep, save_trajectory=(ep == 0))
            self.episode_metrics.append(metrics)

            avg_r = np.mean([m['total_reward'] for m in self.episode_metrics])
            print(f'Episode {ep + 1}/{self.args.num_episodes} '
                  f'| Total reward: {metrics["total_reward"]:>14.2f} '
                  f'| Running avg: {avg_r:>14.2f}')

        print('\n[OK] Evaluation complete!\n')

    def _run_episode(self, episode_num: int, save_trajectory: bool = False) -> dict:
        """Run one evaluation episode and return metrics."""
        obs, _ = self.env.reset()
        max_obs_dim = self.single_agent_obs_dim

        # RNN states: [1, n_agents, recurrent_N, hidden_size]
        rnn_states = np.zeros(
            (1, self.n_agents, 2, 128), dtype=np.float32
        )
        masks = np.ones((1, self.n_agents, 1), dtype=np.float32)

        # Metrics
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
            # Retailer order-count Fill Rate SL
            # An order = one (retailer, sku) demand event per step.
            # Fulfilled from on-hand = demand fully covered without adding any backlog.
            'service_level': [0.0] * self.n_agents,
            '_orders_placed':     [0] * self.n_agents,  # total demand events (count)
            '_orders_from_stock': [0] * self.n_agents,  # demand events fully met from on-hand
            # DC Cycle SL: (orders_received - orders_with_backlog) / orders_received
            'dc_cycle_service_level': {},   # {dc_id: float}
            'final_inventory': None,
            'final_backlog': None,
        }

        if save_trajectory:
            traj = {
                'inventory': [[] for _ in range(self.n_agents)],           # total inventory per agent
                'inventory_skus': [[] for _ in range(self.n_agents)],      # per-SKU inventory per agent
                'backlog': [[] for _ in range(self.n_agents)],
                'actions': [[] for _ in range(self.n_agents)],
                'rewards': [[] for _ in range(self.n_agents)],
                'demand': [[] for _ in range(self.n_agents)],
                # Normalized (same scales as env obs): demand, inventory, order qty for retailers
                'norm_demand': [[] for _ in range(self.n_agents)],
                'norm_inventory': [[] for _ in range(self.n_agents)],
                'norm_order': [[] for _ in range(self.n_agents)],
                # SL: order-count fill rate (0/1 per SKU per step)
                'orders_placed': [[] for _ in range(self.n_agents)],
                'orders_from_stock': [[] for _ in range(self.n_agents)],
                'norm_scales': None,  # set once from env (demand_cap, inv_scale, order range)
            }

        for step in range(self.args.episode_length):
            # Capture market prices BEFORE step so DC ordering cost uses the
            # same price that _calculate_rewards() uses inside step().
            _pre_env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = _pre_env_list[0].market_prices.copy() if _pre_env_list else None

            # Build padded structured obs: [1, n_agents, max_obs_dim]
            obs_structured = np.zeros(
                (1, self.n_agents, max_obs_dim), dtype=np.float32
            )
            for aid in range(self.n_agents):
                raw = np.stack(obs[:, aid])  # [1, obs_dim_i]
                d = raw.shape[1]
                obs_structured[0, aid, :d] = raw[0]

            actions_env = []
            raw_actions = {}
            for agent_id in range(self.n_agents):
                with torch.no_grad():
                    action, rnn_state = self.policies[agent_id].act(
                        obs_structured,           # [1, n_agents, max_obs_dim]
                        self.adj_tensor,
                        agent_id,
                        rnn_states[:, agent_id],  # [1, recurrent_N, hidden]
                        masks[:, agent_id],        # [1, 1]
                        deterministic=False,  # Sample from distribution → step-varying actions
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

                # ── Heuristic disabled: let the DRL policy decide ────────────
                # The inventory-conditioned rescaling below is commented out.
                # With the new obs-dependent DiagGaussian (Method 2), the retrained
                # policy learns its own mean and std per observation — no heuristic
                # override is needed. Re-enable only for ablation comparison.
                #
                # RETAILER_TARGET_INV = 20.0
                # DC_TARGET_INV       = 500.0
                # current_obs = obs_structured[0, agent_id]
                # if agent_id < 2:
                #     inv_indices     = [0, 9, 18]
                #     inv_norm_factor = 1000.0
                #     target_inv      = DC_TARGET_INV
                # else:
                #     inv_indices     = [0, 7, 14]
                #     inv_norm_factor = 150.0
                #     target_inv      = RETAILER_TARGET_INV
                # inventory_weight = np.zeros(self.n_skus, dtype=np.float32)
                # for sku_idx, obs_idx in enumerate(inv_indices):
                #     actual_inv = float(current_obs[obs_idx]) * inv_norm_factor
                #     inventory_weight[sku_idx] = max(0.0, 1.0 - actual_inv / target_inv)
                # raw_action = raw_action * inventory_weight
                # ─────────────────────────────────────────────────────────────

                # Pass actions to the environment (it will clip them internally too).
                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            # Step environment
            obs, rewards, dones, infos = self.env.step([actions_env])

            # Collect metrics from env state
            env_list = getattr(self.env, 'env_list',
                               getattr(self.env, 'envs', None))
            if env_list:
                env_state = env_list[0]

                # Recompute the executed (clipped) actions so logging matches
                # the exact quantities the environment used internally.
                executed_actions = env_state._clip_actions(raw_actions)

                for agent_id in range(self.n_agents):
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward
                    ep_data['agent_rewards'][agent_id] += reward
                    ep_data['agent_costs'][agent_id] += cost
                    ep_data['total_reward'] += reward
                    ep_data['total_cost'] += cost

                    # Cost breakdown
                    h_cost = b_cost = o_cost = 0.0
                    is_dc = agent_id < 2
                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(3):
                            h_cost += (env_state.inventory[agent_id][sku]
                                       * env_state.H_dc[dc_idx][sku])
                            # DC's flat backlog[dc_id] is always 0 after per-retailer migration;
                            # use the sum of dc_retailer_backlog across assigned retailers instead.
                            dc_owed_sku = sum(
                                env_state.dc_retailer_backlog[agent_id][r_id][sku]
                                for r_id in env_state.dc_assignments[agent_id]
                            )
                            b_cost += dc_owed_sku * env_state.B_dc[dc_idx][sku]
                            if executed_actions[agent_id][sku] > 0:
                                # Use PRE-STEP price (matches _calculate_rewards inside step())
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
                            # Retailers use action[0:3] from their ASSIGNED DC only.
                            # action[3:6] is UNUSED (always zero — uniform 6D action space).
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                o_cost += (env_state.C_fixed_retailer[r_idx][sku]
                                           + env_state.C_var_retailer[r_idx][assigned_dc][sku] * order_qty)

                        # --- Retailer Order-Count Fill Rate ---
                        # An order = one (retailer, sku) demand event per step.
                        # Fulfilled from on-hand: demand[sku] covered WITHOUT shortage (no backlog added).
                        # Read directly from env step_orders_placed / step_orders_from_stock.
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
                        # DC: aggregate per-retailer backlog
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
                            # DC: actual demand is the sum of orders placed by its assigned retailers
                            demand_vec = np.zeros(self.n_skus, dtype=float)
                            for r_id in env_state.dc_assignments[agent_id]:
                                op = env_state.step_orders_placed.get(r_id, {})
                                for s in range(self.n_skus):
                                    demand_vec[s] += op.get(s, 0.0)
                        else:
                            # Retailer: log per-SKU customer demand for this step
                            demand_vec = np.array(
                                env_state.step_demand.get(agent_id, np.zeros(self.n_skus, dtype=float)),
                                dtype=float,
                            )
                        traj['demand'][agent_id].append(demand_vec.copy())

                        # Normalized demand, inventory, order (same scales as env obs; retailers only)
                        # Demand: cap = mean + 3*std per SKU (from _get_retailer_observation)
                        # Inventory: retailer own inv / 150.0
                        # Order: (qty - 20) / 50 for retailer [20, 70] -> [0, 1]
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

                        # Store normalization scales once (from env; same as used in obs)
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

            # Read DC Cycle SL from infos (populated by env every step)
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
                # Retailer: order-count Fill Rate
                # = orders fully met from on-hand / total orders placed (as %)
                placed     = ep_data['_orders_placed'][agent_id]
                from_stock = ep_data['_orders_from_stock'][agent_id]
                ep_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:
                # DC: use Cycle SL from env (filled last step of episode)
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
        self._save_step_trajectory_excel()
        self._print_summary(stats)
        print(f'\n[OK] Report saved to: {self.save_dir}\n')

    def _calculate_statistics(self) -> dict:
        def arr(key):
            return [m[key] for m in self.episode_metrics]

        stats = {
            'num_episodes': len(self.episode_metrics),
            'episode_length': self.args.episode_length,
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
            # DC Cycle SL aggregated over episodes
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
                # Retailer: pooled order-count fill rate across ALL episodes
                # = total orders_from_stock / total orders_placed  (not average of per-ep %)
                # Pooling is more accurate when episode SL variance is high.
                total_placed     = sum(m['_orders_placed'][aid]     for m in self.episode_metrics)
                total_from_stock = sum(m['_orders_from_stock'][aid] for m in self.episode_metrics)
                pooled_sl = (total_from_stock / total_placed * 100.0) if total_placed > 0 else 100.0
            else:
                # DC: mean of per-episode cycle SL (already a cumulative episode value)
                pooled_sl = float(np.mean([m['service_level'][aid] for m in self.episode_metrics]))

            stats['per_agent'][label] = {
                'avg_reward': float(np.mean([m['agent_rewards'][aid]
                                             for m in self.episode_metrics])),
                'avg_cost': float(np.mean([m['agent_costs'][aid]
                                           for m in self.episode_metrics])),
                'avg_holding_cost': float(np.mean([m['holding_costs'][aid]
                                                   for m in self.episode_metrics])),
                'avg_backlog_cost': float(np.mean([m['backlog_costs'][aid]
                                                   for m in self.episode_metrics])),
                'avg_ordering_cost': float(np.mean([m['ordering_costs'][aid]
                                                    for m in self.episode_metrics])),
                'avg_inventory': float(np.mean([m['avg_inventory'][aid]
                                                for m in self.episode_metrics])),
                'avg_backlog': float(np.mean([m['avg_backlog'][aid]
                                              for m in self.episode_metrics])),
                'service_level': pooled_sl,
            }

        # System-wide retailer fill rate (pooled across all retailer agents and episodes)
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

    def _save_step_trajectory_excel(self):
        """
        Save one Excel with two sheets:
          - Data: step, agent, state (inv, backlog), demand, action, normalized values, SL (step + cum).
          - Scales: normalization constants (demand_cap, inv_scale, order range) for train/test check.
        SL = order-count fill rate: cum_sl_pct = 100 * cum_orders_from_stock / cum_orders_placed (retailers).
        """
        if not self.detailed_trajectory:
            return

        traj = self.detailed_trajectory
        num_steps = len(traj['inventory'][0])
        cum_orders_placed = {aid: 0 for aid in range(self.n_agents)}
        cum_orders_from_stock = {aid: 0 for aid in range(self.n_agents)}

        rows = []
        for step in range(num_steps):
            for aid in range(self.n_agents):
                inv = traj['inventory'][aid][step]
                bl = traj['backlog'][aid][step]
                action_vec = np.array(traj['actions'][aid][step], dtype=float).flatten()
                demand_vec = np.array(traj['demand'][aid][step], dtype=float).flatten()
                norm_d = np.array(traj['norm_demand'][aid][step], dtype=float).flatten()
                norm_inv = np.array(traj['norm_inventory'][aid][step], dtype=float).flatten()
                norm_ord = np.array(traj['norm_order'][aid][step], dtype=float).flatten()
                op_vec = np.array(traj['orders_placed'][aid][step], dtype=float).flatten()
                ofs_vec = np.array(traj['orders_from_stock'][aid][step], dtype=float).flatten()

                step_placed = int(np.sum(op_vec))
                step_from_stock = int(np.sum(ofs_vec))
                cum_orders_placed[aid] += step_placed
                cum_orders_from_stock[aid] += step_from_stock
                cum_placed = cum_orders_placed[aid]
                cum_from_stock = cum_orders_from_stock[aid]

                row = {
                    'step': step + 1,
                    'agent_id': aid,
                    'agent': f'{"DC" if aid < 2 else "R"}_{aid}',
                    'inv': inv,
                    'backlog': bl,
                    'reward': traj['rewards'][aid][step],
                }
                for i in range(self.n_skus):
                    row[f'demand_{i}'] = demand_vec[i] if i < len(demand_vec) else 0
                    row[f'order_{i}'] = action_vec[i] if i < len(action_vec) else 0
                    row[f'norm_d_{i}'] = round(norm_d[i], 4) if i < len(norm_d) else 0
                    row[f'norm_inv_{i}'] = round(norm_inv[i], 4) if i < len(norm_inv) else 0
                    row[f'norm_ord_{i}'] = round(norm_ord[i], 4) if i < len(norm_ord) else 0
                row['orders_placed'] = step_placed
                row['orders_from_stock'] = step_from_stock
                row['step_sl_pct'] = round(step_from_stock / step_placed * 100, 1) if step_placed > 0 else 100
                row['cum_placed'] = cum_placed
                row['cum_from_stock'] = cum_from_stock
                row['cum_sl_pct'] = round(cum_from_stock / cum_placed * 100, 1) if cum_placed > 0 else 100
                rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.save_dir / 'step_trajectory_ep1.xlsx'
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            scales = traj.get('norm_scales')
            if scales:
                # One row: key scales only (demand_cap = norm divisor, inv/backlog/order range)
                df_scales = pd.DataFrame([{
                    'demand_cap_0': scales.get('demand_cap_0'),
                    'demand_cap_1': scales.get('demand_cap_1'),
                    'demand_cap_2': scales.get('demand_cap_2'),
                    'inv_scale': scales.get('inv_scale_retailer'),
                    'backlog_scale': scales.get('backlog_scale_retailer'),
                    'order_min': scales.get('order_min_retailer'),
                    'order_max': scales.get('order_max_retailer'),
                }])
                df_scales.to_excel(writer, sheet_name='Scales', index=False)
            else:
                pd.DataFrame([{'note': 'Scales not captured'}]).to_excel(writer, sheet_name='Scales', index=False)
        print(f'[OK] Saved step_trajectory_ep1.xlsx (Data + Scales)')

    def _save_metrics_json(self, stats):
        path = self.save_dir / 'evaluation_metrics.json'
        output = {
            'metadata': {
                'model_dir': str(self.args.model_dir),
                'algorithm': 'GNN-HAPPO',
                'num_episodes': self.args.num_episodes,
                'episode_length': self.args.episode_length,
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
        # Primary output: standardised validation file
        results_path = self.save_dir / 'results_gnn_happo.csv'
        # Also keep the generic name for backward compat
        compat_path  = self.save_dir / 'episode_metrics.csv'
        for path in (results_path, compat_path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                                 'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])
                for ep_num, m in enumerate(self.episode_metrics):
                    # Fill_Rate: average of per-agent service_level (original method)
                    fill_rate = float(np.mean(m['service_level']))
                    # Lost_Sales: total unfulfilled demand units pooled across all retailer agents
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
        self._plot_episode_rewards()
        self._plot_cost_breakdown(stats)
        self._plot_service_levels(stats)
        if self.detailed_trajectory:
            self._plot_trajectory()
            self._plot_normalized_trajectory()
            self._plot_dc_inventory_fluctuation()
            self._plot_retailer_inventory_fluctuation()
        self._plot_performance_distribution()
        print('[OK] All visualizations created')

    def _plot_episode_rewards(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        episodes = range(1, len(self.episode_metrics) + 1)
        rewards = [m['total_reward'] for m in self.episode_metrics]
        ax.plot(episodes, rewards, linewidth=2, alpha=0.7, label='Episode Reward')
        window = min(10, len(rewards))
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), roll,
                    linewidth=2.5, color='red',
                    label=f'{window}-Ep Moving Avg')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('GNN-HAPPO Performance Across Episodes',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'episode_rewards.png', dpi=300)
        plt.close()

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
        ax.set_title('GNN-HAPPO Cost Breakdown by Agent',
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
        ax.set_title('Service Level by Agent (Demand Fill-Rate for Retailers)',
                     fontsize=14, fontweight='bold')
        ax.set_ylim([0, 110])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()

    def _plot_trajectory(self):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        days = range(1, self.args.episode_length + 1)
        for aid in range(self.n_agents):
            label = f'{"DC" if aid < 2 else "R"}_{aid}'
            axes[0].plot(days, self.detailed_trajectory['inventory'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
            axes[1].plot(days, self.detailed_trajectory['backlog'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
            axes[2].plot(days, self.detailed_trajectory['rewards'][aid],
                         label=label, linewidth=1.2, alpha=0.75)
        for i, (ax, title, ylabel) in enumerate(zip(
            axes,
            ['Inventory Trajectory (Ep 1)', 'Backlog Trajectory (Ep 1)', 'Reward Trajectory (Ep 1)'],
            ['Total Inventory', 'Total Backlog', 'Reward']
        )):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(loc='upper right', ncol=3, fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[2].set_xlabel('Day', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'detailed_trajectory.png', dpi=300)
        plt.close()

    def _plot_normalized_trajectory(self):
        """Plot normalized demand, normalized inventory, normalized order quantity for retailers (Ep 1)."""
        traj = self.detailed_trajectory
        if not traj or 'norm_demand' not in traj:
            return
        days = range(1, self.args.episode_length + 1)
        # Plot first 5 retailers to avoid clutter (or all if n_agents is small)
        retailer_ids = [i for i in range(self.n_agents) if i >= 2]
        n_show = min(5, len(retailer_ids))
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for i, aid in enumerate(retailer_ids[:n_show]):
            label = f'R_{aid}'
            # Per-SKU average for this retailer (or sum; here we use mean across SKUs for one line)
            norm_d = np.array(traj['norm_demand'][aid], dtype=float)   # [T, n_skus]
            norm_inv = np.array(traj['norm_inventory'][aid], dtype=float)
            norm_ord = np.array(traj['norm_order'][aid], dtype=float)
            axes[0].plot(days, norm_d.mean(axis=1), label=label, linewidth=1.2, alpha=0.85)
            axes[1].plot(days, norm_inv.mean(axis=1), label=label, linewidth=1.2, alpha=0.85)
            axes[2].plot(days, norm_ord.mean(axis=1), label=label, linewidth=1.2, alpha=0.85)
        for ax, title, ylabel in zip(
            axes,
            ['Normalized Demand (Ep 1, mean over SKUs)',
             'Normalized Inventory (Ep 1, mean over SKUs)',
             'Normalized Order Qty (Ep 1, mean over SKUs)'],
            ['Demand / (mean+3*std)', 'Inventory / 150', 'Order: (qty-20)/50']
        ):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.15)
        axes[2].set_xlabel('Day', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'normalized_trajectory.png', dpi=300)
        plt.close()
        print('[OK] Saved normalized trajectory plot: normalized_trajectory.png')

    def _plot_dc_inventory_fluctuation(self):
        """Plot inventory fluctuation over steps for each DC (episode 1).

        Layout: 2 rows x (n_skus+1) columns.
          - Row per DC.
          - First n_skus sub-plots: per-SKU inventory.
          - Last sub-plot: total inventory across all SKUs.
        """
        traj = self.detailed_trajectory
        days = np.arange(1, self.args.episode_length + 1)
        dc_ids = [i for i in range(self.n_agents) if i < 2]
        n_dcs = len(dc_ids)
        n_cols = self.n_skus + 1  # per-SKU cols + total col

        fig, axes = plt.subplots(
            n_dcs, n_cols,
            figsize=(5 * n_cols, 4 * n_dcs),
            squeeze=False,
        )
        fig.suptitle('DC Inventory Fluctuation Over Steps (Episode 1)',
                     fontsize=15, fontweight='bold', y=1.01)

        sku_colors = ['#2196F3', '#4CAF50', '#FF9800']  # one colour per SKU
        total_color = '#9C27B0'

        for row, dc_id in enumerate(dc_ids):
            dc_label = f'DC_{dc_id}'
            inv_skus = np.array(traj['inventory_skus'][dc_id], dtype=float)  # [T, n_skus]
            inv_total = np.array(traj['inventory'][dc_id], dtype=float)       # [T]

            # Per-SKU sub-plots
            for sku in range(self.n_skus):
                ax = axes[row][sku]
                ax.plot(days, inv_skus[:, sku],
                        color=sku_colors[sku % len(sku_colors)],
                        linewidth=1.5, alpha=0.85)
                ax.fill_between(days, inv_skus[:, sku], alpha=0.15,
                                color=sku_colors[sku % len(sku_colors)])
                ax.set_title(f'{dc_label} — SKU {sku}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Inventory', fontsize=10)
                ax.set_xlabel('Step (day)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(days[0], days[-1])

            # Total inventory sub-plot (last column)
            ax_total = axes[row][n_cols - 1]
            ax_total.plot(days, inv_total,
                          color=total_color, linewidth=2.0, alpha=0.9,
                          label='Total')
            ax_total.fill_between(days, inv_total, alpha=0.12, color=total_color)
            ax_total.axhline(np.mean(inv_total), color='red', linestyle='--',
                             linewidth=1.2, label=f'Mean: {np.mean(inv_total):.0f}')
            ax_total.set_title(f'{dc_label} — Total Inventory', fontsize=11, fontweight='bold')
            ax_total.set_ylabel('Total Inventory', fontsize=10)
            ax_total.set_xlabel('Step (day)', fontsize=10)
            ax_total.legend(fontsize=9)
            ax_total.grid(True, alpha=0.3)
            ax_total.set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'dc_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[OK] Saved DC inventory fluctuation plot: {out.name}')

    def _plot_retailer_inventory_fluctuation(self):
        """Plot inventory fluctuation over steps for all retailers (episode 1).

        One figure with two sub-plots:
          - Top: per-retailer total inventory (all on same axes — useful for comparison).
          - Bottom: system-wide average retailer inventory with std band.
        """
        traj = self.detailed_trajectory
        days = np.arange(1, self.args.episode_length + 1)
        retailer_ids = [i for i in range(self.n_agents) if i >= 2]
        n_retailers = len(retailer_ids)

        # Gather total inventory arrays per retailer
        inv_matrix = np.array(
            [traj['inventory'][rid] for rid in retailer_ids], dtype=float
        )  # [n_retailers, T]

        # ---- Figure 1: individual retailer lines ----
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.suptitle('Retailer Inventory Fluctuation Over Steps (Episode 1)',
                     fontsize=15, fontweight='bold')

        cmap = plt.cm.get_cmap('tab20', n_retailers)
        ax_lines = axes[0]
        for i, rid in enumerate(retailer_ids):
            ax_lines.plot(days, inv_matrix[i],
                          color=cmap(i), linewidth=1.2, alpha=0.75,
                          label=f'R_{rid}')
        ax_lines.set_title('Individual Retailer Inventory', fontsize=12, fontweight='bold')
        ax_lines.set_ylabel('Total Inventory', fontsize=11)
        ax_lines.legend(
            loc='upper right',
            ncol=max(1, n_retailers // 5),
            fontsize=7,
            framealpha=0.7,
        )
        ax_lines.grid(True, alpha=0.3)

        # ---- Bottom: system-wide mean ± std ----
        ax_avg = axes[1]
        mean_inv = inv_matrix.mean(axis=0)   # [T]
        std_inv  = inv_matrix.std(axis=0)    # [T]
        ax_avg.plot(days, mean_inv, color='#1565C0', linewidth=2.0,
                    label='Mean (all retailers)')
        ax_avg.fill_between(
            days,
            np.maximum(mean_inv - std_inv, 0),
            mean_inv + std_inv,
            alpha=0.2, color='#1565C0', label='±1 std'
        )
        ax_avg.axhline(mean_inv.mean(), color='red', linestyle='--',
                       linewidth=1.2,
                       label=f'Time-avg: {mean_inv.mean():.0f}')
        ax_avg.set_title('System-Wide Avg Retailer Inventory', fontsize=12, fontweight='bold')
        ax_avg.set_ylabel('Avg Total Inventory', fontsize=11)
        ax_avg.set_xlabel('Step (day)', fontsize=11)
        ax_avg.legend(fontsize=9)
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'retailer_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'[OK] Saved retailer inventory fluctuation plot: {out.name}')

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
        ax.set_title('GNN-HAPPO Episode Reward Distribution',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_distribution.png', dpi=300)
        plt.close()

    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('GNN-HAPPO Evaluation Summary')
        print('=' * 70)
        print(f"Episodes      : {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print(f"Avg reward    : {stats['total_reward']['mean']:>14.2f} "
              f"(+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost      : {stats['total_cost']['mean']:>14.2f} "
              f"(+/-{stats['total_cost']['std']:.2f})")
        print(f"Best reward   : {stats['total_reward']['max']:>14.2f}")
        print(f"Worst reward  : {stats['total_reward']['min']:>14.2f}")
        # System-wide KPI: pooled fill rate across all retailers and all episodes
        sys_fr = stats.get('system_retailer_fill_rate', 0.0)
        print(f"Retailer Fill Rate (system): {sys_fr:>6.1f}%   <- KPI")
        # DC Cycle SL
        dc_csl = stats.get('dc_cycle_service_level', {})
        for dc_id, csl_val in dc_csl.items():
            print(f"DC_{dc_id} Cycle SL : {csl_val:>6.1f}%")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Reward':>12} {'Avg Cost':>12} "
              f"{'Holding':>12} {'Backlog':>12} {'Svc%':>8} {'SL Type':>12}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            is_dc  = agent.startswith('DC')
            sl_type = 'Cycle SL' if is_dc else 'Fill-Rate'
            print(f"{agent:<14} {data['avg_reward']:>12.1f} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}% {sl_type:>12}")
        print('=' * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    evaluator = GNNModelEvaluator(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
