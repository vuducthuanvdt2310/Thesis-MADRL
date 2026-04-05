#!/usr/bin/env python
"""
Test/Evaluation Script for Trained Multi-Agent RL Models
=========================================================

This script evaluates a trained MADRL model on the multi-DC inventory environment.
It generates comprehensive metrics and visualizations to demonstrate that MADRL
can solve the inventory optimization problem.

Usage:
    python test_trained_model.py --model_dir results/experiment_name/run_seed_1/models \
                                  --num_episodes 50 \
                                  --save_dir evaluation_results
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import json
import pandas as pd
from itertools import chain

from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from algorithms.happo_policy import HAPPO_Policy


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


def parse_test_args():
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(description='Test Trained MADRL Model')
    
    # Model paths
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to saved model directory (e.g., results/full_training/run_seed_1/models)')
    parser.add_argument('--config_path', type=str, default='configs/multi_sku_config.yaml',
                       help='Path to environment config file')
    
    # Testing parameters
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes to run (default: 100 for validation)')
    parser.add_argument('--episode_length', type=int, default=90,
                       help='Length of each episode (days)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this evaluation run (default: timestamp)')
    
    # Environment settings (must match training config)
    parser.add_argument('--num_agents', type=int, default=5,
                       help='Number of agents (2 DCs + 3 Retailers)')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--algorithm_name', type=str, default='happo',
                       choices=['happo', 'gnn_happo'],
                       help='Algorithm used during training (happo=baseline, gnn_happo=proposed)')

    args = parser.parse_args()
    return args


class ModelEvaluator:
    """Evaluates trained MADRL models and generates comprehensive metrics."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.experiment_name if args.experiment_name else f"eval_{timestamp}"
        self.save_dir = Path(args.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"MADRL Model Evaluation")
        print(f"{'='*70}")
        print(f"Model directory: {args.model_dir}")
        print(f"Num episodes: {args.num_episodes}")
        print(f"Episode length: {args.episode_length} days")
        print(f"Results will be saved to: {self.save_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Create environment
        self.env = self._create_env()

        # Build adjacency matrix for GNN (if needed)
        self.adj_tensor = None
        if getattr(args, 'algorithm_name', 'happo') == 'gnn_happo':
            from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
            print("Building supply chain graph for GNN evaluation...")
            adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True)
            adj = normalize_adjacency(adj, method='symmetric')
            self.adj_tensor = torch.FloatTensor(adj).to(self.device)
            print(f"✓ Graph created: {adj.shape[0]} nodes, {np.sum(adj > 0)} edges\n")

        # Load trained models
        self.policies = self._load_models()

        # Metrics storage
        self.episode_metrics = []
        self.detailed_trajectory = None
        
    def _create_env(self):
        """Create evaluation environment."""
        print("Creating evaluation environment...")
        
        # Get config
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
        
        # Sync num_agents with environment (config might override args)
        if hasattr(env, 'num_agent'):
            self.args.num_agents = env.num_agent
        
        print(f"✓ Environment created: {env.num_envs} environment(s)")
        print(f"  - Agents: {self.args.num_agents} (2 DCs + {self.args.num_agents - 2} Retailers)")
        print(f"  - Observation spaces: DCs=27D, Retailers=42D")
        print(f"  - Action spaces: All agents=6D continuous\n")
        
        return env
    
    def _load_models(self):
        """Load trained model weights for all agents."""
        print("Loading trained models...")

        algorithm_name = getattr(self.args, 'algorithm_name', 'happo')
        is_gnn = (algorithm_name == 'gnn_happo')

        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        policies = []

        # Get environment config
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
            # Add GNN-specific args so parser doesn't fail
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

        # Max obs dim for GNN padding
        max_obs_dim = max([self.env.observation_space[i].shape[0] for i in range(self.args.num_agents)])

        for agent_id in range(self.args.num_agents):
            # Get observation and action spaces
            obs_space = self.env.observation_space[agent_id]
            share_obs_space = self.env.share_observation_space[agent_id]
            act_space = self.env.action_space[agent_id]

            # Find the best model file for this agent
            agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))

            if not agent_files:
                raise FileNotFoundError(f"No model found for agent {agent_id} in {model_dir}")

            best_file = None
            best_reward_val = -float('inf')

            suffixed_files = []
            for f in agent_files:
                if f.name == f"actor_agent{agent_id}.pt":
                    continue
                try:
                    parts = f.name.split('_reward_')
                    if len(parts) == 2:
                        reward_str = parts[1].replace('.pt', '')
                        reward_val = float(reward_str)
                        suffixed_files.append((reward_val, f))
                except ValueError:
                    continue

            if suffixed_files:
                suffixed_files.sort(key=lambda x: x[0], reverse=True)
                best_reward_val, best_file = suffixed_files[0]
                print(f"  Agent {agent_id}: Found best model with reward {best_reward_val:.2f}")
            else:
                simple_path = model_dir / f"actor_agent{agent_id}.pt"
                if simple_path.exists():
                    best_file = simple_path
                    print(f"  Agent {agent_id}: Using standard model file")
                else:
                    best_file = agent_files[0]
                    print(f"  Agent {agent_id}: Using first available file (unknown naming pattern)")

            print(f"  Loading: {best_file.name}")

            # Load state dict
            state_dict = torch.load(str(best_file), map_location=self.device)

            if is_gnn:
                # GNN policy: obs_space is the padded max_obs_dim
                from gymnasium import spaces as gym_spaces
                padded_obs_space = gym_spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_obs_dim,), dtype=np.float32
                )
                from algorithms.gnn_happo_policy import GNN_HAPPO_Policy
                policy = GNN_HAPPO_Policy(
                    all_args,
                    padded_obs_space,
                    share_obs_space,
                    act_space,
                    n_agents=self.args.num_agents,
                    agent_id=agent_id,
                    device=self.device
                )
            else:
                # Baseline HAPPO policy
                # Check input dimension from first layer weights
                saved_input_dim = None
                if 'base.mlp.fc1.0.weight' in state_dict:
                    saved_input_dim = state_dict['base.mlp.fc1.0.weight'].shape[1]
                elif 'base.cnn.cnn.0.weight' in state_dict:
                    saved_input_dim = state_dict['base.cnn.cnn.0.weight'].shape[1]

                current_obs_dim = obs_space.shape[0]
                if saved_input_dim is not None and saved_input_dim != current_obs_dim:
                    print(f"  DIMENSION MISMATCH: Model expects {saved_input_dim}, Env provides {current_obs_dim}")
                    print(f"  -> Adjusting policy input dimension to {saved_input_dim}")
                    from gymnasium import spaces as gym_spaces
                    obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(saved_input_dim,), dtype=np.float32)

                policy = HAPPO_Policy(
                    all_args,
                    obs_space,
                    share_obs_space,
                    act_space,
                    device=self.device
                )

            # Load weights
            policy.actor.load_state_dict(state_dict)
            policy.actor.eval()

            policies.append(policy)
            print(f"✓ Loaded agent {agent_id} successfully")

        print(f"\n✓ All {self.args.num_agents} agent models loaded successfully!\n")
        return policies
    
    def evaluate(self):
        """Run evaluation episodes and collect metrics."""
        # ── Reproducibility ────────────────────────────────────────────────
        seed = getattr(self.args, 'seed', 42)
        np.random.seed(seed)
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        # ───────────────────────────────────────────────────────────────────

        print(f"{'='*70}")
        print(f"Starting Evaluation: {self.args.num_episodes} episodes  [seed={seed}]")
        print(f"{'='*70}\n")
        
        for episode in range(self.args.num_episodes):
            metrics = self._run_episode(episode, save_trajectory=(episode == 0))
            self.episode_metrics.append(metrics)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([m['total_reward'] for m in self.episode_metrics[-10:]])
                avg_cost = np.mean([m['total_cost'] for m in self.episode_metrics[-10:]])
                print(f"Episode {episode+1}/{self.args.num_episodes} - "
                      f"Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Avg Cost (last 10): {avg_cost:.2f}")
        
        print(f"\n{'='*70}")
        print(f"Evaluation Complete!")
        print(f"{'='*70}\n")
    
    def _run_episode(self, episode_num, save_trajectory=False):
        """Run a single evaluation episode."""
        obs, _ = self.env.reset()

        is_gnn = (getattr(self.args, 'algorithm_name', 'happo') == 'gnn_happo')

        # Initialize RNN states
        rnn_states = np.zeros((1, self.args.num_agents, 2, 128), dtype=np.float32)
        masks = np.ones((1, self.args.num_agents, 1), dtype=np.float32)

        # For GNN: build structured obs [1, n_agents, max_obs_dim]
        if is_gnn:
            max_obs_dim = max([self.env.observation_space[i].shape[0] for i in range(self.args.num_agents)])

        def _build_gnn_obs(obs_array):
            """Build structured obs [1, n_agents, max_obs_dim] from env obs."""
            structured = np.zeros((1, self.args.num_agents, max_obs_dim), dtype=np.float32)
            for aid in range(self.args.num_agents):
                agent_obs = np.stack(obs_array[:, aid])  # [1, obs_dim]
                obs_dim = agent_obs.shape[1]
                structured[0, aid, :obs_dim] = agent_obs[0]
            return structured

        # Metrics for this episode
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
            # Service level: order-count fill-rate (retailers): orders_from_stock / orders_placed
            'service_level': [0] * self.args.num_agents,
            '_demand_total': [0.0] * self.args.num_agents,   # cumulative demand (retailers only)
            '_demand_met':   [0.0] * self.args.num_agents,   # cumulative demand fulfilled
            # Order-count fill-rate accumulators (mirrors test_trained_model_gnn.py)
            '_orders_placed':     [0] * self.args.num_agents,
            '_orders_from_stock': [0] * self.args.num_agents,
        }

        # Detect n_skus from env
        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        # Trajectory data (only for first episode)
        if save_trajectory:
            trajectory = {
                'inventory': [[] for _ in range(self.args.num_agents)],
                'inventory_skus': [[] for _ in range(self.args.num_agents)],   # per-SKU
                'backlog': [[] for _ in range(self.args.num_agents)],
                'actions': [[] for _ in range(self.args.num_agents)],
                'rewards': [[] for _ in range(self.args.num_agents)],
                'demand': [[] for _ in range(self.args.num_agents)],
                'norm_demand': [[] for _ in range(self.args.num_agents)],
                'norm_inventory': [[] for _ in range(self.args.num_agents)],
                'norm_order': [[] for _ in range(self.args.num_agents)],
                'orders_placed': [[] for _ in range(self.args.num_agents)],
                'orders_from_stock': [[] for _ in range(self.args.num_agents)],
                'norm_scales': None,
            }

        # Run episode
        for step in range(self.args.episode_length):
            # Capture market prices BEFORE step so DC ordering cost uses the
            # same price that _calculate_rewards() uses inside step().
            env_list_pre = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            pre_step_prices = None
            if env_list_pre:
                pre_step_prices = env_list_pre[0].market_prices.copy()

            # Get actions from all agents
            actions_env = []
            raw_actions = {}  # {agent_id: raw_action_array} for clip-correction after step

            if is_gnn:
                # Build structured observations once per step for all agents
                obs_structured = _build_gnn_obs(obs)

            for agent_id in range(self.args.num_agents):
                self.policies[agent_id].actor.eval()

                with torch.no_grad():
                    if is_gnn:
                        # GNN policy: pass structured obs + adjacency matrix + agent_id
                        action, rnn_state = self.policies[agent_id].act(
                            obs_structured,           # [1, n_agents, max_obs_dim]
                            self.adj_tensor,          # adjacency matrix
                            agent_id,                 # which agent
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            None,
                            deterministic=True
                        )
                    else:
                        # Baseline HAPPO: pass individual agent obs
                        obs_agent = np.stack(obs[:, agent_id])

                        # Pad if needed (e.g. model trained with larger obs dim)
                        policy_input_dim = self.policies[agent_id].obs_space.shape[0]
                        current_obs_dim = obs_agent.shape[1]
                        if current_obs_dim < policy_input_dim:
                            diff = policy_input_dim - current_obs_dim
                            padding = np.zeros((obs_agent.shape[0], diff), dtype=np.float32)
                            obs_agent = np.concatenate([obs_agent, padding], axis=1)

                        action, rnn_state = self.policies[agent_id].act(
                            obs_agent,
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            deterministic=True,
                            agent_id=agent_id
                        )

                # Update RNN states
                rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                raw_action = action_np[0]
                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()
            
            # Step environment
            obs, rewards, dones, infos = self.env.step([actions_env])

            # Retrieve the EXACT (clipped) actions the env executed.
            # Raw NN outputs may exceed env bounds (e.g. retailer max=10, DC max=5000).
            # env._clip_actions() is the same function called inside env.step().
            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list and len(env_list) > 0:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)
            else:
                # Fallback: use raw actions (no clipping info available)
                executed_actions = {aid: raw_actions[aid] for aid in range(self.args.num_agents)}

            # Also store clipped actions for trajectory before the agent loop
            if save_trajectory:
                for agent_id in range(self.args.num_agents):
                    trajectory['actions'][agent_id].append(
                        np.array(executed_actions[agent_id], dtype=float).copy()
                    )

            if env_list and len(env_list) > 0:
                # Calculate costs from rewards (rewards are negative costs)
                for agent_id in range(self.args.num_agents):
                    # Extract scalar reward (handle (1,) shape from env wrapper)
                    reward = float(np.array(rewards[0][agent_id]).item())
                    cost = -reward

                    episode_data['agent_rewards'][agent_id] += reward
                    episode_data['agent_costs'][agent_id] += cost
                    episode_data['total_reward'] += reward
                    episode_data['total_cost'] += cost

                    # === COST BREAKDOWN CALCULATION ===
                    holding_cost_step = 0
                    backlog_cost_step = 0
                    ordering_cost_step = 0

                    is_dc = agent_id < 2  # First 2 agents are DCs

                    if is_dc:
                        dc_idx = agent_id
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_dc[dc_idx][sku]
                            # DC backlog: flat backlog[dc_id] is always 0 — sum dc_retailer_backlog instead
                            dc_owed_sku = sum(
                                env_state.dc_retailer_backlog[agent_id][r_id][sku]
                                for r_id in env_state.dc_assignments[agent_id]
                            )
                            backlog_cost_step += dc_owed_sku * env_state.B_dc[dc_idx][sku]
                            # Use CLIPPED executed action for ordering cost (matches env's _calculate_rewards)
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                price = pre_step_prices[sku] if pre_step_prices is not None else env_state.market_prices[sku]
                                ordering_cost_step += env_state.C_fixed_dc[dc_idx][sku] + (price * order_qty)
                    else:
                        retailer_idx = agent_id - 2
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_retailer[retailer_idx][sku]
                            backlog_cost_step += env_state.backlog[agent_id][sku] * env_state.B_retailer[retailer_idx][sku]
                            # Use CLIPPED executed action for ordering cost
                            order_qty = executed_actions[agent_id][sku]
                            if order_qty > 0:
                                var_cost = env_state.C_var_retailer[retailer_idx][assigned_dc][sku]
                                ordering_cost_step += env_state.C_fixed_retailer[retailer_idx][sku] + (var_cost * order_qty)

                        # --- Order-count fill rate (mirrors GNN script) ---
                        for sku in range(n_skus):
                            placed = env_state.step_orders_placed.get(agent_id, {}).get(sku, 0)
                            from_stock = env_state.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                            episode_data['_orders_placed'][agent_id]     += placed
                            episode_data['_orders_from_stock'][agent_id] += from_stock
                            actual_demand = 0.0
                            if (sku < len(env_state.demand_history) and
                                    retailer_idx < len(env_state.demand_history[sku]) and
                                    len(env_state.demand_history[sku][retailer_idx]) > 0):
                                actual_demand = float(env_state.demand_history[sku][retailer_idx][-1])
                            episode_data['_demand_total'][agent_id] += actual_demand
                            episode_data['_demand_met'][agent_id] += float(from_stock)

                    # Accumulate cost components
                    episode_data['holding_costs'][agent_id] += holding_cost_step
                    episode_data['backlog_costs'][agent_id] += backlog_cost_step
                    episode_data['ordering_costs'][agent_id] += ordering_cost_step

                    # Track inventory and backlog
                    inv = env_state.inventory[agent_id].sum()
                    bl = env_state.backlog[agent_id].sum()
                    episode_data['avg_inventory'][agent_id] += inv
                    episode_data['avg_backlog'][agent_id] += bl

                    if save_trajectory:
                        inv_vec = env_state.inventory[agent_id]
                        trajectory['inventory'][agent_id].append(float(inv))
                        trajectory['inventory_skus'][agent_id].append(
                            np.array(inv_vec, dtype=float).copy()
                        )
                        trajectory['backlog'][agent_id].append(float(bl))
                        trajectory['rewards'][agent_id].append(reward)
                        # NOTE: trajectory['actions'] already appended above (clipped)

                        # -- Demand logging --
                        if agent_id < 2:  # DC: aggregate orders-placed from assigned retailers
                            demand_vec = np.zeros(n_skus, dtype=float)
                            for r_id in env_state.dc_assignments[agent_id]:
                                op = env_state.step_orders_placed.get(r_id, {})
                                for s in range(n_skus):
                                    demand_vec[s] += op.get(s, 0.0)
                        else:  # Retailer: real customer demand from env
                            demand_vec = np.array(
                                env_state.step_demand.get(agent_id, np.zeros(n_skus, dtype=float)),
                                dtype=float,
                            )
                        trajectory['demand'][agent_id].append(demand_vec.copy())

                        # -- Normalized demand / inventory / order (retailers only) --
                        if agent_id < 2:
                            trajectory['norm_demand'][agent_id].append(np.zeros(n_skus, dtype=float))
                            trajectory['norm_inventory'][agent_id].append(np.zeros(n_skus, dtype=float))
                            trajectory['norm_order'][agent_id].append(np.zeros(n_skus, dtype=float))
                        else:
                            dm = getattr(env_state, 'demand_mean', np.ones(n_skus) * 1.5)
                            ds = getattr(env_state, 'demand_std', np.ones(n_skus) * 1.0)
                            demand_cap = np.maximum(dm + 3.0 * ds, 1e-6)
                            norm_d = (demand_vec / demand_cap).astype(float)
                            norm_inv = (np.array(inv_vec, dtype=float) / 150.0)
                            # Use clipped executed action for normalisation
                            act_clip = np.array(executed_actions[agent_id], dtype=float)
                            norm_ord = np.clip(act_clip / 10.0, 0.0, 1.0)  # retailer max=10
                            trajectory['norm_demand'][agent_id].append(norm_d)
                            trajectory['norm_inventory'][agent_id].append(norm_inv)
                            trajectory['norm_order'][agent_id].append(norm_ord)

                        # -- Order-count fill-rate per step --
                        op_vec = np.array([env_state.step_orders_placed.get(agent_id, {}).get(s, 0)
                                           for s in range(n_skus)], dtype=float)
                        ofs_vec = np.array([env_state.step_orders_from_stock.get(agent_id, {}).get(s, 0)
                                            for s in range(n_skus)], dtype=float)
                        trajectory['orders_placed'][agent_id].append(op_vec)
                        trajectory['orders_from_stock'][agent_id].append(ofs_vec)

                        # Store normalization scales once
                        if trajectory['norm_scales'] is None and hasattr(env_state, 'demand_mean'):
                            dm2 = np.array(env_state.demand_mean, dtype=float).flatten()
                            ds2 = np.array(env_state.demand_std, dtype=float).flatten()
                            trajectory['norm_scales'] = {
                                'demand_mean_0': float(dm2[0]) if len(dm2) > 0 else 0,
                                'demand_mean_1': float(dm2[1]) if len(dm2) > 1 else 0,
                                'demand_mean_2': float(dm2[2]) if len(dm2) > 2 else 0,
                                'demand_std_0': float(ds2[0]) if len(ds2) > 0 else 0,
                                'demand_std_1': float(ds2[1]) if len(ds2) > 1 else 0,
                                'demand_std_2': float(ds2[2]) if len(ds2) > 2 else 0,
                                'demand_cap_0': float(dm2[0] + 3 * ds2[0]) if len(dm2) > 0 else 0,
                                'demand_cap_1': float(dm2[1] + 3 * ds2[1]) if len(dm2) > 1 else 0,
                                'demand_cap_2': float(dm2[2] + 3 * ds2[2]) if len(dm2) > 2 else 0,
                                'inv_scale_retailer': 150.0,
                                'backlog_scale_retailer': 100.0,
                                'order_min_retailer': 0,
                                'order_max_retailer': 10.0,
                            }

                # Store final state
                if step == self.args.episode_length - 1:
                    episode_data['final_inventory'] = [
                        env_state.inventory[i].sum() for i in range(self.args.num_agents)
                    ]
                    episode_data['final_backlog'] = [
                        env_state.backlog[i].sum() for i in range(self.args.num_agents)
                    ]
        
        # Calculate averages and percentages
        T = self.args.episode_length
        for agent_id in range(self.args.num_agents):
            episode_data['avg_inventory'][agent_id] /= T
            episode_data['avg_backlog'][agent_id] /= T
            # Retailer: order-count fill rate (mirrors GNN script)
            if agent_id >= 2:
                placed     = episode_data['_orders_placed'][agent_id]
                from_stock = episode_data['_orders_from_stock'][agent_id]
                episode_data['service_level'][agent_id] = (
                    (from_stock / placed * 100.0) if placed > 0 else 100.0
                )
            else:  # DC: fraction of steps with positive inventory
                episode_data['service_level'][agent_id] = (
                    sum(1 for v in episode_data.get('_dc_inv_positive', {}).get(agent_id, [])
                        if v) / T * 100
                    if episode_data.get('_dc_inv_positive') else
                    100.0
                )
        
        # Save trajectory for first episode
        if save_trajectory:
            self.detailed_trajectory = trajectory

        return episode_data
    
    def generate_report(self):
        """Generate comprehensive evaluation report with metrics and visualizations."""
        print("Generating evaluation report...")

        # Calculate aggregate statistics
        stats = self._calculate_statistics()

        # Save metrics to JSON
        self._save_metrics_json(stats)

        # Save metrics to CSV
        self._save_metrics_csv()

        # Generate visualizations
        self._create_visualizations(stats)

        # Save step trajectory Excel (mirrors GNN script)
        self._save_step_trajectory_excel()

        # Print summary
        self._print_summary(stats)

        print(f"\n✓ Evaluation report saved to: {self.save_dir}\n")
    
    def _calculate_statistics(self):
        """Calculate aggregate statistics from all episodes."""
        stats = {
            'num_episodes': len(self.episode_metrics),
            'episode_length': self.args.episode_length,
            'total_reward': {
                'mean': np.mean([m['total_reward'] for m in self.episode_metrics]),
                'std': np.std([m['total_reward'] for m in self.episode_metrics]),
                'min': np.min([m['total_reward'] for m in self.episode_metrics]),
                'max': np.max([m['total_reward'] for m in self.episode_metrics]),
            },
            'total_cost': {
                'mean': np.mean([m['total_cost'] for m in self.episode_metrics]),
                'std': np.std([m['total_cost'] for m in self.episode_metrics]),
                'min': np.min([m['total_cost'] for m in self.episode_metrics]),
                'max': np.max([m['total_cost'] for m in self.episode_metrics]),
            },
            'per_agent': {}
        }
        
        # Per-agent statistics
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            agent_name = f"{agent_type}_{agent_id}"
            
            stats['per_agent'][agent_name] = {
                'avg_reward': np.mean([m['agent_rewards'][agent_id] for m in self.episode_metrics]),
                'avg_cost': np.mean([m['agent_costs'][agent_id] for m in self.episode_metrics]),
                'avg_holding_cost': np.mean([m['holding_costs'][agent_id] for m in self.episode_metrics]),
                'avg_backlog_cost': np.mean([m['backlog_costs'][agent_id] for m in self.episode_metrics]),
                'avg_ordering_cost': np.mean([m['ordering_costs'][agent_id] for m in self.episode_metrics]),
                'avg_inventory': np.mean([m['avg_inventory'][agent_id] for m in self.episode_metrics]),
                'avg_backlog': np.mean([m['avg_backlog'][agent_id] for m in self.episode_metrics]),
                'service_level': np.mean([m['service_level'][agent_id] for m in self.episode_metrics]),
            }
        
        return stats
    
    def _save_metrics_json(self, stats):
        """Save metrics to JSON file."""
        json_path = self.save_dir / "evaluation_metrics.json"
        
        # Add metadata
        output = {
            'metadata': {
                'model_dir': str(self.args.model_dir),
                'num_episodes': self.args.num_episodes,
                'episode_length': self.args.episode_length,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'statistics': stats,
            'episode_data': self.episode_metrics
        }
        
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Saved metrics JSON: {json_path.name}")
    
    def _save_metrics_csv(self):
        """Save episode metrics to CSV file (one row per episode, system-wide metrics only)."""
        import csv

        results_path = self.save_dir / 'results_standard_happo.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'

        for csv_path in (results_path, compat_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                                 'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])

                for ep_num, metrics in enumerate(self.episode_metrics):
                    fill_rate = float(np.mean(metrics['service_level']))
                    total_placed     = sum(metrics['_orders_placed'][aid]     for aid in range(2, self.args.num_agents))
                    total_from_stock = sum(metrics['_orders_from_stock'][aid] for aid in range(2, self.args.num_agents))
                    lost_sales = total_placed - total_from_stock
                    avg_inventory = float(np.mean(metrics['avg_inventory']))
                    total_holding = float(np.sum(metrics['holding_costs']))
                    total_backlog = float(np.sum(metrics['backlog_costs']))
                    total_ordering = float(np.sum(metrics['ordering_costs']))
                    writer.writerow([
                        ep_num + 1,
                        round(metrics['total_cost'], 4),
                        round(fill_rate, 4),
                        round(lost_sales, 4),
                        round(avg_inventory, 4),
                        round(total_holding, 4),
                        round(total_backlog, 4),
                        round(total_ordering, 4),
                    ])

        print(f"✓ Saved metrics CSV: {results_path.name}  (also {compat_path.name})")
    
    def _create_visualizations(self, stats):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")

        # 1. Episode rewards over time
        self._plot_episode_rewards()

        # 2. Cost breakdown by agent
        self._plot_cost_breakdown(stats)

        # 3. Service level comparison
        self._plot_service_levels(stats)

        # 4. Inventory trajectory (first episode)
        if self.detailed_trajectory:
            self._plot_trajectory()
            self._plot_dc_inventory_fluctuation()
            self._plot_retailer_inventory_fluctuation()

        # 5. Performance distribution
        self._plot_performance_distribution()

        print("✓ All visualizations created")
    
    def _plot_episode_rewards(self):
        """Plot total reward across episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(self.episode_metrics) + 1)
        rewards = [m['total_reward'] for m in self.episode_metrics]
        
        ax.plot(episodes, rewards, linewidth=2, alpha=0.7, label='Episode Reward')
        
        # Rolling average
        window = min(10, len(rewards))
        if len(rewards) >= window:
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), rolling_avg,
                   linewidth=2.5, color='red', label=f'{window}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('MADRL Model Performance Across Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'episode_rewards.png', dpi=300)
        plt.close()
    
    def _plot_cost_breakdown(self, stats):
        """Plot cost breakdown by agent with stacked bars showing cost components."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        agents = list(stats['per_agent'].keys())
        holding_costs = [stats['per_agent'][a]['avg_holding_cost'] for a in agents]
        backlog_costs = [stats['per_agent'][a]['avg_backlog_cost'] for a in agents]
        ordering_costs = [stats['per_agent'][a]['avg_ordering_cost'] for a in agents]
        
        x = np.arange(len(agents))
        width = 0.6
        
        # Create stacked bars (colors match test_trained_model_gnn.py)
        bars1 = ax.bar(x, holding_costs, width, label='Holding', color='#4ECDC4', edgecolor='black')
        bars2 = ax.bar(x, backlog_costs, width, bottom=holding_costs,
                       label='Backlog', color='#FF6B6B', edgecolor='black')
        bars3 = ax.bar(x, ordering_costs, width,
                       bottom=np.array(holding_costs) + np.array(backlog_costs),
                       label='Ordering', color='#FFD700', edgecolor='black')
        
        # Add total cost labels on top of each bar
        for i, agent in enumerate(agents):
            total = holding_costs[i] + backlog_costs[i] + ordering_costs[i]
            ax.text(i, total, f'{total:.0f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Average Cost per Episode', fontsize=12)
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_title('Cost Breakdown by Agent Type (Holding + Backlog + Ordering)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'cost_breakdown.png', dpi=300)
        plt.close()
    
    def _plot_service_levels(self, stats):
        """Plot service level comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents = list(stats['per_agent'].keys())
        service_levels = [stats['per_agent'][a]['service_level'] for a in agents]
        
        n_agents = len(agents)
        colors = ['#2E86AB'] * 2 + ['#A23B72'] * (n_agents - 2)
        bars = ax.bar(agents, service_levels, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add target line (e.g., 95% service level)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
        
        ax.set_ylabel('Service Level (%)', fontsize=12)
        ax.set_title('Service Level by Agent (Demand Fill-Rate for Retailers)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()
    
    def _plot_trajectory(self):
        """Plot detailed trajectory from first episode."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        days = range(1, self.args.episode_length + 1)
        
        # Plot 1: Inventory levels
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[0].plot(days, self.detailed_trajectory['inventory'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[0].set_ylabel('Total Inventory', fontsize=11)
        axes[0].set_title('Inventory Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', ncol=2)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Backlog levels
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[1].plot(days, self.detailed_trajectory['backlog'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[1].set_ylabel('Total Backlog', fontsize=11)
        axes[1].set_title('Backlog Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', ncol=2)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Rewards
        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < 2 else "Retailer"
            label = f"{agent_type}_{agent_id}"
            axes[2].plot(days, self.detailed_trajectory['rewards'][agent_id],
                        label=label, linewidth=1.5, alpha=0.8)
        
        axes[2].set_xlabel('Day', fontsize=11)
        axes[2].set_ylabel('Reward (Negative Cost)', fontsize=11)
        axes[2].set_title('Reward Trajectory (First Episode)', fontsize=12, fontweight='bold')
        axes[2].legend(loc='lower right', ncol=2)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'detailed_trajectory.png', dpi=300)
        plt.close()
    
    def _plot_performance_distribution(self):
        """Plot distribution of episode performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reward distribution
        rewards = [m['total_reward'] for m in self.episode_metrics]
        axes[0].hist(rewards, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rewards):.1f}')
        axes[0].set_xlabel('Total Episode Reward', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Episode Rewards', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cost distribution
        costs = [m['total_cost'] for m in self.episode_metrics]
        axes[1].hist(costs, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(costs), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(costs):.1f}')
        axes[1].set_xlabel('Total Episode Cost', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Episode Costs', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_distribution.png', dpi=300)
        plt.close()

    def _save_step_trajectory_excel(self):
        """
        Save step-level trajectory to Excel for Episode 1 (mirrors test_trained_model_gnn.py).
        Two sheets:
          - Data:   step, agent, inv, backlog, demand, action, norm values, SL (step + cumulative).
          - Scales: normalization constants used for demand/inv/order.
        """
        if not self.detailed_trajectory:
            return

        traj = self.detailed_trajectory
        n_agents = self.args.num_agents
        num_steps = len(traj['inventory'][0])

        # Detect n_skus
        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        cum_orders_placed = {aid: 0 for aid in range(n_agents)}
        cum_orders_from_stock = {aid: 0 for aid in range(n_agents)}

        rows = []
        for step in range(num_steps):
            for aid in range(n_agents):
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
                for i in range(n_skus):
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
                pd.DataFrame([{'note': 'Scales not captured'}]).to_excel(
                    writer, sheet_name='Scales', index=False)
        print(f'✓ Saved step_trajectory_ep1.xlsx (Data + Scales)')

    def _plot_dc_inventory_fluctuation(self):
        """Plot per-SKU and total inventory fluctuation for each DC (Episode 1)."""
        traj = self.detailed_trajectory
        if not traj or 'inventory_skus' not in traj:
            return

        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        days = np.arange(1, self.args.episode_length + 1)
        dc_ids = [i for i in range(self.args.num_agents) if i < 2]
        n_dcs = len(dc_ids)
        n_cols = n_skus + 1  # per-SKU cols + total

        fig, axes = plt.subplots(n_dcs, n_cols,
                                 figsize=(5 * n_cols, 4 * n_dcs),
                                 squeeze=False)
        fig.suptitle('DC Inventory Fluctuation Over Steps (Episode 1)',
                     fontsize=15, fontweight='bold', y=1.01)

        sku_colors = ['#2196F3', '#4CAF50', '#FF9800']
        total_color = '#9C27B0'

        for row, dc_id in enumerate(dc_ids):
            dc_label = f'DC_{dc_id}'
            inv_skus = np.array(traj['inventory_skus'][dc_id], dtype=float)  # [T, n_skus]
            inv_total = np.array(traj['inventory'][dc_id], dtype=float)       # [T]

            for sku in range(n_skus):
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

            ax_total = axes[row][n_cols - 1]
            ax_total.plot(days, inv_total, color=total_color, linewidth=2.0, alpha=0.9, label='Total')
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
        print(f'✓ Saved DC inventory fluctuation plot: {out.name}')

    def _plot_retailer_inventory_fluctuation(self):
        """Plot inventory fluctuation for all retailers (Episode 1): individual lines + mean±std."""
        traj = self.detailed_trajectory
        if not traj:
            return

        days = np.arange(1, self.args.episode_length + 1)
        retailer_ids = [i for i in range(self.args.num_agents) if i >= 2]
        n_retailers = len(retailer_ids)

        inv_matrix = np.array([traj['inventory'][rid] for rid in retailer_ids], dtype=float)  # [n_r, T]

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        fig.suptitle('Retailer Inventory Fluctuation Over Steps (Episode 1)',
                     fontsize=15, fontweight='bold')

        cmap = plt.cm.get_cmap('tab20', n_retailers)
        for i, rid in enumerate(retailer_ids):
            axes[0].plot(days, inv_matrix[i], color=cmap(i), linewidth=1.2, alpha=0.75,
                         label=f'R_{rid}')
        axes[0].set_title('Individual Retailer Inventory', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Total Inventory', fontsize=11)
        axes[0].legend(loc='upper right', ncol=max(1, n_retailers // 5), fontsize=7, framealpha=0.7)
        axes[0].grid(True, alpha=0.3)

        mean_inv = inv_matrix.mean(axis=0)
        std_inv = inv_matrix.std(axis=0)
        axes[1].plot(days, mean_inv, color='#1565C0', linewidth=2.0, label='Mean (all retailers)')
        axes[1].fill_between(days, np.maximum(mean_inv - std_inv, 0), mean_inv + std_inv,
                             alpha=0.2, color='#1565C0', label='±1 std')
        axes[1].axhline(mean_inv.mean(), color='red', linestyle='--', linewidth=1.2,
                        label=f'Time-avg: {mean_inv.mean():.0f}')
        axes[1].set_title('System-Wide Avg Retailer Inventory', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Avg Total Inventory', fontsize=11)
        axes[1].set_xlabel('Step (day)', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'retailer_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved retailer inventory fluctuation plot: {out.name}')

    def _print_summary(self, stats):
        """Print evaluation summary to console."""
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {self.args.model_dir}")
        print(f"Episodes evaluated: {stats['num_episodes']}")
        print(f"Episode length: {stats['episode_length']} days")
        print()
        
        print(f"OVERALL PERFORMANCE:")
        print(f"  Average Total Reward: {stats['total_reward']['mean']:.2f} ± {stats['total_reward']['std']:.2f}")
        print(f"  Average Total Cost:   {stats['total_cost']['mean']:.2f} ± {stats['total_cost']['std']:.2f}")
        print(f"  Best Episode Reward:  {stats['total_reward']['max']:.2f}")
        print(f"  Worst Episode Reward: {stats['total_reward']['min']:.2f}")
        print()
        
        print(f"PER-AGENT BREAKDOWN:")
        print(f"{'Agent':<15} {'Avg Cost':<12} {'Avg Inventory':<15} {'Avg Backlog':<15} {'Service Level':<15}")
        print(f"{'-'*70}")
        for agent_name, agent_stats in stats['per_agent'].items():
            print(f"{agent_name:<15} "
                  f"{agent_stats['avg_cost']:<12.2f} "
                  f"{agent_stats['avg_inventory']:<15.2f} "
                  f"{agent_stats['avg_backlog']:<15.2f} "
                  f"{agent_stats['service_level']:<15.1f}%")
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {self.save_dir}")
        print(f"{'='*70}\n")


def main():
    """Main evaluation function."""
    args = parse_test_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args)
    
    # Run evaluation
    evaluator.evaluate()
    
    # Generate report
    evaluator.generate_report()
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
