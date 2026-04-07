#!/usr/bin/env python
"""
Final Thesis Multi-SKU Robustness Validation
=============================================
Evaluates three inventory management policies across three distinct demand
scenarios, computing advanced thesis-grade metrics and generating
publication-ready visualizations.

Models Evaluated:
  1. (s,S) Heuristic   — classical base-stock policy (no learning)
  2. Standard HAPPO    — baseline MLP-based MARL model
  3. GNN-HAPPO         — proposed GNN-based MARL model

Demand Scenarios (3 SKUs each):
  Scenario 1 — Balanced        : moderate mean, low variance
  Scenario 2 — Heterogeneous   : mixed high-volume / high-volatility SKUs
  Scenario 3 — Extreme Stress  : high mean + high variance for all SKUs

Advanced Metrics (beyond Cost & Fill Rate):
  • Inventory Turnover Ratio  = Total Sales / Avg Inventory
  • On-Shelf Availability     = % of steps where inventory > 0
  • Cost-Service Pareto Data  = (Total Cost, Fill Rate) per episode

Usage:
    python Final_Thesis_MultiSKU_Robustness.py \\
        --gnn_model_dir   results/gnn/run_seed_1/models \\
        --happo_model_dir results/baseline/run_seed_1/models \\
        --num_episodes 100 \\
        --episode_length 365 \\
        --save_dir robustness_results

    # Run only heuristic (no trained models needed):
    python Final_Thesis_MultiSKU_Robustness.py --heuristic_only
"""

import sys
import os
import argparse
import numpy as np
import json
import csv
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from envs.multi_dc_env import MultiDCInventoryEnv

# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Scenario_1_Balanced": {
        "SKU_0": {"mean": 1.41, "std": 1.99},
        "SKU_1": {"mean": 1.06, "std": 1.28},
        "SKU_2": {"mean": 0.77,  "std": 1.06},
        "label": "Balanced\n(Low Stress)",
        "short":  "S1-Balanced",
    },
    "Scenario_2_HighDemand": {
        "SKU_0": {"mean": 2.41, "std": 1.99},
        "SKU_1": {"mean": 2.06, "std": 1.28},
        "SKU_2": {"mean": 1.77,  "std": 1.06},   # Low Volume, High Volatility
        "label": "High Demand \n(Mixed Stress)",
        "short":  "S2-High",
    },
    "Scenario_3_Extreme_Stress": {
        "SKU_0": {"mean": 1.41, "std": 2.99},
        "SKU_1": {"mean": 1.06, "std": 2.28},
        "SKU_2": {"mean": 0.77,  "std": 1.06},
        "label": "Extreme Stress\n(High Demand)",
        "short":  "S3-Extreme",
    },
}

MODEL_COLORS = {
    "BaseStock": "#E67E22",
    "HAPPO":     "#2980B9",
    "GNN-HAPPO": "#27AE60",
}
MODEL_MARKERS = {
    "BaseStock": "o",
    "HAPPO":     "s",
    "GNN-HAPPO": "^",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness Validation: 3 Scenarios × 3 Models"
    )
    p.add_argument("--gnn_model_dir",   type=str, default="results/03Apr_gnn1/run_seed_1/models",
                   help="Path to saved GNN-HAPPO model directory")
    p.add_argument("--happo_model_dir", type=str, default="results/01Apr_base1/run_seed_1/models",
                   help="Path to saved Standard-HAPPO model directory")
    p.add_argument("--config_path",     type=str,
                   default="configs/multi_dc_config.yaml")

    p.add_argument("--num_episodes",    type=int, default=1)
    p.add_argument("--episode_length",  type=int, default=90)
    p.add_argument("--seed",            type=int, default=42)

    p.add_argument("--save_dir",        type=str,
                   default="robustness_results")
    p.add_argument("--experiment_name", type=str, default=None)

    p.add_argument("--heuristic_only",  action="store_true",
                   help="Evaluate only the (s,S) heuristic (no model paths needed)")
    p.add_argument("--cuda",            action="store_true", default=False)

    # (s,S) policy levels
    p.add_argument("--s_dc",        type=float, default=200.0)
    p.add_argument("--S_dc",        type=float, default=2000.0)
    p.add_argument("--s_retailer",  type=float, default=5.0)
    p.add_argument("--S_retailer",  type=float, default=12.0)

    # GNN architecture (auto-detected from checkpoint; override if needed)
    p.add_argument("--gnn_type",            type=str,   default="GAT")
    p.add_argument("--gnn_hidden_dim",      type=int,   default=128)
    p.add_argument("--gnn_num_layers",      type=int,   default=2)
    p.add_argument("--num_attention_heads", type=int,   default=4)
    p.add_argument("--gnn_dropout",         type=float, default=0.1)
    p.add_argument("--use_residual",
                   type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--critic_pooling",      type=str,   default="mean")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def apply_scenario_to_env(env: MultiDCInventoryEnv, scenario: dict):
    """Patch demand_mean / demand_std onto an existing env instance."""
    means = np.array([scenario[f"SKU_{i}"]["mean"] for i in range(env.n_skus)],
                     dtype=np.float32)
    stds  = np.array([scenario[f"SKU_{i}"]["std"]  for i in range(env.n_skus)],
                     dtype=np.float32)
    env.demand_mean = means
    env.demand_std  = stds


def create_base_env(config_path: str, episode_length: int) -> MultiDCInventoryEnv:
    env = MultiDCInventoryEnv(config_path=config_path)
    env.max_days = episode_length
    return env


# ---------------------------------------------------------------------------
# (s,S) Heuristic Policy  (verbatim from Test_baseline_basestock.py)
# ---------------------------------------------------------------------------

class SsPolicy:
    def __init__(self, s_dc, S_dc, s_retailer, S_retailer,
                 n_dcs, n_agents, n_skus):
        self.s_dc = s_dc;  self.S_dc = S_dc
        self.s_retailer = s_retailer;  self.S_retailer = S_retailer
        self.n_dcs = n_dcs;  self.n_agents = n_agents;  self.n_skus = n_skus

    def get_actions(self, env):
        actions = {}
        for agent_id in range(self.n_agents):
            order = np.zeros(self.n_skus, dtype=np.float32)
            for sku in range(self.n_skus):
                on_hand = float(env.inventory[agent_id][sku])
                pipeline_qty = sum(
                    o["qty"] for o in env.pipeline[agent_id] if o["sku"] == sku
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
                    order[sku] = max(0.0, S - ip)
            actions[agent_id] = order
        return actions


# ---------------------------------------------------------------------------
# DRL Policy loaders
# ---------------------------------------------------------------------------

def _load_happo_policies(model_dir: str, env, device, args):
    """Load standard HAPPO (MLP) policies for all agents."""
    import torch
    from config import get_config
    from algorithms.happo_policy import HAPPO_Policy

    model_dir = Path(model_dir)
    n_agents  = env.n_agents

    parser = get_config()
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=n_agents,
        use_centralized_V=True,
        algorithm_name="happo",
        hidden_size=128, layer_N=2,
        use_ReLU=True, use_orthogonal=True,
        gain=0.01, recurrent_N=2,
        use_naive_recurrent_policy=True,
    )
    all_args = parser.parse_known_args([])[0]

    # Build fake share_obs_space from env wrapper equivalent
    from gymnasium import spaces as gym_spaces

    policies = []
    for agent_id in range(n_agents):
        obs_dim = (env.obs_dim_dc if agent_id < env.n_dcs
                   else env.obs_dim_retailer)
        obs_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Shared obs space: concatenate all agents' obs
        share_dim = (env.obs_dim_dc * env.n_dcs
                     + env.obs_dim_retailer * env.n_retailers)
        share_obs_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32
        )
        act_space = (env.action_space_dc if agent_id < env.n_dcs
                     else env.action_space_retailer)

        # Find best checkpoint
        files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No checkpoint for agent {agent_id} in {model_dir}"
            )
        reward_files = []
        for f in files:
            if f.name == f"actor_agent{agent_id}.pt":
                continue
            try:
                rv = float(f.name.split("_reward_")[1].replace(".pt", ""))
                reward_files.append((rv, f))
            except (IndexError, ValueError):
                pass
        if reward_files:
            reward_files.sort(key=lambda x: x[0], reverse=True)
            best_file = reward_files[0][1]
        else:
            plain = model_dir / f"actor_agent{agent_id}.pt"
            best_file = plain if plain.exists() else files[0]

        state_dict = torch.load(str(best_file), map_location=device)

        # Auto-detect obs dim from saved weights
        saved_input_dim = None
        if "base.mlp.fc1.0.weight" in state_dict:
            saved_input_dim = state_dict["base.mlp.fc1.0.weight"].shape[1]
        if saved_input_dim and saved_input_dim != obs_dim:
            obs_space = gym_spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(saved_input_dim,), dtype=np.float32
            )

        policy = HAPPO_Policy(all_args, obs_space, share_obs_space,
                              act_space, device=device)
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)
        print(f"  [HAPPO] Agent {agent_id:2d} loaded <- {best_file.name}")

    return policies


def _load_gnn_policies(model_dir: str, env, device, args):
    """Load GNN-HAPPO policies for all agents."""
    import torch
    from config import get_config
    from algorithms.gnn_happo_policy import GNN_HAPPO_Policy
    from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
    from gymnasium import spaces as gym_spaces

    model_dir = Path(model_dir)
    n_agents  = env.n_agents
    n_dcs     = env.n_dcs

    # Auto-detect obs dim and GNN type from agent-0 checkpoint
    agent0_files = sorted(model_dir.glob("actor_agent0*.pt"))
    if not agent0_files:
        raise FileNotFoundError(f"No actor_agent0*.pt in {model_dir}")
    sd = torch.load(str(agent0_files[0]), map_location="cpu")
    keys = list(sd.keys())
    if any("gnn_base.layers.0.weight" in k for k in keys):
        detected_gnn_type = "GCN"
        obs_dim = sd["gnn_base.layers.0.weight"].shape[0]
    elif any("gnn_base.layers.0.W" in k for k in keys):
        detected_gnn_type = "GAT"
        obs_dim = sd["gnn_base.layers.0.W"].shape[1]
    else:
        detected_gnn_type = args.gnn_type
        obs_dim = max(env.obs_dim_dc, env.obs_dim_retailer)
    print(f"  [GNN] Detected type={detected_gnn_type}  obs_dim={obs_dim}")

    # Build graph adjacency
    adj = build_supply_chain_adjacency(
        n_dcs=n_dcs, n_retailers=n_agents - n_dcs, self_loops=True
    )
    adj = normalize_adjacency(adj, method="symmetric")
    adj_tensor = torch.FloatTensor(adj).to(device)

    parser = get_config()
    parser.add_argument("--gnn_type",            type=str,   default=detected_gnn_type)
    parser.add_argument("--gnn_hidden_dim",      type=int,   default=args.gnn_hidden_dim)
    parser.add_argument("--gnn_num_layers",      type=int,   default=args.gnn_num_layers)
    parser.add_argument("--num_attention_heads", type=int,   default=args.num_attention_heads)
    parser.add_argument("--gnn_dropout",         type=float, default=args.gnn_dropout)
    parser.add_argument("--use_residual",
                        type=lambda x: x.lower() == "true",
                        default=args.use_residual)
    parser.add_argument("--critic_pooling",      type=str,   default=args.critic_pooling)
    parser.add_argument("--single_agent_obs_dim",type=int,   default=obs_dim)
    parser.set_defaults(
        env_name="MultiDC",
        scenario_name="inventory_2echelon",
        num_agents=n_agents,
        use_centralized_V=True,
        algorithm_name="gnn_happo",
        hidden_size=128, layer_N=2,
        use_ReLU=True, use_orthogonal=True,
        gain=0.01, recurrent_N=2,
        use_naive_recurrent_policy=True,
        single_agent_obs_dim=obs_dim,
    )
    all_args = parser.parse_known_args([])[0]

    padded_obs_space = gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    share_dim = (env.obs_dim_dc * env.n_dcs
                 + env.obs_dim_retailer * env.n_retailers)
    share_obs_space = gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32
    )

    policies = []
    for agent_id in range(n_agents):
        act_space = (env.action_space_dc if agent_id < n_dcs
                     else env.action_space_retailer)

        files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No checkpoint for agent {agent_id} in {model_dir}"
            )
        reward_files = []
        for f in files:
            if f.name == f"actor_agent{agent_id}.pt":
                continue
            try:
                rv = float(f.name.split("_reward_")[1].replace(".pt", ""))
                reward_files.append((rv, f))
            except (IndexError, ValueError):
                pass
        if reward_files:
            reward_files.sort(key=lambda x: x[0], reverse=True)
            best_file = reward_files[0][1]
        else:
            plain = model_dir / f"actor_agent{agent_id}.pt"
            best_file = plain if plain.exists() else files[0]

        state_dict = torch.load(str(best_file), map_location=device)

        policy = GNN_HAPPO_Policy(
            all_args, padded_obs_space, share_obs_space, act_space,
            n_agents=n_agents, agent_id=agent_id, device=device,
        )
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)
        print(f"  [GNN] Agent {agent_id:2d} loaded <- {best_file.name}")

    return policies, adj_tensor, obs_dim


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode_heuristic(env, policy, episode_length, n_skus):
    """Run one episode with (s,S) heuristic. Returns metrics dict."""
    env.reset()
    n_agents = env.n_agents
    n_dcs    = env.n_dcs

    acc = _init_accumulators(n_agents, n_skus)

    for step in range(episode_length):
        pre_prices = env.market_prices.copy()
        actions    = policy.get_actions(env)

        try:
            _, rewards, _, infos = env.step(actions)
        except Exception as e:
            print(f"[WARN] env.step raised: {e} at step {step}")
            break

        executed = env._clip_actions(actions)
        _accumulate_step(acc, env, rewards, executed, pre_prices,
                         n_agents, n_dcs, n_skus, step, episode_length)

    return _finalize_episode(acc, env, episode_length, n_agents, n_dcs, n_skus)


def run_episode_happo(env, policies, episode_length, n_skus, device):
    """Run one episode with Standard HAPPO policies."""
    import torch

    # MultiDCInventoryEnv.reset() returns a plain dict {agent_id: obs_array}
    obs_raw = env.reset()
    n_agents = env.n_agents
    n_dcs    = env.n_dcs

    # Same shape as test_trained_model.py: (1, n_agents, recurrent_N=2, hidden=128)
    # Using (n_agents, 2, 128) caused rnn_state (1,2,128) to fail silent assignment,
    # resetting the RNN to zeros every step and crippling temporal context.
    rnn_states = np.zeros((1, n_agents, 2, 128), dtype=np.float32)
    masks      = np.ones( (1, n_agents, 1),       dtype=np.float32)

    acc = _init_accumulators(n_agents, n_skus)

    for step in range(episode_length):
        pre_prices = env.market_prices.copy()

        actions_list   = []
        raw_actions_dict = {}

        for agent_id in range(n_agents):
            obs_dim = (env.obs_dim_dc if agent_id < n_dcs
                       else env.obs_dim_retailer)
            agent_obs = np.zeros((1, obs_dim), dtype=np.float32)
            if agent_id in obs_raw:
                raw = np.array(obs_raw[agent_id], dtype=np.float32).flatten()
                copy_len = min(len(raw), obs_dim)
                agent_obs[0, :copy_len] = raw[:copy_len]

            # Pad if policy expects larger input
            policy_in_dim = policies[agent_id].obs_space.shape[0]
            if agent_obs.shape[1] < policy_in_dim:
                diff = policy_in_dim - agent_obs.shape[1]
                agent_obs = np.concatenate(
                    [agent_obs, np.zeros((1, diff), dtype=np.float32)], axis=1
                )

            with torch.no_grad():
                action, rnn_state = policies[agent_id].act(
                    agent_obs,
                    rnn_states[:, agent_id],   # (1, 2, 128)
                    masks[:, agent_id],         # (1, 1)
                    deterministic=True,
                    agent_id=agent_id,
                )
            rnn_states[:, agent_id] = (
                rnn_state.cpu().numpy()
                if isinstance(rnn_state, torch.Tensor) else rnn_state
            )
            raw_act = (
                action.cpu().numpy()[0]
                if isinstance(action, torch.Tensor) else action[0]
            )
            actions_list.append(raw_act)
            raw_actions_dict[agent_id] = raw_act.copy()

        actions_dict = {i: actions_list[i] for i in range(n_agents)}
        obs_raw, rewards, _, infos = env.step(actions_dict)
        executed = env._clip_actions(raw_actions_dict)
        _accumulate_step(acc, env, rewards, executed, pre_prices,
                         n_agents, n_dcs, n_skus, step, episode_length)

    return _finalize_episode(acc, env, episode_length, n_agents, n_dcs, n_skus)


def run_episode_gnn(env, policies, adj_tensor, obs_dim_padded,
                    episode_length, n_skus, device):
    """Run one episode with GNN-HAPPO policies."""
    import torch

    obs_raw  = env.reset()   # plain dict {agent_id: obs_array}
    n_agents = env.n_agents
    n_dcs    = env.n_dcs

    # Shape must match test_trained_model_gnn.py exactly:
    #   (1, n_agents, recurrent_N=2, hidden_size=128)
    # Using (n_agents, 2, 128) caused act() to return (1,2,128) which was
    # silently dropped on assignment, resetting RNN to zeros every step.
    # That made the DC policy think it had no history → large blind orders.
    rnn_states = np.zeros((1, n_agents, 2, 128), dtype=np.float32)
    masks      = np.ones( (1, n_agents, 1),       dtype=np.float32)

    acc = _init_accumulators(n_agents, n_skus)

    for step in range(episode_length):
        pre_prices = env.market_prices.copy()

        # Build padded structured obs [1, n_agents, obs_dim_padded]
        obs_structured = np.zeros((1, n_agents, obs_dim_padded), dtype=np.float32)
        for aid in range(n_agents):
            if aid in obs_raw:
                raw = np.array(obs_raw[aid], dtype=np.float32).flatten()
                d = min(len(raw), obs_dim_padded)
                obs_structured[0, aid, :d] = raw[:d]

        raw_actions_dict = {}
        actions_list     = []

        for agent_id in range(n_agents):
            with torch.no_grad():
                action, rnn_state = policies[agent_id].act(
                    obs_structured,
                    adj_tensor,
                    agent_id,
                    rnn_states[:, agent_id],   # (1, 2, 128) — matches training shape
                    masks[:, agent_id],         # (1, 1)
                    deterministic=False,
                )
            # Update carried RNN state at the correct slice
            rnn_states[:, agent_id] = (
                rnn_state.cpu().numpy()
                if isinstance(rnn_state, torch.Tensor) else rnn_state
            )
            raw_act = (
                action.cpu().numpy()[0]
                if isinstance(action, torch.Tensor) else action[0]
            )
            actions_list.append(raw_act)
            raw_actions_dict[agent_id] = raw_act.copy()

        actions_dict = {i: actions_list[i] for i in range(n_agents)}
        obs_raw, rewards, _, infos = env.step(actions_dict)
        executed = env._clip_actions(raw_actions_dict)
        _accumulate_step(acc, env, rewards, executed, pre_prices,
                         n_agents, n_dcs, n_skus, step, episode_length)

    return _finalize_episode(acc, env, episode_length, n_agents, n_dcs, n_skus)


# ---------------------------------------------------------------------------
# Internal episode accumulator helpers
# ---------------------------------------------------------------------------

# MultiDCInventoryEnv.reset() always returns a plain dict (not a tuple).
# This constant replaces the old _env_returns_tuple() helper which
# incorrectly called env.reset() on every episode just to detect the
# return type — that extra reset consumed random numbers and produced
# different demand sequences from what the caller expected.
_ENV_RETURNS_TUPLE = False


def _init_accumulators(n_agents, n_skus):
    return {
        "total_reward":    0.0,
        "total_cost":      0.0,
        "holding_costs":   [0.0] * n_agents,
        "backlog_costs":   [0.0] * n_agents,
        "ordering_costs":  [0.0] * n_agents,
        "avg_inventory":   [0.0] * n_agents,
        "total_sales":     [0.0] * n_agents,   # units fulfilled (for turnover)
        "osa_steps":       [0]   * n_agents,   # steps with inv > 0 (OSA)
        "_orders_placed":     [0] * n_agents,
        "_orders_from_stock": [0] * n_agents,
    }


def _accumulate_step(acc, env, rewards, executed, pre_prices,
                     n_agents, n_dcs, n_skus, step, episode_length):
    for agent_id in range(n_agents):
        # Note: reward from env includes PBRS shaping + /max_days normalisation.
        # We track it separately for reference but DO NOT derive cost from it.
        # Cost is computed directly from env attributes (holding/backlog/ordering)
        # to match Test_baseline_basestock.py and test_trained_model_gnn.py exactly.
        reward = float(np.array(rewards[agent_id]).item())
        acc["total_reward"] += reward

        h, b, o = 0.0, 0.0, 0.0
        is_dc = agent_id < n_dcs

        if is_dc:
            dc_idx = agent_id
            for sku in range(n_skus):
                h += env.inventory[agent_id][sku] * env.H_dc[dc_idx][sku]
                dc_owed = sum(
                    env.dc_retailer_backlog[agent_id][r][sku]
                    for r in env.dc_assignments[agent_id]
                )
                b += dc_owed * env.B_dc[dc_idx][sku]
                qty = float(executed[agent_id][sku])
                if qty > 0:
                    price = pre_prices[sku]
                    o += env.C_fixed_dc[dc_idx][sku] + price * qty
        else:
            r_idx = agent_id - n_dcs
            adc   = env.retailer_to_dc[agent_id]
            for sku in range(n_skus):
                h += env.inventory[agent_id][sku] * env.H_retailer[r_idx][sku]
                b += env.backlog[agent_id][sku]   * env.B_retailer[r_idx][sku]
                qty = float(executed[agent_id][sku])
                if qty > 0:
                    o += (env.C_fixed_retailer[r_idx][sku]
                          + env.C_var_retailer[r_idx][adc][sku] * qty)

            # Order-count fill rate
            for sku in range(n_skus):
                placed = env.step_orders_placed.get(agent_id, {}).get(sku, 0)
                fstock = env.step_orders_from_stock.get(agent_id, {}).get(sku, 0)
                acc["_orders_placed"][agent_id]     += placed
                acc["_orders_from_stock"][agent_id] += fstock

            # Units sold this step (for turnover)
            fulfilled = env.demand_fulfilled.get(agent_id,
                                                 np.zeros(n_skus, dtype=np.float32))
            acc["total_sales"][agent_id] += float(np.sum(fulfilled))

        acc["holding_costs"][agent_id]  += h
        acc["backlog_costs"][agent_id]  += b
        acc["ordering_costs"][agent_id] += o

        inv_total = float(env.inventory[agent_id].sum())
        acc["avg_inventory"][agent_id] += inv_total

        # OSA: any SKU must have inventory > 0 (retailer-focused)
        if not is_dc:
            osa_ok = int(any(env.inventory[agent_id][s] > 0
                             for s in range(n_skus)))
            acc["osa_steps"][agent_id] += osa_ok


def _finalize_episode(acc, env, episode_length, n_agents, n_dcs, n_skus):
    T = episode_length
    m = {}
    m["total_reward"] = acc["total_reward"]

    # Cost is the sum of cost components accumulated from raw env attributes.
    # This matches the calculation in Test_baseline_basestock.py and
    # test_trained_model_gnn.py (holding + backlog + ordering per agent per step).
    total_h = sum(acc["holding_costs"])
    total_b = sum(acc["backlog_costs"])
    total_o = sum(acc["ordering_costs"])
    m["total_cost"]    = total_h + total_b + total_o
    m["total_holding"] = total_h
    m["total_backlog"] = total_b
    m["total_ordering"] = total_o

    # Fill Rate (pooled across all retailer agents)
    total_placed = sum(acc["_orders_placed"][i]     for i in range(n_dcs, n_agents))
    total_fstock = sum(acc["_orders_from_stock"][i] for i in range(n_dcs, n_agents))
    m["fill_rate"]  = (total_fstock / total_placed * 100.0) if total_placed > 0 else 100.0
    m["lost_sales"] = total_placed - total_fstock

    # Avg inventory (time-averaged, then agent-averaged)
    avg_inv_per_agent = [acc["avg_inventory"][i] / T for i in range(n_agents)]
    m["avg_inventory"] = float(np.mean(avg_inv_per_agent))
    m["avg_inventory_per_agent"] = avg_inv_per_agent

    # Inventory Turnover Ratio  = Total Sales / Avg Inventory
    total_sales = sum(acc["total_sales"][i] for i in range(n_dcs, n_agents))
    retailer_avg_inv = (
        np.mean([acc["avg_inventory"][i] / T for i in range(n_dcs, n_agents)])
        if n_agents > n_dcs else 1.0
    )
    m["inventory_turnover"] = (
        total_sales / retailer_avg_inv if retailer_avg_inv > 1e-6 else 0.0
    )

    # On-Shelf Availability (retailers only)
    retailer_osa = [
        acc["osa_steps"][i] / T * 100.0
        for i in range(n_dcs, n_agents)
    ]
    m["osa"] = float(np.mean(retailer_osa)) if retailer_osa else 100.0

    return m


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    def __init__(self, args):
        self.args = args
        self.n_skus = 3

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp = args.experiment_name or f"robustness_{ts}"
        self.save_dir = Path(args.save_dir) / exp
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Multi-SKU Robustness Validation — Thesis Evaluation")
        print("=" * 70)
        print(f"  Episodes   : {args.num_episodes}")
        print(f"  Ep length  : {args.episode_length} days")
        print(f"  Seed       : {args.seed}")
        print(f"  Results dir: {self.save_dir}\n")

        # Detect torch device
        if args.cuda:
            try:
                import torch
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            except ImportError:
                self.device = None
        else:
            self.device = None

        # Will hold loaded DRL policies (loaded once per model)
        self._happo_policies  = None
        self._gnn_policies    = None
        self._gnn_adj         = None
        self._gnn_obs_dim     = None

        # Results: results[scenario_key][model_name] = list of episode dicts
        self.results = {}

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _ensure_happo(self, env):
        if self._happo_policies is not None:
            return
        if not self.args.happo_model_dir:
            raise RuntimeError("--happo_model_dir not provided")
        print("[Loading] Standard HAPPO policies...")
        self._happo_policies = _load_happo_policies(
            self.args.happo_model_dir, env, self.device, self.args
        )
        print(f"[OK] HAPPO: {len(self._happo_policies)} agents\n")

    def _ensure_gnn(self, env):
        if self._gnn_policies is not None:
            return
        if not self.args.gnn_model_dir:
            raise RuntimeError("--gnn_model_dir not provided")
        print("[Loading] GNN-HAPPO policies...")
        self._gnn_policies, self._gnn_adj, self._gnn_obs_dim = _load_gnn_policies(
            self.args.gnn_model_dir, env, self.device, self.args
        )
        print(f"[OK] GNN-HAPPO: {len(self._gnn_policies)} agents\n")

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(self):
        np.random.seed(self.args.seed)
        try:
            import torch
            torch.manual_seed(self.args.seed)
        except ImportError:
            pass

        env = create_base_env(self.args.config_path, self.args.episode_length)

        # Determine which models to run
        models_to_run = ["BaseStock"]
        if not self.args.heuristic_only:
            if self.args.happo_model_dir:
                models_to_run.append("HAPPO")
            if self.args.gnn_model_dir:
                models_to_run.append("GNN-HAPPO")

        ss_policy = SsPolicy(
            s_dc=self.args.s_dc,        S_dc=self.args.S_dc,
            s_retailer=self.args.s_retailer, S_retailer=self.args.S_retailer,
            n_dcs=env.n_dcs, n_agents=env.n_agents, n_skus=env.n_skus,
        )

        for sc_name, sc_cfg in SCENARIOS.items():
            print(f"\n{'='*70}")
            print(f"SCENARIO: {sc_name}")
            print(f"  SKU demand means : "
                  f"{[sc_cfg[f'SKU_{i}']['mean'] for i in range(self.n_skus)]}")
            print(f"  SKU demand stds  : "
                  f"{[sc_cfg[f'SKU_{i}']['std']  for i in range(self.n_skus)]}")
            print(f"{'='*70}\n")

            apply_scenario_to_env(env, sc_cfg)
            self.results[sc_name] = {}

            for model_name in models_to_run:
                print(f"  --- Model: {model_name} ---")
                ep_metrics = []

                for ep in range(self.args.num_episodes):
                    np.random.seed(self.args.seed + ep)
                    try:
                        import torch
                        torch.manual_seed(self.args.seed + ep)
                    except ImportError:
                        pass

                    if model_name == "BaseStock":
                        m = run_episode_heuristic(
                            env, ss_policy,
                            self.args.episode_length, self.n_skus
                        )
                    elif model_name == "HAPPO":
                        self._ensure_happo(env)
                        m = run_episode_happo(
                            env, self._happo_policies,
                            self.args.episode_length, self.n_skus, self.device
                        )
                    else:  # GNN-HAPPO
                        self._ensure_gnn(env)
                        m = run_episode_gnn(
                            env, self._gnn_policies,
                            self._gnn_adj, self._gnn_obs_dim,
                            self.args.episode_length, self.n_skus, self.device
                        )

                    ep_metrics.append(m)
                    if (ep + 1) % 20 == 0 or (ep + 1) == self.args.num_episodes:
                        avg_cost = np.mean([x["total_cost"] for x in ep_metrics])
                        avg_fr   = np.mean([x["fill_rate"]  for x in ep_metrics])
                        print(f"    Ep {ep+1:3d}/{self.args.num_episodes} "
                              f"| AvgCost={avg_cost:>12.1f} "
                              f"| FillRate={avg_fr:>6.1f}%")

                self.results[sc_name][model_name] = ep_metrics
                self._print_model_summary(model_name, ep_metrics)

        print("\n[OK] All evaluations complete!\n")

    def _print_model_summary(self, model_name, ep_metrics):
        costs = [m["total_cost"]          for m in ep_metrics]
        frs   = [m["fill_rate"]           for m in ep_metrics]
        turns = [m["inventory_turnover"]  for m in ep_metrics]
        osas  = [m["osa"]                 for m in ep_metrics]
        ls    = [m["lost_sales"]          for m in ep_metrics]
        print(f"  [{model_name}] Summary over {len(ep_metrics)} episodes:")
        print(f"    Cost         : {np.mean(costs):>12.2f} ± {np.std(costs):.2f}")
        print(f"    Fill Rate    : {np.mean(frs):>10.2f}% ± {np.std(frs):.2f}%")
        print(f"    Lost Sales   : {np.mean(ls):>10.2f} ± {np.std(ls):.2f}")
        print(f"    Inv Turnover : {np.mean(turns):>10.3f}")
        print(f"    OSA          : {np.mean(osas):>10.2f}%\n")

    # ------------------------------------------------------------------
    # Save CSV results
    # ------------------------------------------------------------------

    def save_csv(self):
        for sc_name, model_dict in self.results.items():
            for model_name, eps in model_dict.items():
                fname = (f"results_{sc_name}_{model_name.replace('-','_')}.csv")
                path  = self.save_dir / fname
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "Episode_Index", "Total_Cost", "Fill_Rate", "Lost_Sales",
                        "Avg_Inventory", "Inventory_Turnover", "OSA",
                        "Total_Holding", "Total_Backlog", "Total_Ordering"
                    ])
                    for i, m in enumerate(eps):
                        w.writerow([
                            i + 1,
                            round(m["total_cost"],         4),
                            round(m["fill_rate"],          4),
                            round(m["lost_sales"],         4),
                            round(m["avg_inventory"],      4),
                            round(m["inventory_turnover"], 4),
                            round(m["osa"],                4),
                            round(m["total_holding"],      4),
                            round(m["total_backlog"],      4),
                            round(m["total_ordering"],     4),
                        ])
        print(f"[OK] CSVs saved to: {self.save_dir}")

    def save_summary_json(self):
        summary = {}
        for sc_name, model_dict in self.results.items():
            summary[sc_name] = {}
            for model_name, eps in model_dict.items():
                def _s(key):
                    vals = [m[key] for m in eps]
                    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                            "min": float(np.min(vals)),  "max": float(np.max(vals))}
                summary[sc_name][model_name] = {
                    "total_cost":         _s("total_cost"),
                    "fill_rate":          _s("fill_rate"),
                    "lost_sales":         _s("lost_sales"),
                    "avg_inventory":      _s("avg_inventory"),
                    "inventory_turnover": _s("inventory_turnover"),
                    "osa":                _s("osa"),
                }
        path = self.save_dir / "robustness_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary JSON saved: {path.name}")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def generate_plots(self):
        print("\nGenerating thesis visualizations...")
        self._plot_cost_robustness()
        self._plot_service_level_stability()
        self._plot_pareto_frontier()
        self._plot_radar_chart()
        print(f"[OK] All plots saved to: {self.save_dir}\n")

    # ── Plot 1: Cost Robustness (Grouped Bar Chart) ────────────────────

    def _plot_cost_robustness(self):
        sc_names = list(self.results.keys())
        models   = list(next(iter(self.results.values())).keys())
        n_sc     = len(sc_names)
        n_models = len(models)

        x        = np.arange(n_sc)
        width    = 0.22
        offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#0F1923")
        ax.set_facecolor("#0F1923")

        for i, model in enumerate(models):
            means  = [np.mean([m["total_cost"] for m in self.results[sc][model]])
                      for sc in sc_names]
            stds   = [np.std( [m["total_cost"] for m in self.results[sc][model]])
                      for sc in sc_names]
            color  = MODEL_COLORS.get(model, "#AAAAAA")
            bars   = ax.bar(x + offsets[i], means, width,
                            label=model, color=color, alpha=0.88,
                            edgecolor="white", linewidth=0.6,
                            yerr=stds, capsize=4,
                            error_kw={"ecolor": "white", "elinewidth": 1.2})
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{mean:,.0f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [SCENARIOS[sc]["label"] for sc in sc_names],
            fontsize=11, color="white"
        )
        ax.set_ylabel("Mean Total Cost (per episode)", fontsize=12, color="white")
        ax.set_title("Plot 1 — Cost Robustness Across Demand Scenarios",
                     fontsize=14, fontweight="bold", color="white", pad=14)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#3A4A5A")
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", color="#3A4A5A", linewidth=0.7, alpha=0.6)
        ax.legend(fontsize=10, facecolor="#1A2A3A", edgecolor="#3A4A5A",
                  labelcolor="white")
        plt.tight_layout()
        plt.savefig(self.save_dir / "plot1_cost_robustness.png",
                    dpi=200, facecolor="#0F1923")
        plt.close()
        print("  [OK] Plot 1: cost_robustness.png")

    # ── Plot 2: Service Level Stability (Line + Bar hybrid) ───────────

    def _plot_service_level_stability(self):
        sc_names   = list(self.results.keys())
        models     = list(next(iter(self.results.values())).keys())
        sc_labels  = [SCENARIOS[sc]["short"] for sc in sc_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0F1923")
        ax.set_facecolor("#0F1923")

        for model in models:
            means = [np.mean([m["fill_rate"] for m in self.results[sc][model]])
                     for sc in sc_names]
            stds  = [np.std( [m["fill_rate"] for m in self.results[sc][model]])
                     for sc in sc_names]
            color  = MODEL_COLORS.get(model, "#AAAAAA")
            marker = MODEL_MARKERS.get(model, "o")
            ax.errorbar(sc_labels, means, yerr=stds,
                        label=model, color=color,
                        marker=marker, markersize=8,
                        linewidth=2.2, capsize=5,
                        elinewidth=1.4, markeredgecolor="white",
                        markeredgewidth=0.8)
            for j, (lbl, mean) in enumerate(zip(sc_labels, means)):
                ax.annotate(f"{mean:.1f}%",
                            xy=(j, mean), xytext=(0, 9),
                            textcoords="offset points",
                            ha="center", fontsize=8.5,
                            color=color, fontweight="bold")

        ax.axhline(95, color="red", linestyle="--", linewidth=1.6,
                   label="Target 95%", alpha=0.8)
        ax.set_ylabel("Mean Fill Rate (%)", fontsize=12, color="white")
        ax.set_title("Plot 2 — Service Level Stability Under Demand Stress",
                     fontsize=14, fontweight="bold", color="white", pad=14)
        ax.set_ylim([0, 115])
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#3A4A5A")
        ax.grid(color="#3A4A5A", linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=10, facecolor="#1A2A3A", edgecolor="#3A4A5A",
                  labelcolor="white")
        plt.tight_layout()
        plt.savefig(self.save_dir / "plot2_service_level_stability.png",
                    dpi=200, facecolor="#0F1923")
        plt.close()
        print("  [OK] Plot 2: service_level_stability.png")

    # ── Plot 3: Cost vs Fill Rate Pareto Scatter ───────────────────────

    def _plot_pareto_frontier(self):
        sc_names = list(self.results.keys())
        models   = list(next(iter(self.results.values())).keys())

        n_sc = len(sc_names)
        fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 6), sharey=False)
        fig.patch.set_facecolor("#0F1923")
        if n_sc == 1:
            axes = [axes]

        for ax, sc_name in zip(axes, sc_names):
            ax.set_facecolor("#0F1923")
            all_costs, all_frs = [], []

            for model in models:
                eps    = self.results[sc_name][model]
                costs  = [m["total_cost"] for m in eps]
                frs    = [m["fill_rate"]  for m in eps]
                color  = MODEL_COLORS.get(model, "#AAAAAA")
                marker = MODEL_MARKERS.get(model, "o")
                ax.scatter(frs, costs, c=color, marker=marker,
                           s=30, alpha=0.45, edgecolors="none",
                           label=f"{model} (episodes)")
                # Plot mean as large marker
                ax.scatter(np.mean(frs), np.mean(costs),
                           c=color, marker=marker,
                           s=180, edgecolors="white", linewidths=1.2,
                           zorder=10, label=f"{model} mean")
                all_costs.extend(costs)
                all_frs.extend(frs)

            # Pareto frontier
            combined = sorted(zip(all_frs, all_costs), reverse=True)
            pareto_fr, pareto_cost = [], []
            min_cost = float("inf")
            for fr, cost in combined:
                if cost < min_cost:
                    pareto_fr.append(fr)
                    pareto_cost.append(cost)
                    min_cost = cost
            if pareto_fr:
                ax.step(pareto_fr, pareto_cost,
                        where="post", color="#FFD700",
                        linewidth=2.2, linestyle="--",
                        label="Pareto Frontier", zorder=9)

            ax.set_xlabel("Fill Rate (%)", fontsize=11, color="white")
            ax.set_ylabel("Total Cost",    fontsize=11, color="white")
            ax.set_title(SCENARIOS[sc_name]["short"],
                         fontsize=12, fontweight="bold", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#3A4A5A")
            ax.grid(color="#3A4A5A", linewidth=0.6, alpha=0.45)
            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            seen, uniq_h, uniq_l = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    uniq_h.append(h)
                    uniq_l.append(l)
            ax.legend(uniq_h, uniq_l,
                      fontsize=8, facecolor="#1A2A3A",
                      edgecolor="#3A4A5A", labelcolor="white")

        fig.suptitle(
            "Plot 3 — Cost vs. Fill Rate Pareto Frontier (All Episodes)",
            fontsize=14, fontweight="bold", color="white", y=1.02
        )
        plt.tight_layout()
        plt.savefig(self.save_dir / "plot3_pareto_frontier.png",
                    dpi=200, facecolor="#0F1923", bbox_inches="tight")
        plt.close()
        print("  [OK] Plot 3: pareto_frontier.png")

    # ── Plot 4: Robustness Radar Chart ────────────────────────────────

    def _plot_radar_chart(self):
        """Radar chart with 4 normalised axes: Cost, Fill Rate, Turnover, Lost Sales.
        One line per model; averaged over ALL scenarios."""
        models = list(next(iter(self.results.values())).keys())
        axes_labels = ["Fill Rate\n(%)",
                       "Inv. Turnover\n(ratio)",
                       "OSA\n(%)",
                       "Low Lost\nSales"]

        N = len(axes_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        def _norm(raw_vals, minimize=False):
            """Normalise to [0, 1]. If minimize=True, lower=better → invert."""
            arr = np.array(raw_vals, dtype=float)
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-9:
                return np.ones_like(arr) * 0.5
            normed = (arr - lo) / (hi - lo)
            return 1.0 - normed if minimize else normed

        # Collect per-model means across all scenarios
        model_stats = {}  # model → {metric: mean}
        for model in models:
            all_fr, all_turn, all_osa, all_ls = [], [], [], []
            for sc_name, model_dict in self.results.items():
                if model not in model_dict:
                    continue
                eps = model_dict[model]
                all_fr   .extend([m["fill_rate"]          for m in eps])
                all_turn .extend([m["inventory_turnover"] for m in eps])
                all_osa  .extend([m["osa"]                for m in eps])
                all_ls   .extend([m["lost_sales"]         for m in eps])
            model_stats[model] = {
                "fill_rate":   np.mean(all_fr),
                "turnover":    np.mean(all_turn),
                "osa":         np.mean(all_osa),
                "lost_sales":  np.mean(all_ls),
            }

        # Normalise each axis across models
        fr_vals  = [model_stats[m]["fill_rate"]  for m in models]
        tu_vals  = [model_stats[m]["turnover"]   for m in models]
        osa_vals = [model_stats[m]["osa"]        for m in models]
        ls_vals  = [model_stats[m]["lost_sales"] for m in models]

        fr_norm  = _norm(fr_vals,  minimize=False)
        tu_norm  = _norm(tu_vals,  minimize=False)
        osa_norm = _norm(osa_vals, minimize=False)
        ls_norm  = _norm(ls_vals,  minimize=True)   # lower = better

        fig, ax = plt.subplots(figsize=(8, 8),
                                subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0F1923")
        ax.set_facecolor("#0F1923")

        for i, model in enumerate(models):
            values = [fr_norm[i], tu_norm[i], osa_norm[i], ls_norm[i]]
            values += values[:1]
            color  = MODEL_COLORS.get(model, "#AAAAAA")
            ax.plot(angles, values, color=color, linewidth=2.4,
                    marker=MODEL_MARKERS.get(model, "o"),
                    markersize=7, label=model)
            ax.fill(angles, values, color=color, alpha=0.12)

        ax.set_thetagrids(np.degrees(angles[:-1]), axes_labels,
                          fontsize=11, color="white")
        ax.tick_params(colors="white", labelsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=7, color="#AAAAAA")
        ax.grid(color="#3A4A5A", linewidth=0.8, alpha=0.6)
        ax.spines["polar"].set_color("#3A4A5A")
        ax.set_title("Plot 4 — Robustness Radar Chart\n(Normalised across models)",
                     fontsize=13, fontweight="bold", color="white",
                     y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  fontsize=10, facecolor="#1A2A3A",
                  edgecolor="#3A4A5A", labelcolor="white")
        plt.tight_layout()
        plt.savefig(self.save_dir / "plot4_radar_chart.png",
                    dpi=200, facecolor="#0F1923", bbox_inches="tight")
        plt.close()
        print("  [OK] Plot 4: radar_chart.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    evaluator = RobustnessEvaluator(args)
    evaluator.run()
    evaluator.save_csv()
    evaluator.save_summary_json()
    evaluator.generate_plots()
    print("=" * 70)
    print("Robustness Validation Complete!")
    print(f"All outputs saved to: {evaluator.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
