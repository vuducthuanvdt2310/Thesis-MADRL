"""
debug_action_generation.py
==========================
Standalone script to trace every variable involved in action computation
under Approach A (Demand + Residual).

Run from the project root:
    python debug_action_generation.py

No trained model needed — it uses freshly initialised weights.
The purpose is to validate that:
  1. reference_demand is extracted correctly from obs (non-zero, matches env range)
  2. raw_mean is bounded (not saturating at ±10+)
  3. adjustment is within ±max_residual  (= action_range / 3)
  4. target_mean = demand + adjustment moves per-step
  5. scaled_mean is always inside [0, 3] for retailers
  6. action_std varies per step (not collapsed to 0.05)
  7. sampled action varies step by step
"""

import sys
import os
import argparse
import numpy as np
import torch

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.multi_dc_env import MultiDCInventoryEnv
from algorithms.gnn.gnn_actor import GNNActor
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
from gymnasium import spaces

# ─── Config ──────────────────────────────────────────────────────────────────
N_STEPS   = 10      # number of env steps to trace
AGENT_ID  = 3       # 0-1 = DC, 2+ = Retailer  (set >= 2 to test Approach A)
PADDED_DIM = 28     # must match all_args.single_agent_obs_dim

# ─── Build minimal args ──────────────────────────────────────────────────────
args = argparse.Namespace(
    hidden_size         = 128,
    gain                = 0.01,
    use_orthogonal      = True,
    use_policy_active_masks = False,
    use_naive_recurrent_policy = True,
    use_recurrent_policy = False,
    recurrent_N         = 2,
    algorithm_name      = 'gnn_happo',
    gnn_type            = 'GAT',
    gnn_hidden_dim      = 128,
    gnn_num_layers      = 2,
    num_attention_heads = 4,
    gnn_dropout         = 0.1,
    use_residual        = True,
    single_agent_obs_dim = PADDED_DIM,
    std_x_coef          = 2.0,
    std_y_coef          = 1.5,     # MAX_STD
)

# ─── Environment ─────────────────────────────────────────────────────────────
print("=" * 65)
print("DEBUG: Action Generation — Approach A (Demand + Residual)")
print("=" * 65)
env = MultiDCInventoryEnv('configs/multi_dc_config.yaml')
obs_list, _ = env.reset()

n_agents = env.n_agents

# Pad obs to PADDED_DIM
def pad_obs(obs_list, padded_dim, n_agents):
    padded = np.zeros((1, n_agents, padded_dim), dtype=np.float32)
    for i, o in enumerate(obs_list):
        l = min(len(o), padded_dim)
        padded[0, i, :l] = o[:l]
    return padded

# Adjacency
adj_raw = build_supply_chain_adjacency(
    n_dcs=env.n_dcs, n_retailers=env.n_retailers,
    dc_assignments=env.dc_assignments
)
adj = normalize_adjacency(adj_raw)
adj_t = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)  # [1, n, n]

# ─── Actor ───────────────────────────────────────────────────────────────────
action_space = env.action_spaces[AGENT_ID]
obs_space    = env.observation_spaces[AGENT_ID]

actor = GNNActor(args, obs_space, action_space, n_agents)
actor.eval()

rnn_states = torch.zeros(1, args.recurrent_N, args.hidden_size)
masks      = torch.ones(1, 1)

# ─── Monkey-patch DiagGaussian.forward to print internals ────────────────────
from algorithms.utils.distributions import DiagGaussian
import torch.nn.functional as F

_original_forward = DiagGaussian.forward

def _debug_forward(self, x, available_actions=None, reference_demand=None):
    raw_mean = self.fc_mean(x)          # unbounded

    if reference_demand is not None:
        action_low   = self.action_low   if self.action_low   is not None else 0.0
        action_range = self.action_range if self.action_range is not None else 3.0
        max_residual = action_range / 3.0
        adjustment        = torch.tanh(raw_mean) * max_residual
        target_mean       = reference_demand + adjustment
        action_mean_scaled = torch.clamp(target_mean, min=action_low, max=action_low + action_range)

        print(f"\n  ── DiagGaussian internals ──────────────────────────────────────")
        print(f"  reference_demand   : {reference_demand.detach().numpy().flatten().round(4)}")
        print(f"  raw_mean (MLP out) : {raw_mean.detach().numpy().flatten().round(4)}")
        print(f"  tanh(raw_mean)     : {torch.tanh(raw_mean).detach().numpy().flatten().round(4)}")
        print(f"  max_residual       : {max_residual:.4f}  (= action_range/3 = {action_range}/3)")
        print(f"  adjustment         : {adjustment.detach().numpy().flatten().round(4)}")
        print(f"  target_mean        : {target_mean.detach().numpy().flatten().round(4)}")
        print(f"  scaled_mean        : {action_mean_scaled.detach().numpy().flatten().round(4)}")

    elif self.action_low is not None and self.action_range is not None:
        tanh_out           = torch.tanh(raw_mean)
        action_mean_scaled = self.action_low + ((tanh_out + 1.0) / 2.0) * self.action_range
        print(f"\n  ── DiagGaussian internals (DC tanh fallback) ───────────────────")
        print(f"  raw_mean (MLP out) : {raw_mean.detach().numpy().flatten()[:3].round(2)}")
        print(f"  scaled_mean        : {action_mean_scaled.detach().numpy().flatten()[:3].round(2)}")
    else:
        action_mean_scaled = raw_mean

    MIN_STD     = 0.05
    MAX_STD     = self.std_y_coef
    log_std_raw = self.fc_log_std(x)
    action_std  = torch.clamp(F.softplus(log_std_raw), min=MIN_STD, max=MAX_STD)

    print(f"  log_std_raw        : {log_std_raw.detach().numpy().flatten().round(4)}")
    print(f"  action_std         : {action_std.detach().numpy().flatten().round(4)}")

    from algorithms.utils.distributions import FixedNormal
    return FixedNormal(action_mean_scaled, action_std)

DiagGaussian.forward = _debug_forward

# ─── Run N steps ─────────────────────────────────────────────────────────────
print(f"\nTracing agent_id={AGENT_ID}  "
      f"({'Retailer' if AGENT_ID >= 2 else 'DC'}), "
      f"action_space=[{action_space.low[0]}, {action_space.high[0]}]\n")

for step in range(N_STEPS):
    print(f"\n{'='*65}")
    print(f"STEP {step + 1}")
    print(f"{'='*65}")

    obs_padded = pad_obs(obs_list, PADDED_DIM, n_agents)
    obs_t      = torch.tensor(obs_padded, dtype=torch.float32)  # [1, n, 28]

    # Show raw obs demand slice for this agent
    demand_raw_indices = [6, 13, 20]
    demand_norm  = obs_padded[0, AGENT_ID, demand_raw_indices]
    demand_phys  = demand_norm * 3.8
    print(f"  Raw obs demand (normalised) [{demand_raw_indices}]: {demand_norm.round(4)}")
    print(f"  Reference demand (physical) : {demand_phys.round(4)}")

    with torch.no_grad():
        actions, log_probs, rnn_states = actor.forward(
            obs_t, adj_t, AGENT_ID, rnn_states, masks,
            deterministic=False
        )

    action_np = actions.numpy().flatten()
    print(f"  sampled action             : {action_np.round(4)}")
    print(f"  action log_probs           : {log_probs.numpy().flatten().round(4)}")

    # Step env with dummy actions for all agents
    dummy_actions = [[
        np.clip(np.random.uniform(0, 3, 3), 0, 3).astype(np.float32)
        if i >= 2 else np.array([1000., 1000., 1000.], dtype=np.float32)
        for i in range(n_agents)
    ]]
    obs_list, rewards, dones, infos = env.step(dummy_actions)
    print(f"  env reward (agent {AGENT_ID})     : {rewards[0][AGENT_ID - 2]:.4f}")

print(f"\n{'='*65}")
print("Debug complete. Check that:")
print("  1. reference_demand changes each step (demand fluctuates)")
print("  2. raw_mean is NOT stuck at +10/+20 (fresh weights start near 0)")
print("  3. adjustment stays within ±1.0 for retailers (max_residual=1.0)")
print("  4. scaled_mean = demand + adjustment, clamped inside [0, 3]")
print("  5. action_std varies per step (not constant at 0.05 or 1.5)")
print("  6. sampled action varies each step")
print("=" * 65)
