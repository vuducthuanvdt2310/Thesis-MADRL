"""
optuna_search.py  — Clean redesign
==============================================
Lightweight Optuna optimizer for HAPPO hyperparameters.

DESIGN PHILOSOPHY (why this is fast):
  1. No CRunner — skips TensorBoard, CSV, model saving overhead entirely.
  2. No LSTM — uses pure MLP for fast forward/backward on CPU.
  3. Custom tight training loop: collect rollout → compute GAE → PPO update.
  4. Small network (hidden=64) during search; best params transfer to full training.
  5. 15 short train episodes + 3 eval episodes per trial → ~1-2 min/trial on CPU.

After the study finishes, copy the printed params into train_multi_dc_baseline.py.

Run:
    python optuna_search.py
"""

import os, sys, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path
from itertools import chain

# ── project imports ────────────────────────────────────────────────────────────
from config import get_config
from envs.multi_dc_env import MultiDCInventoryEnv
from algorithms.happo_policy import HAPPO_Policy as Policy
from utils.separated_buffer import SeparatedReplayBuffer

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS  (all tunable at the top)
# ══════════════════════════════════════════════════════════════════════════════

N_TRAIN_EPISODES = 20   # rollout episodes before evaluation
N_EVAL_EPISODES  = 3    # evaluation episodes (no gradient)
EPISODE_LENGTH   = 90   # days per episode  (90 = one quarter, enough for ordering signal)
N_TRIALS         = 50   # Optuna trials
TIMEOUT_HOURS    = 3    # wall-clock timeout for the whole study
HIDDEN_SIZE      = 64   # network width during search (64 is 2× faster than 128)

# Read actual agent count from environment to avoid mismatch
_dummy_env = MultiDCInventoryEnv(config_path='configs/multi_dc_config.yaml')
N_AGENTS = _dummy_env.n_agents
del _dummy_env

DEVICE           = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _t2n(x):
    return x.detach().cpu().numpy()


def make_args(trial):
    """Build a minimal args namespace for one Optuna trial."""
    parser = get_config()
    parser.set_defaults(
        env_name="MultiDC",
        num_agents=N_AGENTS,
        algorithm_name="happo",
        episode_length=EPISODE_LENGTH,
        seed=[0],
    )
    args = parser.parse_args([])

    # ── Speed flags (bypass argparse action='store_true' bug by direct assignment) ──
    args.hidden_size                = HIDDEN_SIZE
    args.use_naive_recurrent_policy = False   # MUST be False → MLP only
    args.use_recurrent_policy       = False
    args.recurrent_N                = 1
    args.data_chunk_length          = EPISODE_LENGTH
    args.use_centralized_V          = True
    args.use_obs_instead_of_state   = False

    # ── Hyperparameters suggested by Optuna ──
    args.lr           = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
    args.critic_lr    = trial.suggest_float("critic_lr",    1e-5, 5e-4, log=True)
    args.clip_param   = trial.suggest_float("clip_param",   0.10, 0.30)
    args.entropy_coef = trial.suggest_float("entropy_coef", 0.0,  0.05)
    args.gae_lambda   = trial.suggest_float("gae_lambda",   0.90, 0.99)
    args.gamma        = trial.suggest_float("gamma",        0.90, 0.99)
    args.ppo_epoch    = trial.suggest_int  ("ppo_epoch",    3,    10)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT ENV WRAPPER
# Wraps MultiDCInventoryEnv → returns numpy arrays compatible with Policy/Buffer
# ══════════════════════════════════════════════════════════════════════════════

class LiteEnvWrapper:
    """
    Minimal single-instance wrapper around MultiDCInventoryEnv.
    Returns obs/rewards correctly shaped per agent.
    """
    def __init__(self):
        self.env = MultiDCInventoryEnv(config_path='configs/multi_dc_config.yaml')
        self.n_agents = self.env.n_agents
        
        self.observation_space = [self.env.observation_spaces[i] for i in range(self.n_agents)]
        self.action_space = [self.env.action_spaces[i] for i in range(self.n_agents)]
        
        self.max_obs_dim = self.env.obs_dim_dc
        total_obs = self.max_obs_dim * self.n_agents
        
        from gymnasium import spaces
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

    def _pack_share(self, obs_dict):
        out = np.zeros((self.n_agents, self.max_obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            a = obs_dict[i]
            out[i, :len(a)] = a
        return out.reshape(1, -1)

    def _add_batch_dim(self, obs_dict):
        return {i: obs_dict[i][np.newaxis, :] for i in range(self.n_agents)}

    def reset(self):
        obs_dict = self.env.reset()
        share_obs = self._pack_share(obs_dict)
        return self._add_batch_dim(obs_dict), share_obs

    def step(self, actions_list_of_arrays):
        # actions_list_of_arrays is a list of [ (1, action_dim), ... ]
        action_dict = {i: actions_list_of_arrays[i][0] for i in range(self.n_agents)}
        obs_dict, rew_dict, done_dict, _ = self.env.step(action_dict)
        
        share_obs = self._pack_share(obs_dict)
        
        rewards = {i: np.array([[rew_dict[i]]], dtype=np.float32) for i in range(self.n_agents)}
        dones = {i: np.array([[done_dict[i]]], dtype=np.float32) for i in range(self.n_agents)}
        
        return self._add_batch_dim(obs_dict), share_obs, rewards, dones


# ══════════════════════════════════════════════════════════════════════════════
# ROLLOUT COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_rollout(env: LiteEnvWrapper, policies, trainers, buffers, args):
    n_agents   = env.n_agents
    rnn_states = np.zeros((1, n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    rnn_states_critic = np.zeros_like(rnn_states)
    masks      = np.ones((1, n_agents, 1), dtype=np.float32)

    obs_dict, share_obs_arr = env.reset()

    # Seed buffers with initial obs
    for agent_id in range(n_agents):
        buffers[agent_id].share_obs[0] = share_obs_arr.copy()
        buffers[agent_id].obs[0]       = obs_dict[agent_id].copy()

    total_reward = 0.0

    for step in range(args.episode_length):
        values, actions_list, log_probs_list = [], [], []
        rnn_new, rnn_c_new = [], []

        for agent_id in range(n_agents):
            trainers[agent_id].prep_rollout()
            value, action, log_prob, rnn_s, rnn_c = policies[agent_id].get_actions(
                share_obs_arr,                   # (1, total_obs)
                obs_dict[agent_id],              # (1, obs_dim)
                rnn_states[:, agent_id],         # (1, recN, hidden)
                rnn_states_critic[:, agent_id],
                masks[:, agent_id],
                None,                            # no available_actions
                agent_id=agent_id
            )
            values.append(_t2n(value))
            actions_list.append(_t2n(action))
            log_probs_list.append(_t2n(log_prob))
            rnn_new.append(_t2n(rnn_s))
            rnn_c_new.append(_t2n(rnn_c))

        # Step env
        next_obs_dict, next_share, rewards_dict, dones_dict = env.step(actions_list)
        
        total_reward += sum([float(rewards_dict[i][0,0]) for i in range(n_agents)])

        dones_env = np.all([dones_dict[i][0,0] for i in range(n_agents)])
        
        if dones_env:
            masks = np.zeros((1, n_agents, 1), dtype=np.float32)
        else:
            masks = np.ones((1, n_agents, 1), dtype=np.float32)

        # Reset RNN on episode end
        for agent_id in range(n_agents):
            if dones_env:
                rnn_new[agent_id]   = np.zeros_like(rnn_new[agent_id])
                rnn_c_new[agent_id] = np.zeros_like(rnn_c_new[agent_id])

        # Insert into buffers
        for agent_id in range(n_agents):
            buffers[agent_id].insert(
                next_share,
                next_obs_dict[agent_id],
                np.array(rnn_new)[agent_id:agent_id+1].reshape(1, args.recurrent_N, args.hidden_size),
                np.array(rnn_c_new)[agent_id:agent_id+1].reshape(1, *buffers[0].rnn_states_critic.shape[2:]),
                actions_list[agent_id],
                log_probs_list[agent_id],
                values[agent_id],
                rewards_dict[agent_id],
                masks[:, agent_id],
            )

        obs_dict      = next_obs_dict
        share_obs_arr = next_share
        rnn_states        = np.stack(rnn_new,   axis=1)  # (1, n_agents, recN, hidden)
        rnn_states_critic = np.stack(rnn_c_new, axis=1)

        if dones_env:
            break

    return total_reward


# ══════════════════════════════════════════════════════════════════════════════
# HAPPO UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def happo_update(trainers, buffers, args):
    n_agents = args.num_agents
    factor = np.ones((args.episode_length, args.n_rollout_threads, 1), dtype=np.float32)

    for agent_id in range(n_agents):
        trainers[agent_id].prep_training()
        buffers[agent_id].update_factor(factor)
        trainers[agent_id].train(buffers[agent_id])
        buffers[agent_id].after_update()


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE RETURNS  (GAE)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_returns(policies, trainers, buffers, args):
    for agent_id in range(args.num_agents):
        trainers[agent_id].prep_rollout()
        last_share_obs = buffers[agent_id].share_obs[-1]
        last_rnn_c     = buffers[agent_id].rnn_states_critic[-1]
        last_masks     = buffers[agent_id].masks[-1]
        next_value = trainers[agent_id].policy.get_values(
            last_share_obs, last_rnn_c, last_masks
        )
        next_value = _t2n(next_value)
        buffers[agent_id].compute_returns(next_value, trainers[agent_id].value_normalizer)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (pure greedy, no gradient)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(env: LiteEnvWrapper, policies, trainers, args, n_episodes=3):
    rewards_all = []
    for _ in range(n_episodes):
        obs_dict, share_obs_arr = env.reset()
        rnn_states = np.zeros((1, args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks      = np.ones((1, args.num_agents, 1), dtype=np.float32)
        ep_reward  = 0.0

        for step in range(args.episode_length):
            actions_list = []
            for agent_id in range(args.num_agents):
                trainers[agent_id].prep_rollout()
                action, rnn_s = policies[agent_id].act(
                    obs_dict[agent_id],
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    None,
                    deterministic=True,
                    agent_id=agent_id
                )
                actions_list.append(_t2n(action))
                rnn_states[0, agent_id] = _t2n(rnn_s)

            obs_dict, share_obs_arr, rewards_dict, dones_dict = env.step(actions_list)
            
            ep_reward += sum([float(rewards_dict[i][0,0]) for i in range(args.num_agents)])

            if np.all([dones_dict[i][0,0] for i in range(args.num_agents)]):
                break

        rewards_all.append(ep_reward)
    return float(np.mean(rewards_all))


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial):
    args = make_args(trial)
    args.num_agents = N_AGENTS

    print(f"\n[Trial {trial.number}] "
          f"lr={args.lr:.2e}  clip={args.clip_param:.2f}  "
          f"ent={args.entropy_coef:.4f}  gae_λ={args.gae_lambda:.3f}  "
          f"γ={args.gamma:.3f}  ppo_epoch={args.ppo_epoch}")

    try:
        env = LiteEnvWrapper()

        from algorithms.happo_trainer import HAPPO as Trainer
        policies, trainers, buffers = [], [], []

        for agent_id in range(N_AGENTS):
            share_obs_space = env.share_observation_space[agent_id]
            obs_space       = env.observation_space[agent_id]
            act_space       = env.action_space[agent_id]

            policy = Policy(args, obs_space, share_obs_space, act_space, device=DEVICE)
            trainer = Trainer(args, policy, device=DEVICE)
            buf = SeparatedReplayBuffer(args, obs_space, share_obs_space, act_space)

            policies.append(policy)
            trainers.append(trainer)
            buffers.append(buf)

        for ep in range(N_TRAIN_EPISODES):
            total_rew = collect_rollout(env, policies, trainers, buffers, args)
            compute_returns(policies, trainers, buffers, args)
            happo_update(trainers, buffers, args)

        eval_reward = evaluate(env, policies, trainers, args, n_episodes=N_EVAL_EPISODES)

        print(f"[Trial {trial.number}] eval reward = {eval_reward:.2f}")
        trial.report(eval_reward, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return eval_reward

    except optuna.TrialPruned:
        raise
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] FAILED with error: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("HAPPO Hyperparameter Optimization — Lightweight Direct Loop")
    print("=" * 70)
    print(f"  Training episodes/trial : {N_TRAIN_EPISODES}")
    print(f"  Eval    episodes/trial  : {N_EVAL_EPISODES}")
    print(f"  Episode length          : {EPISODE_LENGTH} days")
    print(f"  Network hidden size     : {HIDDEN_SIZE}")
    print(f"  Trials                  : {N_TRIALS}")
    print(f"  Timeout                 : {TIMEOUT_HOURS}h")
    print("=" * 70 + "\n")

    study = optuna.create_study(
        study_name="happo_inventory_lite",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0),
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_HOURS * 3600)

    pruned   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 70)
    print("Optimization Complete")
    print("=" * 70)
    print(f"  Total trials    : {len(study.trials)}")
    print(f"  Complete        : {len(complete)}")
    print(f"  Pruned          : {len(pruned)}")

    best = study.best_trial
    print(f"\nBest Trial #{best.number}  —  Eval Reward: {best.value:.2f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    p = best.params
    print("\n" + "=" * 70)
    print("Copy-paste into train_multi_dc_baseline.py → parser.set_defaults(...):")
    print("=" * 70)
    print(f"    lr           = {p.get('lr',           1e-4)},")
    print(f"    critic_lr    = {p.get('critic_lr',    1e-4)},")
    print(f"    clip_param   = {p.get('clip_param',   0.2)},")
    print(f"    entropy_coef = {p.get('entropy_coef', 0.01)},")
    print(f"    gae_lambda   = {p.get('gae_lambda',   0.95)},")
    print(f"    gamma        = {p.get('gamma',        0.95)},")
    print(f"    ppo_epoch    = {p.get('ppo_epoch',    15)},")
    print("=" * 70)

    try:
        print("\nGenerating plots...")
        plot_optimization_history(study).write_html("optuna_optimization_history.html")
        print("  Saved: optuna_optimization_history.html")
        plot_param_importances(study).write_html("optuna_param_importance.html")
        print("  Saved: optuna_param_importance.html")
    except Exception as e:
        print(f"  Plot error: {e}")
