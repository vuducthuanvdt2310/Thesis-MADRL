import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from envs.multi_dc_env import MultiDCInventoryEnv
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC
from algorithms.happo_policy import HAPPO_Policy
from algorithms.gnn_happo_policy import GNN_HAPPO_Policy
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
from test_trained_model_mappo1 import MAPPOModelEvaluatorV1

class SsPolicy:
    def __init__(self, s_dc, S_dc, s_retailer, S_retailer, n_dcs, n_agents, n_skus):
        self.s_dc = s_dc
        self.S_dc = S_dc
        self.s_retailer = s_retailer
        self.S_retailer = S_retailer
        self.n_dcs = n_dcs
        self.n_agents = n_agents
        self.n_skus = n_skus

    def get_actions(self, env):
        actions = {}
        for agent_id in range(self.n_agents):
            order = np.zeros(self.n_skus, dtype=np.float32)
            for sku in range(self.n_skus):
                on_hand = float(env.inventory[agent_id][sku])
                pipeline_qty = sum(o['qty'] for o in env.pipeline[agent_id] if o['sku'] == sku)
                if agent_id < self.n_dcs:
                    owed = sum(env.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env.dc_assignments[agent_id])
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

def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_model_dir', type=str, default='results/14Apr_gnn_kaggle_vari/run_seed_1/models')
    parser.add_argument('--happo_model_dir', type=str, default='results/01Apr_base/run_seed_1/models')
    parser.add_argument('--mappo_model_dir', type=str, default='results/25Apr_MAPPO/run_seed_1/models')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=90)
    parser.add_argument('--happo_episode_length', type=int, default=120)
    parser.add_argument('--mappo_episode_length', type=int, default=130)
    parser.add_argument('--basestock_episode_length', type=int, default=110)
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
    parser.add_argument('--save_dir', type=str, default='evaluation_results/chap5_cost_breakdown')
    return parser.parse_args([])

def run_basestock(args):
    env = MultiDCInventoryEnv(config_path=args.config_path)
    env.max_days = args.basestock_episode_length
    policy = SsPolicy(args.s_dc, args.S_dc, args.s_retailer, args.S_retailer, env.n_dcs, env.n_agents, env.n_skus)
    
    np.random.seed(args.seed)
    _original_clip = env._clip_actions
    def _ss_clip_actions(acts):
        clipped = {}
        for aid, act in acts.items():
            if aid in env.dc_ids: clipped[aid] = np.clip(act, 0, 5000)
            else: clipped[aid] = np.maximum(0.0, np.array(act, dtype=np.float32))
        return clipped
    env._clip_actions = _ss_clip_actions

    metrics = {'holding': 0, 'backlog': 0, 'ordering': 0}
    env.reset()
    for step in range(args.basestock_episode_length):
        pre_step_prices = env.market_prices.copy()
        actions = policy.get_actions(env)
        try:
            _, _, _, _ = env.step(actions)
        except Exception:
            break
        executed_actions = {aid: np.maximum(0.0, np.array(a, dtype=np.float32)) for aid, a in actions.items()}
        for agent_id in range(env.n_agents):
            if agent_id < env.n_dcs:
                for sku in range(env.n_skus):
                    metrics['holding'] += env.inventory[agent_id][sku] * env.H_dc[agent_id][sku]
                    owed = sum(env.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env.dc_assignments[agent_id])
                    metrics['backlog'] += owed * env.B_dc[agent_id][sku]
                    act_sku = float(executed_actions[agent_id][sku])
                    if act_sku > 0:
                        metrics['ordering'] += env.C_fixed_dc[agent_id][sku] + pre_step_prices[sku] * act_sku
            else:
                r_idx = agent_id - env.n_dcs
                assigned_dc = env.retailer_to_dc[agent_id]
                for sku in range(env.n_skus):
                    metrics['holding'] += env.inventory[agent_id][sku] * env.H_retailer[r_idx][sku]
                    metrics['backlog'] += env.backlog[agent_id][sku] * env.B_retailer[r_idx][sku]
                    order_qty = float(executed_actions[agent_id][sku])
                    if order_qty > 0:
                        metrics['ordering'] += env.C_fixed_retailer[r_idx][sku]
    env._clip_actions = _original_clip
    return metrics

def run_happo(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    parser = get_config()
    parser.set_defaults(env_name="MultiDC", scenario_name="inventory_2echelon", num_agents=args.num_agents, episode_length=args.happo_episode_length, n_eval_rollout_threads=1, use_centralized_V=True, algorithm_name="happo", hidden_size=128, layer_N=2, use_ReLU=True, use_orthogonal=True, gain=0.01, recurrent_N=2, use_naive_recurrent_policy=True)
    all_args = parser.parse_known_args([])[0]
    env = DummyVecEnvMultiDC(all_args)
    
    policies = []
    model_dir = Path(args.happo_model_dir)
    for agent_id in range(args.num_agents):
        obs_space = env.observation_space[agent_id]
        agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
        suffixed = []
        for f in agent_files:
            if f.name == f"actor_agent{agent_id}.pt": continue
            try: suffixed.append((float(f.name.split('_reward_')[1].replace('.pt', '')), f))
            except: pass
        best_file = sorted(suffixed, key=lambda x: x[0], reverse=True)[0][1] if suffixed else (model_dir / f"actor_agent{agent_id}.pt" if (model_dir / f"actor_agent{agent_id}.pt").exists() else agent_files[0])
        
        state_dict = torch.load(str(best_file), map_location=device)
        saved_dim = state_dict['base.mlp.fc1.0.weight'].shape[1] if 'base.mlp.fc1.0.weight' in state_dict else None
        if saved_dim and saved_dim != obs_space.shape[0]:
            from gymnasium import spaces as gym_spaces
            obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(saved_dim,), dtype=np.float32)
            
        policy = HAPPO_Policy(all_args, obs_space, env.share_observation_space[agent_id], env.action_space[agent_id], device=device)
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)

    obs, _ = env.reset()
    rnn_states = np.zeros((1, args.num_agents, 2, 128), dtype=np.float32)
    masks = np.ones((1, args.num_agents, 1), dtype=np.float32)
    metrics = {'holding': 0, 'backlog': 0, 'ordering': 0}
    env_state = getattr(env, 'env_list', getattr(env, 'envs', None))[0]
    n_skus = env_state.n_skus
    
    for step in range(args.happo_episode_length):
        pre_prices = env_state.market_prices.copy()
        actions_env, raw_actions = [], {}
        for agent_id in range(args.num_agents):
            with torch.no_grad():
                obs_agent = np.stack(obs[:, agent_id])
                p_dim = policies[agent_id].obs_space.shape[0]
                if obs_agent.shape[1] < p_dim:
                    obs_agent = np.concatenate([obs_agent, np.zeros((obs_agent.shape[0], p_dim - obs_agent.shape[1]), dtype=np.float32)], axis=1)
                action, rnn_state = policies[agent_id].act(obs_agent, rnn_states[:, agent_id], masks[:, agent_id], deterministic=True, agent_id=agent_id)
            rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
            raw_action = (action.cpu().numpy() if isinstance(action, torch.Tensor) else action)[0]
            
            if agent_id < 2:
                _z, _lt, _n_ret = 1.65, 14, len(env_state.dc_assignments[agent_id])
                _zero_action = True
                for _sku in range(n_skus):
                    _mu, _sigma = float(env_state.demand_mean[_sku]) * _n_ret, float(env_state.demand_std[_sku]) * _n_ret
                    _out = _mu * _lt + _z * _sigma * np.sqrt(_lt)
                    _ip = float(env_state.inventory[agent_id][_sku]) - sum(env_state.dc_retailer_backlog[agent_id][r_id][_sku] for r_id in env_state.dc_assignments[agent_id]) + sum(o['qty'] for o in env_state.pipeline[agent_id] if o['sku'] == _sku)
                    if _ip < _out:
                        _zero_action = False
                        break
                if _zero_action: raw_action = np.zeros_like(raw_action)
            actions_env.append(raw_action)
            raw_actions[agent_id] = raw_action.copy()
            
        obs, _, _, _ = env.step([actions_env])
        exec_acts = env_state._clip_actions(raw_actions)
        
        for agent_id in range(args.num_agents):
            if agent_id < 2:
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_dc[agent_id][sku]
                    metrics['backlog'] += sum(env_state.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env_state.dc_assignments[agent_id]) * env_state.B_dc[agent_id][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_dc[agent_id][sku] + pre_prices[sku] * exec_acts[agent_id][sku]
            else:
                r_idx = agent_id - 2
                assigned_dc = env_state.retailer_to_dc[agent_id]
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                    metrics['backlog'] += env_state.backlog[agent_id][sku] * env_state.B_retailer[r_idx][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_retailer[r_idx][sku]
    return metrics

def run_mappo(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    parser = get_config()
    parser.set_defaults(env_name="MultiDC", scenario_name="inventory_2echelon", num_agents=args.num_agents, episode_length=args.mappo_episode_length, n_eval_rollout_threads=1, use_centralized_V=True, algorithm_name="mappo", hidden_size=128, layer_N=2, use_ReLU=True, use_orthogonal=True, gain=0.01, recurrent_N=2, use_naive_recurrent_policy=True)
    all_args = parser.parse_known_args([])[0]
    env = DummyVecEnvMultiDC(all_args)
    
    policies = []
    model_dir = Path(args.mappo_model_dir)
    for agent_id in range(args.num_agents):
        from algorithms.mappo_policy import MAPPO_Policy
        obs_space = env.observation_space[agent_id]
        agent_files = list(model_dir.glob(f"actor_agent{agent_id}*.pt"))
        suffixed = []
        for f in agent_files:
            if f.name == f"actor_agent{agent_id}.pt": continue
            try: suffixed.append((float(f.name.split('_reward_')[1].replace('.pt', '')), f))
            except: pass
        best_file = sorted(suffixed, key=lambda x: x[0], reverse=True)[0][1] if suffixed else (model_dir / f"actor_agent{agent_id}.pt" if (model_dir / f"actor_agent{agent_id}.pt").exists() else agent_files[0])
        
        state_dict = torch.load(str(best_file), map_location=device)
        saved_dim = state_dict['base.mlp.fc1.0.weight'].shape[1] if 'base.mlp.fc1.0.weight' in state_dict else None
        if saved_dim and saved_dim != obs_space.shape[0]:
            from gymnasium import spaces as gym_spaces
            obs_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(saved_dim,), dtype=np.float32)
            
        policy = MAPPO_Policy(all_args, obs_space, env.share_observation_space[agent_id], env.action_space[agent_id], device=device)
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)

    obs, _ = env.reset()
    rnn_states = np.zeros((1, args.num_agents, 2, 128), dtype=np.float32)
    masks = np.ones((1, args.num_agents, 1), dtype=np.float32)
    metrics = {'holding': 0, 'backlog': 0, 'ordering': 0}
    env_state = getattr(env, 'env_list', getattr(env, 'envs', None))[0]
    n_skus = env_state.n_skus
    
    for step in range(args.mappo_episode_length):
        pre_prices = env_state.market_prices.copy()
        actions_env, raw_actions = [], {}
        for agent_id in range(args.num_agents):
            with torch.no_grad():
                obs_agent = np.stack(obs[:, agent_id])
                p_dim = policies[agent_id].obs_space.shape[0]
                if obs_agent.shape[1] < p_dim:
                    obs_agent = np.concatenate([obs_agent, np.zeros((obs_agent.shape[0], p_dim - obs_agent.shape[1]), dtype=np.float32)], axis=1)
                action, rnn_state = policies[agent_id].act(obs_agent, rnn_states[:, agent_id], masks[:, agent_id], deterministic=True, agent_id=agent_id)
            rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
            raw_action = (action.cpu().numpy() if isinstance(action, torch.Tensor) else action)[0]
            
            if agent_id < 2:
                _z, _lt, _n_ret = 1.4, 7, len(env_state.dc_assignments[agent_id])
                _zero_action = True
                for _sku in range(n_skus):
                    _mu, _sigma = float(env_state.demand_mean[_sku]) * _n_ret, float(env_state.demand_std[_sku]) * _n_ret
                    _out = _mu * _lt + _z * _sigma * np.sqrt(_lt)
                    _ip = float(env_state.inventory[agent_id][_sku]) - sum(env_state.dc_retailer_backlog[agent_id][r_id][_sku] for r_id in env_state.dc_assignments[agent_id]) + sum(o['qty'] for o in env_state.pipeline[agent_id] if o['sku'] == _sku)
                    if _ip < _out:
                        _zero_action = False
                        break
                if _zero_action: raw_action = np.zeros_like(raw_action)
            actions_env.append(raw_action)
            raw_actions[agent_id] = raw_action.copy()
            
        obs, _, _, _ = env.step([actions_env])
        exec_acts = env_state._clip_actions(raw_actions)
        
        for agent_id in range(args.num_agents):
            if agent_id < 2:
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_dc[agent_id][sku]
                    metrics['backlog'] += sum(env_state.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env_state.dc_assignments[agent_id]) * env_state.B_dc[agent_id][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_dc[agent_id][sku] + pre_prices[sku] * exec_acts[agent_id][sku]
            else:
                r_idx = agent_id - 2
                assigned_dc = env_state.retailer_to_dc[agent_id]
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                    metrics['backlog'] += env_state.backlog[agent_id][sku] * env_state.B_retailer[r_idx][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_retailer[r_idx][sku]
    return metrics


def run_gnn(args):
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    parser = get_config()
    parser.set_defaults(env_name='MultiDC', scenario_name='inventory_2echelon', episode_length=args.episode_length, n_eval_rollout_threads=1, use_centralized_V=True, algorithm_name='gnn_happo')
    all_args = parser.parse_known_args([])[0]
    env = DummyVecEnvMultiDC(all_args)
    n_agents = env.num_agent if hasattr(env, 'num_agent') else 17
    
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=n_agents - 2, self_loops=True)
    adj_tensor = torch.FloatTensor(normalize_adjacency(adj, method='symmetric')).to(device)
    
    model_dir = Path(args.gnn_model_dir)
    agent0_files = sorted(model_dir.glob('actor_agent0*.pt'))
    sd = torch.load(str(agent0_files[0]), map_location='cpu')
    keys = list(sd.keys())
    if any('gnn_base.layers.0.weight' in k for k in keys): detected_type = 'GCN'; obs_dim = sd['gnn_base.layers.0.weight'].shape[0]
    elif any('gnn_base.layers.0.W' in k for k in keys): detected_type = 'GAT'; obs_dim = sd['gnn_base.layers.0.W'].shape[1]
    else: detected_type = args.gnn_type; obs_dim = max(env.observation_space[i].shape[0] for i in range(n_agents))
    
    parser = get_config()
    parser.add_argument('--gnn_type', type=str, default=detected_type)
    parser.add_argument('--gnn_hidden_dim', type=int, default=args.gnn_hidden_dim)
    parser.add_argument('--gnn_num_layers', type=int, default=args.gnn_num_layers)
    parser.add_argument('--num_attention_heads', type=int, default=args.num_attention_heads)
    parser.add_argument('--gnn_dropout', type=float, default=args.gnn_dropout)
    parser.add_argument('--use_residual', type=lambda x: str(x).lower() == 'true', default=args.use_residual)
    parser.add_argument('--critic_pooling', type=str, default=args.critic_pooling)
    parser.add_argument('--single_agent_obs_dim', type=int, default=obs_dim)
    parser.set_defaults(env_name='MultiDC', scenario_name='inventory_2echelon', num_agents=n_agents, use_centralized_V=True, algorithm_name='gnn_happo', hidden_size=128, layer_N=2, use_ReLU=True, use_orthogonal=True, gain=0.01, recurrent_N=2, use_naive_recurrent_policy=True, single_agent_obs_dim=obs_dim)
    all_args = parser.parse_known_args([])[0]
    
    from gymnasium import spaces as gym_spaces
    padded_obs = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    policies = []
    for agent_id in range(n_agents):
        all_f = list(model_dir.glob(f'actor_agent{agent_id}*.pt'))
        r_f = []
        for f in all_f:
            if f.name == f'actor_agent{agent_id}.pt': continue
            try: r_f.append((float(f.name.split('_reward_')[1].replace('.pt', '')), f))
            except: pass
        best_f = sorted(r_f, key=lambda x: x[0], reverse=True)[0][1] if r_f else (model_dir / f'actor_agent{agent_id}.pt' if (model_dir / f'actor_agent{agent_id}.pt').exists() else all_f[0])
        
        policy = GNN_HAPPO_Policy(all_args, padded_obs, env.share_observation_space[agent_id], env.action_space[agent_id], n_agents=n_agents, agent_id=agent_id, device=device)
        policy.actor.load_state_dict(torch.load(str(best_f), map_location=device))
        policy.actor.eval()
        policies.append(policy)

    obs, _ = env.reset()
    rnn_states = np.zeros((1, n_agents, 2, 128), dtype=np.float32)
    masks = np.ones((1, n_agents, 1), dtype=np.float32)
    metrics = {'holding': 0, 'backlog': 0, 'ordering': 0}
    env_state = getattr(env, 'env_list', getattr(env, 'envs', None))[0]
    n_skus = env_state.n_skus
    
    for step in range(args.episode_length):
        pre_prices = env_state.market_prices.copy()
        obs_struct = np.zeros((1, n_agents, obs_dim), dtype=np.float32)
        for aid in range(n_agents):
            raw = np.stack(obs[:, aid])
            obs_struct[0, aid, :raw.shape[1]] = raw[0]
            
        actions_env, raw_actions = [], {}
        for agent_id in range(n_agents):
            with torch.no_grad():
                action, rnn_state = policies[agent_id].act(obs_struct, adj_tensor, agent_id, rnn_states[:, agent_id], masks[:, agent_id], deterministic=False)
            rnn_states[:, agent_id] = rnn_state.cpu().numpy() if isinstance(rnn_state, torch.Tensor) else rnn_state
            raw_action = (action.cpu().numpy() if isinstance(action, torch.Tensor) else action)[0]
            
            if agent_id < 2:
                _z, _lt, _n_ret = 1.4, 7, len(env_state.dc_assignments[agent_id])
                _zero_action = True
                for _sku in range(n_skus):
                    _mu, _sigma = float(env_state.demand_mean[_sku]) * _n_ret, float(env_state.demand_std[_sku]) * _n_ret
                    _out = _mu * _lt + _z * _sigma * np.sqrt(_lt)
                    _ip = float(env_state.inventory[agent_id][_sku]) - sum(env_state.dc_retailer_backlog[agent_id][r_id][_sku] for r_id in env_state.dc_assignments[agent_id]) + sum(o['qty'] for o in env_state.pipeline[agent_id] if o['sku'] == _sku)
                    if _ip < _out:
                        _zero_action = False
                        break
                if _zero_action: raw_action = np.zeros_like(raw_action)
            actions_env.append(raw_action)
            raw_actions[agent_id] = raw_action.copy()
            
        obs, _, _, _ = env.step([actions_env])
        exec_acts = env_state._clip_actions(raw_actions)
        
        for agent_id in range(n_agents):
            if agent_id < 2:
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_dc[agent_id][sku]
                    metrics['backlog'] += sum(env_state.dc_retailer_backlog[agent_id][r_id][sku] for r_id in env_state.dc_assignments[agent_id]) * env_state.B_dc[agent_id][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_dc[agent_id][sku] + pre_prices[sku] * exec_acts[agent_id][sku]
            else:
                r_idx = agent_id - 2
                assigned_dc = env_state.retailer_to_dc[agent_id]
                for sku in range(n_skus):
                    metrics['holding'] += env_state.inventory[agent_id][sku] * env_state.H_retailer[r_idx][sku]
                    metrics['backlog'] += env_state.backlog[agent_id][sku] * env_state.B_retailer[r_idx][sku]
                    if exec_acts[agent_id][sku] > 0:
                        metrics['ordering'] += env_state.C_fixed_retailer[r_idx][sku]
    return metrics

def plot_cost_breakdown(results, save_dir):
    models = list(results.keys())
    holdings = [results[m]['holding'] for m in models]
    backlogs = [results[m]['backlog'] for m in models]
    orders = [results[m]['ordering'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p1 = ax.bar(models, holdings, label='Holding Cost')
    p2 = ax.bar(models, backlogs, bottom=holdings, label='Backlog Cost')
    p3 = ax.bar(models, orders, bottom=np.array(holdings) + np.array(backlogs), label='Ordering Cost')
    
    ax.set_ylabel("Cost ('000 VND)")
    ax.set_title('Detailed Cost Breakdown across Models')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = Path(save_dir) / "chap5_cost_breakdown.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved detailed cost breakdown chart to {save_path}")

def main():
    args = get_base_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print("Evaluating GNN-HAPPO...")
    gnn_metrics = run_gnn(args)
    print("Evaluating Standard HAPPO...")
    happo_metrics = run_happo(args)
    print("Evaluating MAPPO...")
    mappo_metrics = run_mappo(args)
    print("Evaluating (s,S) Heuristic...")
    bs_metrics = run_basestock(args)
    
    results = {
        'GNN-HAPPO': gnn_metrics,
        'Standard HAPPO': happo_metrics,
        'MAPPO': mappo_metrics,
        '(s,S) Heuristic': bs_metrics
    }
    
    plot_cost_breakdown(results, args.save_dir)

if __name__ == '__main__':
    main()
