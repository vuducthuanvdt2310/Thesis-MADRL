#!/usr/bin/env python
"""
Test/Evaluation Script for Trained HAPPO Models — 2 DC x 30 Retailer Scale
===========================================================================

Fully self-contained evaluator that inherits directly from
``test_trained_model.ModelEvaluator`` and inlines all topology-aware
overrides (formerly in ``test_evaluator_topology.py``).

Usage:
    python test_trained_model_happo_2x30.py \
        --model_dir results/happo_2x30/run_seed_1/models \
        --num_episodes 100 --episode_length 365
"""

import numpy as np
import torch
from pathlib import Path

from test_trained_model import ModelEvaluator, NumpyEncoder, parse_test_args
from config import get_config
from envs.env_wrappers import DummyVecEnvMultiDC

N_DCS = 2
N_RETAILERS = 30
N_AGENTS = N_DCS + N_RETAILERS  # 32
CONFIG_PATH = 'configs/multi_dc_2x30_config.yaml'


class HAPPOEvaluator2x30(ModelEvaluator):
    """HAPPO evaluator for the 2 DC x 30 Retailer environment.

    Inherits directly from ``ModelEvaluator`` and inlines every
    topology-aware override so the file is fully self-contained.
    """

    def __init__(self, args):
        # Set defaults *before* the parent __init__ calls _create_env / _load_models
        self.n_dcs = N_DCS
        self.policy_class_name = 'happo'
        super().__init__(args)

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------
    def _create_env(self):
        print('Creating evaluation environment (2x30 scale)...')
        parser = get_config()
        parser.set_defaults(
            env_name='MultiDC',
            scenario_name='inventory_2echelon_2x30',
            num_agents=N_AGENTS,
            episode_length=self.args.episode_length,
            n_eval_rollout_threads=1,
            use_centralized_V=True,
            algorithm_name='happo',
        )
        all_args = parser.parse_known_args([])[0]
        all_args.env_config_path = CONFIG_PATH
        env = DummyVecEnvMultiDC(all_args)

        self.args.num_agents = env.num_agent if hasattr(env, 'num_agent') else N_AGENTS
        self.n_dcs = N_DCS
        print(f'[OK] Environment: {self.args.num_agents} agents '
              f'({N_DCS} DCs + {N_RETAILERS} Retailers)')
        obs_dims = [env.observation_space[i].shape[0] for i in range(self.args.num_agents)]
        print(f'     Obs: DC={obs_dims[0]}D, Retailer={obs_dims[N_DCS]}D | '
              f'Action: {env.action_space[0].shape[0]}D\n')
        return env

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        """Load trained model weights for all agents.

        The policy class is chosen based on ``self.policy_class_name``
        (``'happo'`` or ``'mappo'``) instead of always importing
        ``HAPPO_Policy``.
        """
        print("Loading trained models...")

        algorithm_name = getattr(self.args, 'algorithm_name', 'happo')
        is_gnn = (algorithm_name == 'gnn_happo')

        model_dir = Path(self.args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        policies = []

        # Get environment config
        from config import get_config
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

        # Max obs dim for GNN padding
        max_obs_dim = max([self.env.observation_space[i].shape[0] for i in range(self.args.num_agents)])

        # --- Policy class selection based on self.policy_class_name ---
        if self.policy_class_name == 'mappo':
            from algorithms.mappo_policy import MAPPO_Policy as PolicyClass
        else:
            from algorithms.happo_policy import HAPPO_Policy as PolicyClass

        for agent_id in range(self.args.num_agents):
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

                policy = PolicyClass(
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
            print(f"  Loaded agent {agent_id} successfully")

        print(f"\n  All {self.args.num_agents} agent models loaded successfully!\n")
        return policies

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------
    def _run_episode(self, episode_num, save_trajectory=False):
        """Run a single evaluation episode.

        Full copy of the topology-aware ``_run_episode`` with every
        hardcoded ``2`` (used for DC-vs-retailer detection) replaced by
        ``self.n_dcs``.
        """
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
            'service_level': [0] * self.args.num_agents,
            '_demand_total': [0.0] * self.args.num_agents,
            '_demand_met':   [0.0] * self.args.num_agents,
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
                'inventory_skus': [[] for _ in range(self.args.num_agents)],
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
            raw_actions = {}

            if is_gnn:
                obs_structured = _build_gnn_obs(obs)

            for agent_id in range(self.args.num_agents):
                self.policies[agent_id].actor.eval()

                with torch.no_grad():
                    if is_gnn:
                        action, rnn_state = self.policies[agent_id].act(
                            obs_structured,
                            self.adj_tensor,
                            agent_id,
                            rnn_states[:, agent_id],
                            masks[:, agent_id],
                            None,
                            deterministic=True
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

                # -- DC IP-sufficiency guard (inference-time) --
                if agent_id < self.n_dcs and env_list_pre:
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
                # -----------------------------------------------

                actions_env.append(raw_action)
                raw_actions[agent_id] = raw_action.copy()

            # Step environment
            obs, rewards, dones, infos = self.env.step([actions_env])

            # Retrieve the EXACT (clipped) actions the env executed.
            env_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
            if env_list and len(env_list) > 0:
                env_state = env_list[0]
                executed_actions = env_state._clip_actions(raw_actions)
            else:
                executed_actions = {aid: raw_actions[aid] for aid in range(self.args.num_agents)}

            # Also store clipped actions for trajectory before the agent loop
            if save_trajectory:
                for agent_id in range(self.args.num_agents):
                    trajectory['actions'][agent_id].append(
                        np.array(executed_actions[agent_id], dtype=float).copy()
                    )

            if env_list and len(env_list) > 0:
                for agent_id in range(self.args.num_agents):
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

                    is_dc = agent_id < self.n_dcs

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
                                ordering_cost_step += env_state.C_fixed_dc[dc_idx][sku] + (price * order_qty)
                    else:
                        retailer_idx = agent_id - self.n_dcs
                        assigned_dc = env_state.retailer_to_dc[agent_id]
                        for sku in range(n_skus):
                            holding_cost_step += env_state.inventory[agent_id][sku] * env_state.H_retailer[retailer_idx][sku]
                            backlog_cost_step += env_state.backlog[agent_id][sku] * env_state.B_retailer[retailer_idx][sku]
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

                        # -- Demand logging --
                        if agent_id < self.n_dcs:  # DC: aggregate orders-placed from assigned retailers
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
                        if agent_id < self.n_dcs:
                            trajectory['norm_demand'][agent_id].append(np.zeros(n_skus, dtype=float))
                            trajectory['norm_inventory'][agent_id].append(np.zeros(n_skus, dtype=float))
                            trajectory['norm_order'][agent_id].append(np.zeros(n_skus, dtype=float))
                        else:
                            dm = getattr(env_state, 'demand_mean', np.ones(n_skus) * 1.5)
                            ds = getattr(env_state, 'demand_std', np.ones(n_skus) * 1.0)
                            demand_cap = np.maximum(dm + 3.0 * ds, 1e-6)
                            norm_d = (demand_vec / demand_cap).astype(float)
                            norm_inv = (np.array(inv_vec, dtype=float) / 150.0)
                            act_clip = np.array(executed_actions[agent_id], dtype=float)
                            norm_ord = np.clip(act_clip / 10.0, 0.0, 1.0)
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
            # Retailer: order-count fill rate
            if agent_id >= self.n_dcs:
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

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _calculate_statistics(self):
        """Calculate aggregate statistics from all episodes.

        Uses ``self.n_dcs`` instead of hardcoded ``2`` for agent-type detection.
        """
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

        for agent_id in range(self.args.num_agents):
            agent_type = "DC" if agent_id < self.n_dcs else "Retailer"
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

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def _save_metrics_csv(self):
        """Save episode metrics to CSV (uses ``self.n_dcs`` for retailer range)."""
        import csv

        results_path = self.save_dir / 'results_standard_happo_2x30.csv'
        compat_path  = self.save_dir / 'episode_metrics.csv'

        for csv_path in (results_path, compat_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode_Index', 'Total_Cost', 'Fill_Rate', 'Lost_Sales', 'Avg_Inventory',
                                 'Total_Holding_Cost', 'Total_Backlog_Cost', 'Total_Ordering_Cost'])

                for ep_num, metrics in enumerate(self.episode_metrics):
                    fill_rate = float(np.mean(metrics['service_level']))
                    total_placed     = sum(metrics['_orders_placed'][aid]     for aid in range(self.n_dcs, self.args.num_agents))
                    total_from_stock = sum(metrics['_orders_from_stock'][aid] for aid in range(self.n_dcs, self.args.num_agents))
                    lost_sales = total_placed - total_from_stock
                    avg_inventory = float(np.mean(metrics['avg_inventory']))
                    total_holding = float(np.sum(metrics['holding_costs']))
                    total_backlog = float(np.sum(metrics['backlog_costs']))
                    total_ordering = float(np.sum(metrics['ordering_costs']))
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

        print(f"  Saved metrics CSV: {results_path.name}  (also {compat_path.name})")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def _plot_service_levels(self, stats):
        """Plot service level comparison (uses ``self.n_dcs`` for bar colouring)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        agents = list(stats['per_agent'].keys())
        service_levels = [stats['per_agent'][a]['service_level'] for a in agents]

        n_agents = len(agents)
        colors = ['#2E86AB'] * self.n_dcs + ['#A23B72'] * (n_agents - self.n_dcs)
        bars = ax.bar(agents, service_levels, color=colors, alpha=0.8, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')

        ax.set_ylabel('Service Level (%)', fontsize=12)
        ax.set_title('Service Level by Agent (Demand Fill-Rate for Retailers)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'service_levels.png', dpi=300)
        plt.close()

    def _plot_dc_inventory_fluctuation(self):
        """Plot per-SKU and total inventory fluctuation for each DC (Episode 1).

        Uses ``self.n_dcs`` instead of ``i < 2`` for DC detection.
        """
        import matplotlib.pyplot as plt

        traj = self.detailed_trajectory
        if not traj or 'inventory_skus' not in traj:
            return

        _base_list = getattr(self.env, 'env_list', getattr(self.env, 'envs', None))
        n_skus = getattr(_base_list[0], 'n_skus', 3) if _base_list else 3

        days = np.arange(1, self.args.episode_length + 1)
        dc_ids = [i for i in range(self.args.num_agents) if i < self.n_dcs]
        n_dcs = len(dc_ids)
        n_cols = n_skus + 1

        fig, axes = plt.subplots(n_dcs, n_cols,
                                 figsize=(5 * n_cols, 4 * n_dcs),
                                 squeeze=False)
        fig.suptitle('DC Inventory Fluctuation Over Steps (Episode 1)',
                     fontsize=15, fontweight='bold', y=1.01)

        sku_colors = ['#2196F3', '#4CAF50', '#FF9800']
        total_color = '#9C27B0'

        for row, dc_id in enumerate(dc_ids):
            dc_label = f'DC_{dc_id}'
            inv_skus = np.array(traj['inventory_skus'][dc_id], dtype=float)
            inv_total = np.array(traj['inventory'][dc_id], dtype=float)

            for sku in range(n_skus):
                ax = axes[row][sku]
                ax.plot(days, inv_skus[:, sku],
                        color=sku_colors[sku % len(sku_colors)],
                        linewidth=1.5, alpha=0.85)
                ax.fill_between(days, inv_skus[:, sku], alpha=0.15,
                                color=sku_colors[sku % len(sku_colors)])
                ax.set_title(f'{dc_label} -- SKU {sku}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Inventory', fontsize=10)
                ax.set_xlabel('Step (day)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(days[0], days[-1])

            ax_total = axes[row][n_cols - 1]
            ax_total.plot(days, inv_total, color=total_color, linewidth=2.0, alpha=0.9, label='Total')
            ax_total.fill_between(days, inv_total, alpha=0.12, color=total_color)
            ax_total.axhline(np.mean(inv_total), color='red', linestyle='--',
                             linewidth=1.2, label=f'Mean: {np.mean(inv_total):.0f}')
            ax_total.set_title(f'{dc_label} -- Total Inventory', fontsize=11, fontweight='bold')
            ax_total.set_ylabel('Total Inventory', fontsize=10)
            ax_total.set_xlabel('Step (day)', fontsize=10)
            ax_total.legend(fontsize=9)
            ax_total.grid(True, alpha=0.3)
            ax_total.set_xlim(days[0], days[-1])

        plt.tight_layout()
        out = self.save_dir / 'dc_inventory_fluctuation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved DC inventory fluctuation plot: {out.name}')

    def _plot_retailer_inventory_fluctuation(self):
        """Plot inventory fluctuation for all retailers (Episode 1).

        Uses ``self.n_dcs`` instead of ``i >= 2`` for retailer detection.
        """
        import matplotlib.pyplot as plt

        traj = self.detailed_trajectory
        if not traj:
            return

        days = np.arange(1, self.args.episode_length + 1)
        retailer_ids = [i for i in range(self.args.num_agents) if i >= self.n_dcs]
        n_retailers = len(retailer_ids)

        inv_matrix = np.array([traj['inventory'][rid] for rid in retailer_ids], dtype=float)

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
                             alpha=0.2, color='#1565C0', label='+-1 std')
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
        print(f'  Saved retailer inventory fluctuation plot: {out.name}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _print_summary(self, stats):
        print('\n' + '=' * 70)
        print('HAPPO Evaluation Summary \u2014 2\u00d730 Scale')
        print('=' * 70)
        print(f"Episodes: {stats['num_episodes']} | Length: {stats['episode_length']} days")
        print(f"Avg reward: {stats['total_reward']['mean']:.2f} (+/-{stats['total_reward']['std']:.2f})")
        print(f"Avg cost  : {stats['total_cost']['mean']:.2f} (+/-{stats['total_cost']['std']:.2f})")
        print('-' * 70)
        print(f"{'Agent':<14} {'Avg Cost':>12} {'Holding':>12} {'Backlog':>12} {'Svc%':>8}")
        print('-' * 70)
        for agent, data in stats['per_agent'].items():
            print(f"{agent:<14} {data['avg_cost']:>12.1f} "
                  f"{data['avg_holding_cost']:>12.1f} {data['avg_backlog_cost']:>12.1f} "
                  f"{data['service_level']:>7.1f}%")
        print('=' * 70)


def main():
    args = parse_test_args()
    evaluator = HAPPOEvaluator2x30(args)
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == '__main__':
    main()
