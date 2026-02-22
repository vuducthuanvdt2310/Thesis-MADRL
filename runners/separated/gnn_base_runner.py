"""
GNN Runner for Multi-DC Environment

This runner handles Graph Neural Networks for GNN-HAPPO training.
Key difference from baseline: Passes adjacency matrix and structured
observations [batch, n_agents, obs_dim] through all forward calls.

All methods mirror the baseline CRunner exactly, with GNN-specific
observation handling added on top.
"""

import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule
from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency

def _t2n(x):
    return x.detach().cpu().numpy()

class GNNRunner(object):
    """
    Runner for GNN-HAPPO training with graph-aware policies.
    Mirrors CRunner (baseline) exactly, adding GNN-specific:
      - Adjacency matrix construction
      - Structured observations [batch, n_agents, obs_dim]
    """
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # Build and normalize adjacency matrix
        print("\nBuilding supply chain graph...")
        adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True)
        adj = normalize_adjacency(adj, method='symmetric')
        self.adj_tensor = torch.FloatTensor(adj).to(self.device)
        print(f"[OK] Graph created: {adj.shape[0]} nodes, {np.sum(adj > 0)} edges\n")

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.n_warmup_evaluations = self.all_args.n_warmup_evaluations
        self.n_no_improvement_thres = self.all_args.n_no_improvement_thres

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # GNN-specific imports
        from algorithms.gnn_happo_trainer import GNN_HAPPO as TrainAlgo
        from algorithms.gnn_happo_policy import GNN_HAPPO_Policy as Policy

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        n_agents=self.num_agents,
                        agent_id=agent_id,
                        device=self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id],
                          n_agents=self.num_agents, device=self.device)
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    # =========================================================================
    # Helper: Build structured observations [batch, n_agents, max_obs_dim]
    # =========================================================================
    def _build_structured_obs(self, step=None, obs_list=None):
        """
        Build structured observations [batch, n_agents, max_obs_dim] for GNN.
        
        Args:
            step: If provided, reads from self.buffer[aid].obs[step]
            obs_list: If provided (during eval), list of per-env obs arrays
        Returns:
            np.ndarray of shape [batch, n_agents, max_obs_dim]
        """
        max_obs_dim = max([self.envs.observation_space[i].shape[0] for i in range(self.num_agents)])
        
        if step is not None:
            batch_size = self.buffer[0].obs[step].shape[0]
            obs_structured = np.zeros((batch_size, self.num_agents, max_obs_dim), dtype=np.float32)
            for aid in range(self.num_agents):
                agent_obs = self.buffer[aid].obs[step]
                obs_dim = agent_obs.shape[-1]
                obs_structured[:, aid, :obs_dim] = agent_obs
        else:
            # obs_list: shape [n_envs, n_agents] of variable-length arrays
            n_envs = len(obs_list)
            obs_structured = np.zeros((n_envs, self.num_agents, max_obs_dim), dtype=np.float32)
            for env_idx in range(n_envs):
                for aid in range(self.num_agents):
                    obs = obs_list[env_idx][aid]
                    obs_dim = len(obs)
                    obs_structured[env_idx, aid, :obs_dim] = obs
        
        return obs_structured

    # =========================================================================
    # run() — mirrors CRunner.run() exactly
    # =========================================================================
    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        rewards_log = []
        inv_log = []
        actions_log = []
        demand_log = []
        overall_reward = []
        best_reward = float('-inf')
        best_bw = []
        record = 0
        start_episode = 0

        # Initialize CSV logging
        csv_path = os.path.join(str(self.run_dir), "progress.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("episode,steps,total_episode_reward\n")

        # Load training state if available
        print(f"[DEBUG] self.model_dir = {self.model_dir}")

        if self.model_dir is not None:
            state_path = os.path.join(self.model_dir, 'models', 'training_state.pt')
            print(f"[DEBUG] Checking path 1: {state_path}")
            print(f"[DEBUG] Path 1 exists: {os.path.exists(state_path)}")

            if not os.path.exists(state_path):
                state_path = os.path.join(self.model_dir, 'training_state.pt')
                print(f"[DEBUG] Checking path 2: {state_path}")
                print(f"[DEBUG] Path 2 exists: {os.path.exists(state_path)}")

            if os.path.exists(state_path):
                state = torch.load(state_path, map_location='cpu', weights_only=False)
                start_episode = state.get('episode', 0)
                best_reward = state.get('best_reward', float('-inf'))
                best_bw = state.get('best_bw', [])
                record = state.get('record', 0)
                print(f"[OK] Loaded training state from: {state_path}")
                print(f"[OK] Resuming training from episode {start_episode} with best reward {best_reward:.2f}")
            else:
                print(f"[WARNING] No training_state.pt found. Starting with best_reward = -inf")

        for episode in range(start_episode, episodes):
            episode_rewards = []
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.eval_interval == 0 and self.use_eval:
                re, bw_res = self.eval()
                print()
                print("Eval total episode reward: ", re, " Eval ordering fluctuation measurement (downstream to upstream): ", bw_res)

                if re > best_reward and episode > 0:
                    self.save(reward=re)

                # Log evaluation reward to TensorBoard
                self.writter.add_scalar("eval/total_episode_reward", re, total_num_steps)
                self.writter.add_scalar("eval/bullwhip_effect", np.mean(bw_res) if len(bw_res) > 0 else 0, total_num_steps)

                if re > best_reward and episode > 0:
                    training_state = {
                        'episode': episode,
                        'best_reward': re,
                        'best_bw': bw_res,
                        'record': record
                    }
                    torch.save(training_state, os.path.join(self.save_dir, "training_state.pt"))
                    print(f"[OK] Better model saved! Reward: {re:.2f} (previous best: {best_reward:.2f})")
                    best_reward = re
                    best_bw = bw_res
                    record = 0
                elif episode > self.n_warmup_evaluations:
                    record += 1
                    if record == self.n_no_improvement_thres:
                        print("Training finished because of no improvement for " + str(self.n_no_improvement_thres) + " evaluations")
                        return best_reward, best_bw

            self.warmup()
            if self.use_linear_lr_decay:
                self.trainer[0].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))

                available_actions = np.array([[None for agent_id in range(self.num_agents)] for info in infos])

                # Accumulate system reward for this step: sum of all agents, averaged over threads
                step_reward = np.sum(np.mean(rewards, axis=0))
                episode_rewards.append(step_reward)

                rewards_log.append(rewards)

                inv, demand, orders = self.envs.get_property()
                inv_log.append(inv)
                demand_log.append(demand)
                actions_log.append(orders)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                self.insert(data)

            # Episode-level CSV logging
            episode_total_reward = sum(episode_rewards)
            try:
                with open(csv_path, "a") as f:
                    f.write(f"{episode},{total_num_steps},{episode_total_reward:.4f}\n")
            except Exception as e:
                print(f"Error writing to CSV: {e}")

            # Compute returns and update network
            self.compute()
            train_infos = self.train()

            # Log training metrics to TensorBoard
            if episode % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)

            # Console log
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                threads_rew = [[] for i in range(self.n_rollout_threads)]
                threads_inv = [[] for i in range(self.n_rollout_threads)]
                threads_act = [[] for i in range(self.n_rollout_threads)]
                threads_demand = [[] for i in range(self.n_rollout_threads)]
                for i in range(len(rewards_log)):
                    for j in range(self.n_rollout_threads):
                        threads_rew[j].append(rewards_log[i][j])
                        threads_inv[j].append(inv_log[i][j])
                        threads_act[j].append(actions_log[i][j])
                        threads_demand[j].append(demand_log[i][j])

                overall_reward.append(np.mean(threads_rew))

                for t in range(len(threads_rew)):
                    rew = [[] for i in range(self.num_agents)]
                    inv = [[] for i in range(self.num_agents)]
                    act = [[] for i in range(self.num_agents)]
                    for i in range(len(threads_rew[t])):
                        for j in range(self.num_agents):
                            rew[j].append(threads_rew[t][i][j])
                            inv[j].append(threads_inv[t][i][j])
                            act[j].append(threads_act[t][i][j])
                    rew = [round(np.mean(l), 2) for l in rew]
                    inv = [round(np.mean(l), 2) for l in inv]
                    act = [round(np.mean(l), 2) for l in act]

                    if len(threads_demand[t]) > 0 and isinstance(threads_demand[t][0], dict):
                        print(f" --- Step {total_num_steps} Log ---")
                    else:
                        print("Reward for thread " + str(t+1) + ": " + str(rew) + " " + str(round(np.mean(rew), 2)) + "  Inventory: " + str(inv) + "  Order: " + str(act) + " Demand: " + str(np.mean(threads_demand[t], 0)))

                rewards_log = []
                inv_log = []
                actions_log = []
                demand_log = []

        print(f"\n{'='*70}")
        print(f"Training completed successfully!")
        print(f"Best reward achieved: {best_reward:.2f}")
        print(f"{'='*70}\n")
        return best_reward, best_bw

    # =========================================================================
    # warmup() — identical to CRunner.warmup()
    # =========================================================================
    def warmup(self):
        obs, available_actions = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            if self.buffer[agent_id].available_actions is not None:
                self.buffer[agent_id].available_actions[0] = None

    # =========================================================================
    # collect() — GNN version: uses structured obs + adjacency matrix
    # =========================================================================
    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        temp_actions_env = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []

        # Build structured observations [batch, n_agents, max_obs_dim] once
        obs_structured = self._build_structured_obs(step=step)

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            avail_actions = self.buffer[agent_id].available_actions[step] if self.buffer[agent_id].available_actions is not None else None

            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(
                    obs_structured,          # [batch, n_agents, obs_dim] for GNN critic
                    obs_structured,          # [batch, n_agents, obs_dim] for GNN actor
                    self.adj_tensor,         # adjacency matrix
                    agent_id,               # which agent
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step],
                    avail_actions)

            value_collector.append(_t2n(value))
            action_numpy = _t2n(action)
            action_collector.append(action_numpy)

            # Handle different action space types
            action_space_type = self.envs.action_space[agent_id].__class__.__name__

            if action_space_type == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action_numpy[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif action_space_type == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action_numpy], 1)
            elif action_space_type == 'Box':
                action_env = action_numpy
            else:
                raise NotImplementedError(f"Action space type {action_space_type} not supported")

            temp_actions_env.append(action_env)
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

        # [self.envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    # =========================================================================
    # insert() — identical to CRunner.insert()
    # =========================================================================
    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs, np.array(list(obs[:, agent_id])), rnn_states[:, agent_id],
                    rnn_states_critic[:, agent_id], actions[:, agent_id], action_log_probs[:, agent_id],
                    values[:, agent_id], rewards[:, agent_id], masks[:, agent_id])

    # =========================================================================
    # compute() — GNN version: builds structured obs for critic
    # =========================================================================
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            # Build structured observations from individual agent buffers
            # Use the last timestep's observations
            max_obs_dim = max([self.envs.observation_space[i].shape[0] for i in range(self.num_agents)])
            share_obs_batch = self.buffer[agent_id].share_obs[-1]
            batch_size = share_obs_batch.shape[0]

            obs_structured = np.zeros((batch_size, self.num_agents, max_obs_dim), dtype=np.float32)
            for aid in range(self.num_agents):
                agent_obs = self.buffer[aid].obs[-1]
                obs_dim = agent_obs.shape[-1]
                obs_structured[:, aid, :obs_dim] = agent_obs

            obs_structured_tensor = torch.FloatTensor(obs_structured).to(self.device)

            next_value = self.trainer[agent_id].policy.get_values(
                obs_structured_tensor,
                self.adj_tensor,
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1]
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    # =========================================================================
    # train() — GNN version: passes adj_tensor and agent_id
    # =========================================================================
    def train(self):
        train_infos = []
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            # GNN trainer expects adjacency matrix and agent_id
            train_info = self.trainer[agent_id].train(
                self.buffer[agent_id],
                self.adj_tensor,
                agent_id
            )

            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    # =========================================================================
    # eval() — GNN version: uses structured obs + adjacency matrix
    # =========================================================================
    @torch.no_grad()
    def eval(self):
        overall_reward = []
        eval_num = self.eval_envs.get_eval_num()

        for _ in range(eval_num):
            eval_obs, eval_available_actions = self.eval_envs.reset()

            eval_share_obs = []
            for o in eval_obs:
                eval_share_obs.append(list(chain(*o)))
            eval_share_obs = np.array(eval_share_obs)

            # Build structured observations [batch, n_agents, max_obs_dim]
            eval_obs_structured = self._build_structured_obs(obs_list=eval_obs)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                temp_actions_env = []

                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()

                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(
                            eval_obs_structured,      # [batch, n_agents, obs_dim]
                            self.adj_tensor,          # adjacency matrix
                            agent_id,                 # which agent
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            None,
                            deterministic=True)

                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    action = eval_actions.detach().cpu().numpy()

                    action_space_type = self.envs.action_space[agent_id].__class__.__name__

                    if action_space_type == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif action_space_type == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    elif action_space_type == 'Box':
                        action_env = action
                    else:
                        raise NotImplementedError(f"Action space type {action_space_type} not supported")

                    temp_actions_env.append(action_env)

                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    eval_actions_env.append(eval_one_hot_action_env)

                # Observe reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

                eval_share_obs = []
                for o in eval_obs:
                    eval_share_obs.append(list(chain(*o)))
                eval_share_obs = np.array(eval_share_obs)

                # Update structured observations
                eval_obs_structured = self._build_structured_obs(obs_list=eval_obs)

                eval_available_actions = None

                # Calculate system reward for this step (sum of all agents)
                step_reward = np.sum(np.mean(eval_rewards, axis=0))
                overall_reward.append(step_reward)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bw_res = self.eval_envs.get_eval_bw_res()
        return np.sum(overall_reward), bw_res

    # =========================================================================
    # save() — same as baseline
    # =========================================================================
    def save(self, reward=None):
        reward_suffix = f"_reward_{reward:.2f}" if reward is not None else ""

        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), os.path.join(self.save_dir, f"model_agent{agent_id}{reward_suffix}.pt"))
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), os.path.join(self.save_dir, f"actor_agent{agent_id}{reward_suffix}.pt"))
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), os.path.join(self.save_dir, f"critic_agent{agent_id}{reward_suffix}.pt"))
                if self.trainer[agent_id].policy.actor_optimizer is not None:
                    torch.save(self.trainer[agent_id].policy.actor_optimizer.state_dict(), os.path.join(self.save_dir, f"actor_optimizer_agent{agent_id}{reward_suffix}.pt"))
                if self.trainer[agent_id].policy.critic_optimizer is not None:
                    torch.save(self.trainer[agent_id].policy.critic_optimizer.state_dict(), os.path.join(self.save_dir, f"critic_optimizer_agent{agent_id}{reward_suffix}.pt"))

        # Cleanup old models
        if reward is not None:
            import glob
            try:
                patterns = ["actor_agent", "critic_agent", "actor_optimizer_agent", "critic_optimizer_agent", "model_agent"]

                for agent_id in range(self.num_agents):
                    for patterned_prefix in patterns:
                        search_pattern = os.path.join(self.save_dir, f"{patterned_prefix}{agent_id}_reward_*.pt")
                        files = glob.glob(search_pattern)

                        for file_path in files:
                            try:
                                filename = os.path.basename(file_path)
                                reward_str = filename.split('_reward_')[1].replace('.pt', '')
                                file_reward = float(reward_str)
                                current_reward_str = f"{reward:.2f}"
                                if reward_str == current_reward_str:
                                    continue
                                if file_reward < reward:
                                    print(f"  Cleanup: Removing suboptimal model {filename} ({file_reward:.2f} < {reward:.2f})")
                                    os.remove(file_path)
                            except (IndexError, ValueError):
                                continue
            except Exception as e:
                print(f"Warning: Model cleanup failed with error: {e}")

    # =========================================================================
    # restore() — same as baseline, with map_location fix
    # =========================================================================
    def restore(self):
        import glob

        models_dir = os.path.join(self.model_dir, 'models')
        if os.path.exists(models_dir):
            load_dir = models_dir
            print(f"Loading models from: {load_dir}")
        else:
            load_dir = self.model_dir
            print(f"Loading models from: {load_dir}")

        for agent_id in range(self.num_agents):
            if self.use_single_network:
                exact_path = os.path.join(load_dir, f'model_agent{agent_id}.pt')
                if os.path.exists(exact_path):
                    policy_model_state_dict = torch.load(exact_path, map_location=self.device)
                    self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
                else:
                    pattern = os.path.join(load_dir, f'model_agent{agent_id}_reward_*.pt')
                    model_files = glob.glob(pattern)
                    if model_files:
                        best_model = max(model_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                        print(f"  Agent {agent_id}: Loading {os.path.basename(best_model)}")
                        policy_model_state_dict = torch.load(best_model, map_location=self.device)
                        self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
                    else:
                        raise FileNotFoundError(f"No model file found for agent {agent_id} in {load_dir}")
            else:
                exact_actor_path = os.path.join(load_dir, f'actor_agent{agent_id}.pt')
                if os.path.exists(exact_actor_path):
                    actor_path = exact_actor_path
                else:
                    pattern = os.path.join(load_dir, f'actor_agent{agent_id}_reward_*.pt')
                    actor_files = glob.glob(pattern)
                    if actor_files:
                        actor_path = max(actor_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                        print(f"  Agent {agent_id}: Loading {os.path.basename(actor_path)}")
                    else:
                        raise FileNotFoundError(f"No actor file found for agent {agent_id} in {load_dir}")

                policy_actor_state_dict = torch.load(actor_path, map_location=self.device)
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)

                exact_critic_path = os.path.join(load_dir, f'critic_agent{agent_id}.pt')
                if os.path.exists(exact_critic_path):
                    critic_path = exact_critic_path
                else:
                    pattern = os.path.join(load_dir, f'critic_agent{agent_id}_reward_*.pt')
                    critic_files = glob.glob(pattern)
                    if critic_files:
                        critic_path = max(critic_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                    else:
                        raise FileNotFoundError(f"No critic file found for agent {agent_id} in {load_dir}")

                policy_critic_state_dict = torch.load(critic_path, map_location=self.device)
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

                actor_opt_pattern = os.path.join(load_dir, f'actor_optimizer_agent{agent_id}_reward_*.pt')
                critic_opt_pattern = os.path.join(load_dir, f'critic_optimizer_agent{agent_id}_reward_*.pt')

                actor_opt_files = glob.glob(actor_opt_pattern)
                critic_opt_files = glob.glob(critic_opt_pattern)

                if actor_opt_files:
                    actor_opt_path = max(actor_opt_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                    self.policy[agent_id].actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=self.device))

                if critic_opt_files:
                    critic_opt_path = max(critic_opt_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                    self.policy[agent_id].critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location=self.device))

        print("[OK] All models loaded successfully!")

    # =========================================================================
    # log_train() — same as CRunner.log_train()
    # =========================================================================
    def log_train(self, train_infos, total_num_steps):
        total_agent_reward = 0
        for agent_id in range(self.num_agents):
            agent_rew = np.mean(self.buffer[agent_id].rewards)
            train_infos[agent_id]["average_step_rewards"] = agent_rew
            total_agent_reward += agent_rew

            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

        # Log total system reward (sum of all agents)
        self.writter.add_scalar("system/total_average_step_reward", total_agent_reward, total_num_steps)
        self.writter.add_scalar("system/total_episode_reward_estimated", total_agent_reward * self.episode_length, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
