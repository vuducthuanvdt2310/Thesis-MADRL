    
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

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


        if self.all_args.algorithm_name == "happo":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        else:
            raise NotImplementedError

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, reward=None):
        # Create reward suffix if provided
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

    def restore(self):
        """Load saved models, handling both exact filenames and reward-suffixed filenames."""
        import glob
        
        # First check if we're loading from models/ subdirectory
        models_dir = os.path.join(self.model_dir, 'models')
        if os.path.exists(models_dir):
            load_dir = models_dir
            print(f"Loading models from: {load_dir}")
        else:
            load_dir = self.model_dir
            print(f"Loading models from: {load_dir}")
        
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                # Try exact filename first
                exact_path = os.path.join(load_dir, f'model_agent{agent_id}.pt')
                if os.path.exists(exact_path):
                    policy_model_state_dict = torch.load(exact_path)
                    self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
                else:
                    # Find best model with reward suffix
                    pattern = os.path.join(load_dir, f'model_agent{agent_id}_reward_*.pt')
                    model_files = glob.glob(pattern)
                    if model_files:
                        # Sort by reward (extract from filename)
                        best_model = max(model_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                        print(f"  Agent {agent_id}: Loading {os.path.basename(best_model)}")
                        policy_model_state_dict = torch.load(best_model)
                        self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
                    else:
                        raise FileNotFoundError(f"No model file found for agent {agent_id} in {load_dir}")
            else:
                # Load actor
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
                
                policy_actor_state_dict = torch.load(actor_path)
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                
                # Load critic
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
                
                policy_critic_state_dict = torch.load(critic_path)
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
                
                # Load optimizer states (try both with and without reward suffix)
                actor_opt_exact = os.path.join(load_dir, f'actor_optimizer_agent{agent_id}.pt')
                critic_opt_exact = os.path.join(load_dir, f'critic_optimizer_agent{agent_id}.pt')
                
                # Try to find optimizer files with reward suffix
                actor_opt_pattern = os.path.join(load_dir, f'actor_optimizer_agent{agent_id}_reward_*.pt')
                critic_opt_pattern = os.path.join(load_dir, f'critic_optimizer_agent{agent_id}_reward_*.pt')
                
                actor_opt_files = glob.glob(actor_opt_pattern)
                critic_opt_files = glob.glob(critic_opt_pattern)
                
                if os.path.exists(actor_opt_exact):
                    self.policy[agent_id].actor_optimizer.load_state_dict(torch.load(actor_opt_exact))
                elif actor_opt_files:
                    actor_opt_path = max(actor_opt_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                    self.policy[agent_id].actor_optimizer.load_state_dict(torch.load(actor_opt_path))
                
                if os.path.exists(critic_opt_exact):
                    self.policy[agent_id].critic_optimizer.load_state_dict(torch.load(critic_opt_exact))
                elif critic_opt_files:
                    critic_opt_path = max(critic_opt_files, key=lambda x: float(x.split('_reward_')[1].replace('.pt', '')))
                    self.policy[agent_id].critic_optimizer.load_state_dict(torch.load(critic_opt_path))
        
        print("✓ All models loaded successfully!")

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
