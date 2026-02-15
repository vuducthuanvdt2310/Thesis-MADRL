import numpy as np
import gymnasium as gym
from gymnasium import spaces
#from envs.serial import Env
from envs.net_2x3 import Env
from envs.multi_dc_env import MultiDCInventoryEnv


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.env_list = [Env() for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
         inv = [env.get_inventory() for env in self.env_list]
         demand = [env.get_demand() for env in self.env_list]
         orders = [env.get_orders() for env in self.env_list]
         return inv, demand, orders
         
    def reset(self):
        obs = [env.reset() for env in self.env_list]
        return np.stack(obs), None

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


# single env
class DummyVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.env_list = [Env() for i in range(1)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent_num in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = self.signal_obs_dim  # 单个智能体的观测维度
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):

        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = [env.reset(train = False) for env in self.env_list]
        return np.stack(obs), None
    
    def get_eval_bw_res(self):
        res = self.env_list[0].get_eval_bw_res()
        return res
    
    def get_eval_num(self):
        eval_num = self.env_list[0].get_eval_num()
        return eval_num
        
    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


# ============================================================================
# Multi-DC Environment Wrappers (for heterogeneous agents)
# ============================================================================

class SubprocVecEnvMultiDC(object):
    """Vectorized environment wrapper for Multi-DC inventory environment."""
    
    def __init__(self, all_args):
        """
        Initialize parallel Multi-DC environments.
        
        Args:
            all_args: Configuration arguments with n_rollout_threads
        """
        # Create parallel environments
        self.env_list = [MultiDCInventoryEnv(config_path='configs/multi_dc_config.yaml') 
                        for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        
        # Get environment properties from first env
        self.num_agent = self.env_list[0].n_agents  # 5 agents
        
        # Multi-DC has heterogeneous agents - DCs vs Retailers
        self.n_dcs = self.env_list[0].n_dcs  # 2
        self.n_retailers = self.env_list[0].n_retailers  # 3
        
        # Note: This is continuous action space
        self.discrete_action_space = False
        self.discrete_action_input = False
        self.force_discrete_action = False
        
        
        # === UNIFORM ACTION SPACES (All agents 6D) ===
        # DC agents: use first 3 dimensions, last 3 ignored
        # Retailer agents: use all 6 dimensions
        # This uniformity is required for HAPPO runner compatibility
        
        self.action_dim = 6  # Uniform for all agents
        
        # Initialize empty lists
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        for agent_id in range(self.num_agent):
            # Observations remain heterogeneous
            if agent_id < self.n_dcs:
                obs_dim = 30  # DC observation (increased from 27)
            else:
                obs_dim = 36  # Retailer observation (reduced from 42)
            
            self.observation_space.append(
                spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            )
            
            # All agents have same 6D action space
            self.action_space.append(
                spaces.Box(low=0, high=50, shape=(self.action_dim,), dtype=np.float32)
            )
        # Shared observation space (concatenate all observations)
        total_obs_dim = 30 * self.n_dcs + 36 * self.n_retailers  # 168 (30*2 + 36*3)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(total_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
    
    def step(self, actions):
        """
        Step all parallel environments.
        
        Args:
            actions: List of action dicts, one per parallel env
                Each dict: {agent_id: action_array}
        
        Returns:
            obs, rewards, dones, infos (all stacked)
        """
        results = []
        for env, action_list in zip(self.env_list, actions):
            # Convert list of actions to dict {agent_id: action_array}
            action_dict = {agent_id: action_list[agent_id] for agent_id in range(self.num_agent)}
            result = env.step(action_dict)
            results.append(result)
        
        # Unpack results
        obs, rews, dones, infos = zip(*results)
        
        # Convert to numpy arrays
        # obs: list of dicts -> array of shape (n_envs, n_agents, obs_dim)
        obs_arrays = []
        rew_arrays = []
        done_arrays = []
        
        for env_idx in range(self.num_envs):
            env_obs = []
            env_rews = []
            env_dones = []
            
            for agent_id in range(self.num_agent):
                env_obs.append(obs[env_idx][agent_id])
                env_rews.append(rews[env_idx][agent_id])
                env_dones.append(dones[env_idx][agent_id])
            
            obs_arrays.append(env_obs)
            rew_arrays.append(env_rews)
            done_arrays.append(env_dones)
        
        # Use dtype=object for heterogeneous agent observations
        # Rewards need shape (n_envs, n_agents, 1) for buffer
        rewards_reshaped = np.expand_dims(np.array(rew_arrays), axis=-1)
        return np.array(obs_arrays, dtype=object), rewards_reshaped, np.array(done_arrays), infos
    
    def reset(self):
        """Reset all parallel environments."""
        obs_list = [env.reset() for env in self.env_list]
        
        # Convert dict observations to arrays
        obs_arrays = []
        for env_obs_dict in obs_list:
            env_obs = [env_obs_dict[agent_id] for agent_id in range(self.num_agent)]
            obs_arrays.append(env_obs)
        
        # Use dtype=object for heterogeneous agent observations
        return np.array(obs_arrays, dtype=object), None
    
    def close(self):
        """Close all environments."""
        pass
    
    def render(self, mode="rgb_array"):
        """Render (no-op for this environment)."""
        pass
    
    def get_property(self):
        """Get environment properties (inventory, demand, orders)."""
        # Convert inventory from dict {agent_id: np.array(n_skus)} 
        # to dict {agent_id: total_inventory} for logging
        inv = []
        for env in self.env_list:
            # Sum inventory across SKUs for each agent
            inv_dict = {agent_id: np.sum(inv_array) for agent_id, inv_array in env.inventory.items()}
            inv.append(inv_dict)
        
        # Multi-DC tracks actions internally, not as separate orders/demand
        # Multi-DC tracks actions internally, not as separate orders/demand
        # Return dictionaries with agent_id keys for logging compatibility
        demand = [{agent_id: 0 for agent_id in range(self.num_agent)} for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders
    
    def get_eval_num(self):
        """Return number of evaluation episodes to run."""
        # Return number of evaluation episodes (default: 5)
        # This determines how many full episodes to run during evaluation
        return 5
    
    def get_eval_bw_res(self):
        """Return evaluation bullwhip results (for compatibility)."""
        # Multi-DC doesn't track bullwhip effect in same way
        # Return empty list for compatibility
        return []


class DummyVecEnvMultiDC(object):
    """Single (non-parallel) environment wrapper for Multi-DC."""
    
    def __init__(self, all_args):
        """Initialize single Multi-DC environment for evaluation."""
        self.env_list = [MultiDCInventoryEnv(config_path='configs/multi_dc_config.yaml')]
        self.num_envs = 1
        
        self.num_agent = self.env_list[0].n_agents
        self.n_dcs = self.env_list[0].n_dcs
        self.n_retailers = self.env_list[0].n_retailers
        
        self.discrete_action_space = False
        self.discrete_action_input = False
        self.force_discrete_action = False
        
        
        # Configure spaces (uniform 6D actions)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        self.action_dim = 6  # Uniform for all agents
        
        for agent_id in range(self.num_agent):
            # Heterogeneous observations
            if agent_id < self.n_dcs:
                obs_dim = 30
            else:
                obs_dim = 36  # Reduced from 42
            
            self.observation_space.append(
                spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            )
            # Uniform 6D actions
            self.action_space.append(
                spaces.Box(low=0, high=50, shape=(self.action_dim,), dtype=np.float32)
            )
        
        total_obs_dim = 30 * self.n_dcs + 36 * self.n_retailers  # 168
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(total_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
    
    def step(self, actions):
        """Step the single environment."""
        env = self.env_list[0]
        action_list = actions[0]  # First (and only) env - this is a LIST
        
        # Convert list to dict {agent_id: action_array}
        action_dict = {agent_id: action_list[agent_id] for agent_id in range(self.num_agent)}
        
        obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)
        
        # Convert to arrays
        obs = [[obs_dict[i] for i in range(self.num_agent)]]
        rews = [[rew_dict[i] for i in range(self.num_agent)]]
        dones = [[done_dict[i] for i in range(self.num_agent)]]
        
        # Use dtype=object for heterogeneous observations
        # Rewards need shape (n_envs, n_agents, 1) for buffer
        rewards_reshaped = np.expand_dims(np.array(rews), axis=-1)
        return np.array(obs, dtype=object), rewards_reshaped, np.array(dones), [info_dict]
    
    def reset(self):
        """Reset the single environment."""
        env = self.env_list[0]
        obs_dict = env.reset()
        
        obs = [[obs_dict[i] for i in range(self.num_agent)]]
        # Use dtype=object for heterogeneous observations
        return np.array(obs, dtype=object), None
    
    def close(self):
        """Close environment."""
        pass
    
    def render(self, mode="rgb_array"):
        """Render (no-op)."""
        pass
    
    def get_property(self):
        """Get environment properties."""
        env = self.env_list[0]
        inv = [env.inventory]
        demand = [{}]
        orders = [env.get_orders()]
        return inv, demand, orders
    
    def get_eval_num(self):
        """Return number of evaluation episodes to run."""
        # Return number of evaluation episodes (default: 5)
        # This must match the eval_episodes config in train_multi_dc.py
        return 5
    
    def get_eval_bw_res(self):
        """Return evaluation bullwhip results (for compatibility)."""
        return []

