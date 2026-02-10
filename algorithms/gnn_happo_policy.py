"""
GNN-HAPPO Policy Wrapper.

This wraps GNNActor and GNNCritic to maintain compatibility with 
the existing HAPPO training infrastructure.

Key differences from standard HAPPO_Policy:
1. Uses GNNActor and GNNCritic instead of standard Actor/Critic
2. Requires adjacency matrix in all forward passes
3. Handles heterogeneous agent observations (different obs dims for DC vs Retailer)
"""

import torch
from algorithms.gnn.gnn_actor import GNNActor
from algorithms.gnn.gnn_critic import GNNCritic
from utils.util import update_linear_schedule


class GNN_HAPPO_Policy:
    """
    GNN-HAPPO Policy class. Wraps GNN actor and critic networks.
    
    This maintains the same interface as HAPPO_Policy for compatibility
    with existing training loop.
    """
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, n_agents,
                 agent_id=0, device=torch.device("cpu")):
        """
        Args:
            args: Configuration arguments
            obs_space: Single agent observation space
            cent_obs_space: Centralized observation space
            act_space: Action space
            n_agents: Total number of agents
            agent_id: ID of this agent (for actor)
            device: torch device
        """
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.n_agents = n_agents
        self.agent_id = agent_id
        
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        # Create GNN Actor and Critic
        self.actor = GNNActor(args, obs_space, act_space, n_agents, device)
        
        # Note: We create one critic for each agent for compatibility with HAPPO,
        # but they share the same architecture and are trained with the same data
        self.critic = GNNCritic(args, cent_obs_space, n_agents, device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
    
    def lr_decay(self, episode, episodes):
        """
        Decay learning rates linearly.
        
        Args:
            episode: Current episode
            episodes: Total episodes
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
    
    def get_actions(self, cent_obs, obs, adj, agent_id, rnn_states_actor, rnn_states_critic,
                   masks, available_actions=None, deterministic=False):
        """
        Compute actions and value predictions.
        
        Args:
            cent_obs: Centralized observations [batch, n_agents, obs_dim]
            obs: All agent observations [batch, n_agents, obs_dim]
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent to compute actions for
            rnn_states_actor: Actor RNN states
            rnn_states_critic: Critic RNN states
            masks: Reset masks
            available_actions: Available actions (for discrete spaces)
            deterministic: Whether to sample or take mode
        
        Returns:
            values: Value predictions
            actions: Actions to take
            action_log_probs: Log probabilities of actions
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
        """
        # Actor forward
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, adj, agent_id, rnn_states_actor, masks,
            available_actions, deterministic
        )
        
        # Critic forward
        values, rnn_states_critic = self.critic(
            cent_obs, adj, rnn_states_critic, masks
        )
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def get_values(self, cent_obs, adj, rnn_states_critic, masks):
        """
        Get value function predictions.
        
        Args:
            cent_obs: Centralized observations [batch, n_agents, obs_dim]
            adj: Adjacency matrix [n_agents, n_agents]
            rnn_states_critic: Critic RNN states
            masks: Reset masks
        
        Returns:
            values: Value predictions
        """
        values, _ = self.critic(cent_obs, adj, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(self, cent_obs, obs, adj, agent_id, rnn_states_actor,
                        rnn_states_critic, action, masks,
                        available_actions=None, active_masks=None):
        """
        Evaluate actions for policy update.
        
        Args:
            cent_obs: Centralized observations [batch, n_agents, obs_dim]
            obs: All agent observations [batch, n_agents, obs_dim]
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent
            rnn_states_actor: Actor RNN states
            rnn_states_critic: Critic RNN states
            action: Actions to evaluate
            masks: Reset masks
            available_actions: Available actions
            active_masks: Active masks
        
        Returns:
            values: Value predictions
            action_log_probs: Log probabilities of actions
            dist_entropy: Action distribution entropy
        """
        # Actor evaluate
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, adj, agent_id, rnn_states_actor, action, masks,
            available_actions, active_masks
        )
        
        # Critic forward
        values, _ = self.critic(cent_obs, adj, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, adj, agent_id, rnn_states_actor, masks,
           available_actions=None, deterministic=False):
        """
        Compute actions only (for evaluation/testing).
        
        Args:
            obs: All agent observations [batch, n_agents, obs_dim]
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent
            rnn_states_actor: Actor RNN states
            masks: Reset masks
            available_actions: Available actions
            deterministic: Whether to sample or take mode
        
        Returns:
            actions: Actions to take
            rnn_states_actor: Updated actor RNN states
        """
        actions, _, rnn_states_actor = self.actor(
            obs, adj, agent_id, rnn_states_actor, masks,
            available_actions, deterministic
        )
        return actions, rnn_states_actor


if __name__ == "__main__":
    """Test GNN-HAPPO Policy."""
    print("Testing GNN-HAPPO Policy...")
    
    import argparse
    from gymnasium import spaces
    from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
    
    # Create mock args
    args = argparse.Namespace(
        hidden_size=64,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=False,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=False,
        recurrent_N=1,
        algorithm_name='happo',
        lr=3e-4,
        critic_lr=3e-4,
        opti_eps=1e-5,
        weight_decay=0,
        # GNN-specific
        gnn_type='GAT',
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        num_attention_heads=4,
        gnn_dropout=0.1,
        use_residual=True,
        critic_pooling='mean',
        single_agent_obs_dim=27
    )
    
    # Setup
    batch_size = 4
    n_agents = 17
    obs_dim = 27
    cent_obs_dim = 162
    action_dim = 3
    
    obs_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=float)
    cent_obs_space = spaces.Box(low=0, high=1, shape=(cent_obs_dim,), dtype=float)
    action_space = spaces.Box(low=0, high=50, shape=(action_dim,), dtype=float)
    
    # Create policy for agent 0
    policy = GNN_HAPPO_Policy(
        args, obs_space, cent_obs_space, action_space,
        n_agents=n_agents, agent_id=0
    )
    
    # Create dummy inputs
    obs = torch.randn(batch_size, n_agents, obs_dim)
    cent_obs = torch.randn(batch_size, n_agents, obs_dim)
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    adj = normalize_adjacency(adj, method='symmetric')
    adj = torch.FloatTensor(adj)
    
    rnn_states_actor = torch.zeros(batch_size, args.recurrent_N, args.hidden_size)
    rnn_states_critic = torch.zeros(batch_size, args.recurrent_N, args.hidden_size)
    masks = torch.ones(batch_size, 1)
    
    # Test get_actions
    print("\nTesting get_actions...")
    values, actions, log_probs, rnn_actor, rnn_critic = policy.get_actions(
        cent_obs, obs, adj, agent_id=0,
        rnn_states_actor=rnn_states_actor,
        rnn_states_critic=rnn_states_critic,
        masks=masks
    )
    print(f"  Values shape: {values.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Log probs shape: {log_probs.shape}")
    
    # Test get_values
    print("\nTesting get_values...")
    values_only = policy.get_values(cent_obs, adj, rnn_states_critic, masks)
    print(f"  Values shape: {values_only.shape}")
    
    # Test evaluate_actions
    print("\nTesting evaluate_actions...")
    test_actions = torch.randn(batch_size, action_dim)
    values_eval, log_probs_eval, entropy = policy.evaluate_actions(
        cent_obs, obs, adj, agent_id=0,
        rnn_states_actor=rnn_states_actor,
        rnn_states_critic=rnn_states_critic,
        action=test_actions,
        masks=masks
    )
    print(f"  Values shape: {values_eval.shape}")
    print(f"  Log probs shape: {log_probs_eval.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    
    # Test act
    print("\nTesting act...")
    actions_only, rnn_actor_new = policy.act(
        obs, adj, agent_id=0,
        rnn_states_actor=rnn_states_actor,
        masks=masks,
        deterministic=True
    )
    print(f"  Actions shape: {actions_only.shape}")
    
    print("\n✓ GNN-HAPPO Policy tests passed!")
