"""
GNN-based Actor Network for GNN-HAPPO.

The key difference from standard MLP-based actor:
- Standard Actor: obs → MLP → action
- GNN Actor: all_obs + graph → GNN → node_embeddings → extract agent's embedding → MLP → action

WHY THIS MATTERS FOR SUPPLY CHAIN:
- Retailer can "see" DC inventory levels through graph aggregation
- DC can "see" aggregate retailer demand
- Actions are implicitly coordinated through shared graph representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.gnn.gnn_base import GNNBase
from algorithms.utils.util import init, check
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space


class GNNActor(nn.Module):
    """
    GNN-based Actor Network for Policy.
    
    Architecture:
        All Agent Observations → GNN Layers (with graph structure)
                                      ↓
                              Node Embeddings [n_agents, hidden_dim]
                                      ↓
                       Extract specific agent's embedding
                                      ↓
                              MLP Head → Action Distribution
    
    The GNN allows each agent to implicitly "communicate" through the graph structure,
    improving coordination without explicit communication channels.
    """
    
    def __init__(self, args, obs_space, action_space, n_agents, device=torch.device("cpu")):
        """
        Args:
            args: Configuration arguments
            obs_space: Single agent observation space
            action_space: Single agent action space
            n_agents: Total number of agents (for graph processing)
            device: torch device
        """
        super(GNNActor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self.n_agents = n_agents
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Get observation dimension
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        
        # GNN-specific parameters (with defaults if not in args)
        self.gnn_type = getattr(args, 'gnn_type', 'GAT')
        self.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 128)
        self.gnn_num_layers = getattr(args, 'gnn_num_layers', 2)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)
        self.gnn_dropout = getattr(args, 'gnn_dropout', 0.1)
        self.use_residual = getattr(args, 'use_residual', True)
        
        # CRITICAL: Use single_agent_obs_dim (max obs dim across all agents) instead of obs_dim
        # This is because observations are padded to max_obs_dim in the runner before being passed to GNN
        # DCs have 27D, retailers have 36D → all are padded to 36D for GNN processing
        self.gnn_input_dim = getattr(args, 'single_agent_obs_dim', self.obs_dim)
        
        # GNN feature extractor
        # Takes all agent observations and produces node embeddings
        self.gnn_base = GNNBase(
            in_features=self.gnn_input_dim,  # Use padded dimension (36), not agent-specific obs_dim (27)
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
            gnn_type=self.gnn_type,
            num_heads=self.num_attention_heads,
            dropout=self.gnn_dropout,
            use_residual=self.use_residual
        )
        
        # MLP head: converts node embedding to policy input
        # This allows agent-specific processing after graph aggregation
        self.mlp_head = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # RNN layer (optional, for temporal dependencies)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, 
                               self._recurrent_N, self._use_orthogonal)
        
        # Action head (same as standard HAPPO)
        self.act = ACTLayer(action_space, self.hidden_size, 
                           self._use_orthogonal, self._gain, args)
        
        self.to(device)
    
    def forward(self, obs, adj, agent_id, rnn_states, masks, 
                available_actions=None, deterministic=False):
        """
        Forward pass to compute actions.
        
        Args:
            obs: ALL agent observations [batch_size, n_agents, obs_dim]
                 (Note: this is different from standard actor which only takes single agent obs)
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent's action to compute (int)
            rnn_states: RNN hidden states [batch_size, recurrent_N, hidden_size]
            masks: Reset masks [batch_size, 1]
            available_actions: Available actions (for discrete action spaces)
            deterministic: Whether to sample or take mode of distribution
        
        Returns:
            actions: [batch_size, action_dim]
            action_log_probs: [batch_size, action_dim]
            rnn_states: Updated RNN states
        """
        # Convert to tensors
        obs = check(obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv) if not isinstance(adj, torch.Tensor) else adj.to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        # GNN forward pass: aggregate graph information
        # Input: [batch, n_agents, obs_dim] → Output: [batch, n_agents, gnn_hidden_dim]
        node_embeddings = self.gnn_base(obs, adj)
        
        # Extract this agent's embedding
        # [batch, gnn_hidden_dim]
        agent_embedding = node_embeddings[:, agent_id, :]
        
        # MLP head: process agent-specific information
        actor_features = self.mlp_head(agent_embedding)
        
        # RNN (if used)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        # Action distribution
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, adj, agent_id, rnn_states, action, masks,
                        available_actions=None, active_masks=None):
        """
        Evaluate actions (for policy update).
        
        Args:
            obs: ALL agent observations [batch_size, n_agents, obs_dim]
            adj: Adjacency matrix [n_agents, n_agents]
            agent_id: Which agent (int)
            rnn_states: RNN states
            action: Actions to evaluate [batch_size, action_dim]
            masks: Reset masks
            available_actions: Available actions
            active_masks: Active masks
        
        Returns:
            action_log_probs: Log probabilities of actions
            dist_entropy: Entropy of action distribution
        """
        # Convert to tensors
        obs = check(obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv) if not isinstance(adj, torch.Tensor) else adj.to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        # GNN forward
        node_embeddings = self.gnn_base(obs, adj)
        agent_embedding = node_embeddings[:, agent_id, :]
        actor_features = self.mlp_head(agent_embedding)
        
        # RNN
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        # Evaluate actions
        if self.args.algorithm_name == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = \
                self.act.evaluate_actions_trpo(
                    actor_features, action, available_actions,
                    active_masks=active_masks if self._use_policy_active_masks else None
                )
            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None
            )
            return action_log_probs, dist_entropy


if __name__ == "__main__":
    """Test GNN Actor."""
    print("Testing GNN Actor...")
    
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
        # GNN-specific
        gnn_type='GAT',
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        num_attention_heads=4,
        gnn_dropout=0.1,
        use_residual=True
    )
    
    # Setup
    batch_size = 4
    n_agents = 17
    obs_dim = 27
    action_dim = 3
    
    obs_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=float)
    action_space = spaces.Box(low=0, high=50, shape=(action_dim,), dtype=float)
    
    # Create actor
    actor = GNNActor(args, obs_space, action_space, n_agents=n_agents)
    
    # Create dummy inputs
    obs = torch.randn(batch_size, n_agents, obs_dim)  # All agent observations
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    adj = normalize_adjacency(adj, method='symmetric')
    adj = torch.FloatTensor(adj)
    
    rnn_states = torch.zeros(batch_size, args.recurrent_N, args.hidden_size)
    masks = torch.ones(batch_size, 1)
    
    # Forward pass for agent 0 (a DC)
    print(f"\nTesting forward pass for agent 0 (DC)...")
    actions, log_probs, rnn_states_new = actor(obs, adj, agent_id=0, 
                                                rnn_states=rnn_states, masks=masks)
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    assert actions.shape == (batch_size, action_dim)
    
    # Forward pass for agent 5 (a retailer)
    print(f"\nTesting forward pass for agent 5 (Retailer)...")
    actions, log_probs, rnn_states_new = actor(obs, adj, agent_id=5,
                                                rnn_states=rnn_states, masks=masks)
    print(f"Actions shape: {actions.shape}")
    assert actions.shape == (batch_size, action_dim)
    
    # Test evaluate actions
    print(f"\nTesting evaluate_actions...")
    test_actions = torch.randn(batch_size, action_dim)
    log_probs_eval, entropy = actor.evaluate_actions(
        obs, adj, agent_id=0, rnn_states=rnn_states, 
        action=test_actions, masks=masks
    )
    print(f"Evaluated log probs shape: {log_probs_eval.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    print("\n✓ GNN Actor tests passed!")
