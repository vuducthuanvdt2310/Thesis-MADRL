"""
GNN-based Critic Network for GNN-HAPPO.

Key difference from standard critic:
- Standard Critic: centralized_obs → MLP → value
- GNN Critic: all_obs + graph → GNN → global pooling → MLP → value

WHY CENTRALIZED GNN CRITIC:
- HAPPO uses centralized critic (sees all agent observations)
- GNN makes critic aware of supply chain structure
- Critic can better estimate global value by understanding agent relationships
- Example: High DC inventory + Low retailer inventory = lower value than balanced inventory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.gnn.gnn_base import GNNBase
from algorithms.utils.util import init, check
from algorithms.utils.rnn import RNNLayer
from utils.util import get_shape_from_obs_space


class GNNCritic(nn.Module):
    """
    GNN-based Critic Network for Value Estimation.
    
    Architecture:
        All Agent Observations → GNN Layers (with graph structure)
                                      ↓
                              Node Embeddings [n_agents, gnn_hidden_dim]
                                      ↓
                          Global Pooling (mean/max/attention)
                                      ↓
                              MLP Head → Value (scalar)
    
    The centralized critic benefits from graph structure to estimate global state value.
    """
    
    def __init__(self, args, cent_obs_space, n_agents, device=torch.device("cpu")):
        """
        Args:
            args: Configuration arguments
            cent_obs_space: Centralized observation space (typically concatenated obs)
            n_agents: Number of agents
            device: torch device
        """
        super(GNNCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self.n_agents = n_agents
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # For GNN critic, we need to know individual agent obs dim
        # cent_obs_space is total obs (e.g., 162D for 2 DCs + 15 retailers)
        # We'll infer single agent obs dim (this is a simplification; 
        # in practice you might pass this explicitly)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        cent_obs_dim = cent_obs_shape[0] if isinstance(cent_obs_shape, tuple) else cent_obs_shape
        
        # GNN-specific parameters
        self.gnn_type = getattr(args, 'gnn_type', 'GAT')
        self.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 128)
        self.gnn_num_layers = getattr(args, 'gnn_num_layers', 2)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)
        self.gnn_dropout = getattr(args, 'gnn_dropout', 0.1)
        self.use_residual = getattr(args, 'use_residual', True)
        self.pooling_method = getattr(args, 'critic_pooling', 'mean')  # 'mean', 'max', or 'concat'
        
        # CRITICAL: Use single_agent_obs_dim (max obs dim across all agents) 
        # Observations are padded to max_obs_dim in the runner before being passed to GNN
        # DCs have 27D, retailers have 36D → all are padded to 36D for GNN processing
        # Default to 36 (retailer obs dim) instead of 27 (DC obs dim)
        self.obs_dim = getattr(args, 'single_agent_obs_dim', 36)
        
        # GNN feature extractor
        self.gnn_base = GNNBase(
            in_features=self.obs_dim,  # Use padded dimension (36)
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
            gnn_type=self.gnn_type,
            num_heads=self.num_attention_heads,
            dropout=self.gnn_dropout,
            use_residual=self.use_residual
        )
        
        # Determine input size for MLP head based on pooling
        if self.pooling_method == 'concat':
            mlp_input_dim = self.gnn_hidden_dim * n_agents
        else:  # mean or max
            mlp_input_dim = self.gnn_hidden_dim
        
        # MLP head: global state → value
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # RNN layer (optional)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                               self._recurrent_N, self._use_orthogonal)
        
        # Value output layer
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        
        self.to(device)
    
    def forward(self, cent_obs, adj, rnn_states, masks):
        """
        Forward pass to compute value.
        
        Args:
            cent_obs: Centralized observations [batch_size, n_agents, obs_dim]
                     Note: This should be all agent observations, not concatenated
            adj: Adjacency matrix [n_agents, n_agents]
            rnn_states: RNN hidden states [batch_size, recurrent_N, hidden_size]
            masks: Reset masks [batch_size, 1]
        
        Returns:
            values: State value estimates [batch_size, 1]
            rnn_states: Updated RNN states
        """
        # Convert to tensors
        cent_obs = check(cent_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv) if not isinstance(adj, torch.Tensor) else adj.to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # GNN forward: get node embeddings
        # [batch, n_agents, obs_dim] → [batch, n_agents, gnn_hidden_dim]
        node_embeddings = self.gnn_base(cent_obs, adj)
        
        # Global pooling: aggregate all node embeddings into global state
        if self.pooling_method == 'mean':
            global_state = node_embeddings.mean(dim=1)  # [batch, gnn_hidden_dim]
        elif self.pooling_method == 'max':
            global_state = node_embeddings.max(dim=1)[0]  # [batch, gnn_hidden_dim]
        elif self.pooling_method == 'concat':
            global_state = node_embeddings.view(node_embeddings.size(0), -1)  # [batch, n_agents * gnn_hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # MLP head
        critic_features = self.mlp_head(global_state)
        
        # RNN (if used)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        # Value output
        values = self.v_out(critic_features)
        
        return values, rnn_states


if __name__ == "__main__":
    """Test GNN Critic."""
    print("Testing GNN Critic...")
    
    import argparse
    from gymnasium import spaces
    from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
    
    # Create mock args
    args = argparse.Namespace(
        hidden_size=64,
        use_orthogonal=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=False,
        recurrent_N=1,
        # GNN-specific
        gnn_type='GAT',
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        num_attention_heads=4,
        gnn_dropout=0.1,
        use_residual=True,
        critic_pooling='mean',  # or 'max' or 'concat'
        single_agent_obs_dim=27
    )
    
    # Setup
    batch_size = 4
    n_agents = 17
    obs_dim = 27
    cent_obs_dim = 162  # 2*27 + 15*36 (heterogeneous agents)
    
    cent_obs_space = spaces.Box(low=0, high=1, shape=(cent_obs_dim,), dtype=float)
    
    # Create critic
    critic = GNNCritic(args, cent_obs_space, n_agents=n_agents)
    
    # Create dummy inputs
    cent_obs = torch.randn(batch_size, n_agents, obs_dim)  # All agent observations
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    adj = normalize_adjacency(adj, method='symmetric')
    adj = torch.FloatTensor(adj)
    
    rnn_states = torch.zeros(batch_size, args.recurrent_N, args.hidden_size)
    masks = torch.ones(batch_size, 1)
    
    # Forward pass
    print(f"\nTesting forward pass...")
    values, rnn_states_new = critic(cent_obs, adj, rnn_states, masks)
    print(f"Values shape: {values.shape}")
    assert values.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {values.shape}"
    
    print(f"Value estimates: {values.squeeze().detach().numpy()}")
    
    # Test different pooling methods
    for pooling in ['mean', 'max', 'concat']:
        print(f"\nTesting with {pooling} pooling...")
        args.critic_pooling = pooling
        critic_test = GNNCritic(args, cent_obs_space, n_agents=n_agents)
        values_test, _ = critic_test(cent_obs, adj, rnn_states, masks)
        print(f"  Values shape: {values_test.shape}")
        assert values_test.shape == (batch_size, 1)
    
    print("\n✓ GNN Critic tests passed!")
