"""
Base Graph Neural Network Layers for GNN-HAPPO.

This module implements:
1. GCN (Graph Convolutional Network) - Simple, fast baseline
2. GAT (Graph Attention Network) - More expressive, learns edge importance

WHY USE GNN FOR SUPPLY CHAIN?
-----------------------------
Standard MLP treats each agent's observation independently, missing the fact that:
- Retailers depend on DCs for inventory replenishment
- DCs aggregate demand from multiple retailers
- Actions should be coordinated along the supply chain topology

GNN explicitly models these relationships through graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer.
    
    Mathematical formulation:
        H' = σ(D^{-1/2} A D^{-1/2} H W)
    
    Where:
    - H: Input node features [N_agents, in_features]
    - A: Adjacency matrix [N_agents, N_agents]
    - W: Learnable weight matrix [in_features, out_features]
    - σ: Activation function (ReLU)
    
    Key insight: Each node's new feature is a weighted sum of its neighbors' features.
    """
    
    def __init__(self, in_features, out_features, use_bias=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            use_bias: Whether to use bias term
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node features [batch_size, n_agents, in_features]
            adj: Normalized adjacency matrix [n_agents, n_agents] or [batch_size, n_agents, n_agents]
        
        Returns:
            out: Updated node features [batch_size, n_agents, out_features]
        """
        # Linear transformation: H W
        support = torch.matmul(x, self.weight)  # [batch, n_agents, out_features]
        
        # Graph convolution: A H W
        if adj.dim() == 2:
            # Static adjacency matrix: broadcast across batch
            output = torch.matmul(adj, support)  # [batch, n_agents, out_features]
        else:
            # Batch-specific adjacency matrices
            output = torch.bmm(adj, support)  # [batch, n_agents, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GATLayer(nn.Module):
    """
    Graph Attention Network Layer.
    
    Key innovation over GCN: Learns attention weights for each edge.
    
    For edge (i → j), attention weight α_ij represents:
    "How much should node j attend to node i's information?"
    
    In supply chain context:
    - Retailer j learns to attend more to its primary DC
    - DC i learns to attend to retailers with high/urgent demand
    
    Mathematical formulation:
        α_ij = softmax_j(e_ij)  where e_ij = LeakyReLU(a^T [W h_i || W h_j])
        h'_i = σ(Σ_j α_ij W h_j)
    """
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads (multi-head attention)
            dropout: Dropout rate
            concat: If True, concatenate multi-head outputs; else average
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # Per-head output dimension
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads
        
        # Learnable parameters for each attention head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, self.head_dim))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj):
        """
        Forward pass with multi-head attention.
        
        Args:
            x: Node features [batch_size, n_agents, in_features]
            adj: Adjacency matrix [n_agents, n_agents] (0/1 values, indicates valid edges)
        
        Returns:
            out: Updated node features [batch_size, n_agents, out_features]
            attention_weights: Attention weights [batch_size, num_heads, n_agents, n_agents] (optional, for visualization)
        """
        batch_size, n_agents, _ = x.shape
        
        # Linear transformation for each head: [batch, num_heads, n_agents, head_dim]
        h = torch.einsum('bni,hij->bhnj', x, self.W)
        
        # Compute attention scores for all edges
        # For each head, compute: a^T [W h_i || W h_j] for all pairs (i, j)
        
        # Expand for broadcasting: [batch, num_heads, n_agents, 1, head_dim]
        h_i = h.unsqueeze(3)
        # Expand for broadcasting: [batch, num_heads, 1, n_agents, head_dim]
        h_j = h.unsqueeze(2)
        
        # Concatenate: [batch, num_heads, n_agents, n_agents, 2*head_dim]
        h_cat = torch.cat([h_i.expand(-1, -1, -1, n_agents, -1),
                          h_j.expand(-1, -1, n_agents, -1, -1)], dim=-1)
        
        # Attention mechanism: [batch, num_heads, n_agents, n_agents]
        e = torch.einsum('bhnmd,hdk->bhnm', h_cat, self.a).squeeze(-1)
        e = self.leaky_relu(e)
        
        # Mask attention for non-existent edges
        # adj: [n_agents, n_agents], expand to match e's shape
        mask = (adj == 0)
        e = e.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax to get attention weights: [batch, num_heads, n_agents, n_agents]
        attention = F.softmax(e, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features using attention weights
        # [batch, num_heads, n_agents, n_agents] @ [batch, num_heads, n_agents, head_dim]
        # → [batch, num_heads, n_agents, head_dim]
        h_prime = torch.einsum('bhnm,bhmj->bhnj', attention, h)
        
        # Combine multi-head outputs
        if self.concat:
            # Concatenate: [batch, n_agents, num_heads * head_dim]
            output = h_prime.transpose(1, 2).contiguous().view(batch_size, n_agents, -1)
        else:
            # Average: [batch, n_agents, head_dim]
            output = h_prime.mean(dim=1)
        
        return output, attention


class GNNBase(nn.Module):
    """
    Base GNN encoder: Stacks multiple GNN layers.
    
    Architecture:
        Input → GNN Layer 1 → ReLU → GNN Layer 2 → ReLU → ... → Output
    
    Used as feature extractor in both GNNActor and GNNCritic.
    """
    
    def __init__(self, in_features, hidden_dim, num_layers=2, gnn_type='GAT', 
                 num_heads=4, dropout=0.1, use_residual=True):
        """
        Args:
            in_features: Input feature dimension (observation size)
            hidden_dim: Hidden dimension for each layer
            num_layers: Number of GNN layers to stack
            gnn_type: 'GAT' or 'GCN'
            num_heads: Number of attention heads (for GAT only)
            dropout: Dropout rate
            use_residual: Whether to use residual connections (helps training)
        """
        super(GNNBase, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.use_residual = use_residual
        
        # Build layer list
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_dim
            
            if gnn_type == 'GAT':
                layer = GATLayer(in_dim, hidden_dim, num_heads=num_heads, 
                               dropout=dropout, concat=True)
            elif gnn_type == 'GCN':
                layer = GCNLayer(in_dim, hidden_dim, use_bias=True)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.layers.append(layer)
        
        # Layer normalization for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, return_attention=False):
        """
        Forward pass through all GNN layers.
        
        Args:
            x: Node features [batch_size, n_agents, in_features]
            adj: Adjacency matrix [n_agents, n_agents]
            return_attention: If True, return attention weights (for GAT only)
        
        Returns:
            h: Node embeddings [batch_size, n_agents, hidden_dim]
            attention_weights: List of attention weights (if return_attention=True and GAT)
        """
        h = x
        attention_weights = []
        
        for i, layer in enumerate(self.layers):
            h_old = h
            
            # GNN layer
            if self.gnn_type == 'GAT':
                h, attn = layer(h, adj)
                if return_attention:
                    attention_weights.append(attn)
            else:
                h = layer(h, adj)
            
            # Layer normalization
            h = self.layer_norms[i](h)
            
            # Activation
            if i < self.num_layers - 1:  # No activation on last layer
                h = F.relu(h)
                h = self.dropout(h)
            
            # Residual connection (skip connection)
            if self.use_residual and h_old.shape[-1] == h.shape[-1]:
                h = h + h_old
        
        if return_attention and self.gnn_type == 'GAT':
            return h, attention_weights
        else:
            return h


if __name__ == "__main__":
    """Test the GNN layers."""
    print("Testing GNN layers...")
    
    # Setup
    batch_size = 4
    n_agents = 17  # 2 DCs + 15 retailers
    in_features = 27  # Example: DC observation size
    hidden_dim = 128
    
    # Create dummy data
    x = torch.randn(batch_size, n_agents, in_features)
    
    # Create adjacency matrix (supply chain topology)
    from utils.graph_utils import build_supply_chain_adjacency, normalize_adjacency
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    adj = normalize_adjacency(adj, method='symmetric')
    adj = torch.FloatTensor(adj)
    
    print(f"Input shape: {x.shape}")
    print(f"Adjacency shape: {adj.shape}")
    
    # Test GCN Layer
    print("\n--- Testing GCN Layer ---")
    gcn = GCNLayer(in_features, hidden_dim)
    out_gcn = gcn(x, adj)
    print(f"GCN output shape: {out_gcn.shape}")
    assert out_gcn.shape == (batch_size, n_agents, hidden_dim), "GCN shape mismatch!"
    
    # Test GAT Layer
    print("\n--- Testing GAT Layer ---")
    gat = GATLayer(in_features, hidden_dim, num_heads=4)
    out_gat, attn_weights = gat(x, adj)
    print(f"GAT output shape: {out_gat.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    assert out_gat.shape == (batch_size, n_agents, hidden_dim), "GAT shape mismatch!"
    
    # Test GNNBase with GCN
    print("\n--- Testing GNNBase (GCN) ---")
    gnn_base_gcn = GNNBase(in_features, hidden_dim, num_layers=2, gnn_type='GCN')
    out_base_gcn = gnn_base_gcn(x, adj)
    print(f"GNNBase (GCN) output shape: {out_base_gcn.shape}")
    
    # Test GNNBase with GAT
    print("\n--- Testing GNNBase (GAT) ---")
    gnn_base_gat = GNNBase(in_features, hidden_dim, num_layers=2, gnn_type='GAT', num_heads=4)
    out_base_gat, attn_list = gnn_base_gat(x, adj, return_attention=True)
    print(f"GNNBase (GAT) output shape: {out_base_gat.shape}")
    print(f"Number of attention weight tensors: {len(attn_list)}")
    
    print("\n✓ All GNN layer tests passed!")
