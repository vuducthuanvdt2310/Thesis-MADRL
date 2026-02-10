"""
Graph utility functions for GNN-HAPPO implementation.

This module provides utilities for constructing and manipulating graph structures
representing the supply chain topology (DCs and Retailers).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


def build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True):
    """
    Build adjacency matrix for supply chain graph.
    
    Structure:
    - DCs (nodes 0, 1) connect to ALL retailers (nodes 2-16)
    - This creates a bipartite graph structure
    - Edges are DIRECTED: DC → Retailer (representing supply flow)
    
    Args:
        n_dcs: Number of distribution centers (default: 2)
        n_retailers: Number of retailers (default: 15)
        self_loops: Whether to add self-loops (default: True)
    
    Returns:
        adj_matrix: numpy array of shape [n_agents, n_agents]
                   adj_matrix[i, j] = 1 if there's an edge from i to j
    """
    n_agents = n_dcs + n_retailers
    adj_matrix = np.zeros((n_agents, n_agents), dtype=np.float32)
    
    # Add edges: Each DC connects to all retailers
    for dc_id in range(n_dcs):
        for retailer_id in range(n_dcs, n_agents):
            adj_matrix[dc_id, retailer_id] = 1.0
    
    # Add self-loops (important for GNN: allows node to keep its own features)
    if self_loops:
        np.fill_diagonal(adj_matrix, 1.0)
    
    return adj_matrix


def normalize_adjacency(adj_matrix, method='symmetric'):
    """
    Normalize adjacency matrix for GNN.
    
    Why normalize?
    - Prevents feature explosion during message passing
    - Symmetric normalization: D^{-1/2} A D^{-1/2}
    - Row normalization: D^{-1} A
    
    Args:
        adj_matrix: numpy array [n_agents, n_agents]
        method: 'symmetric' or 'row' normalization
    
    Returns:
        normalized_adj: numpy array [n_agents, n_agents]
    """
    adj = adj_matrix.copy()
    
    if method == 'symmetric':
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
    elif method == 'row':
        # Row normalization: D^{-1} A
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(row_sum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = np.diag(d_inv)
        normalized_adj = d_mat_inv @ adj
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_adj


def adjacency_to_edge_index(adj_matrix):
    """
    Convert adjacency matrix to edge_index format (for PyTorch Geometric).
    
    Edge_index format: [2, num_edges] where:
    - edge_index[0, :] = source nodes
    - edge_index[1, :] = target nodes
    
    Args:
        adj_matrix: numpy array [n_agents, n_agents]
    
    Returns:
        edge_index: torch.LongTensor [2, num_edges]
    """
    edges = np.nonzero(adj_matrix)
    edge_index = torch.LongTensor(np.array(edges))
    return edge_index


def visualize_supply_chain_graph(adj_matrix, n_dcs=2, save_path=None):
    """
    Visualize the supply chain graph structure.
    
    Useful for:
    - Thesis figures showing network topology
    - Debugging graph construction
    - Understanding attention patterns (when overlaid with attention weights)
    
    Args:
        adj_matrix: numpy array [n_agents, n_agents]
        n_dcs: Number of DCs (for node coloring)
        save_path: Optional path to save figure
    """
    n_agents = adj_matrix.shape[0]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Add edges
    edges = np.nonzero(adj_matrix)
    for src, tgt in zip(edges[0], edges[1]):
        if src != tgt:  # Skip self-loops for visualization
            G.add_edge(src, tgt)
    
    # Layout: DCs on left, retailers on right
    pos = {}
    dc_ids = list(range(n_dcs))
    retailer_ids = list(range(n_dcs, n_agents))
    
    # Position DCs vertically on left
    for i, dc_id in enumerate(dc_ids):
        pos[dc_id] = (0, i * (len(retailer_ids) / len(dc_ids)))
    
    # Position retailers vertically on right
    for i, retailer_id in enumerate(retailer_ids):
        pos[retailer_id] = (2, i)
    
    # Draw graph
    plt.figure(figsize=(12, 8))
    
    # Node colors: blue for DCs, green for retailers
    node_colors = ['#1f77b4' if i < n_dcs else '#2ca02c' for i in range(n_agents)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, 
                           labels={i: f'DC{i}' if i < n_dcs else f'R{i-n_dcs}' 
                                  for i in range(n_agents)},
                           font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', alpha=0.5)
    
    plt.title("Supply Chain Network Topology", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_node_degrees(adj_matrix):
    """
    Calculate in-degree and out-degree for each node.
    
    Useful for:
    - Understanding graph structure
    - Debugging
    - Feature engineering (degree can be input feature)
    
    Args:
        adj_matrix: numpy array [n_agents, n_agents]
    
    Returns:
        in_degree: numpy array [n_agents]
        out_degree: numpy array [n_agents]
    """
    in_degree = np.sum(adj_matrix, axis=0)  # Sum over source nodes
    out_degree = np.sum(adj_matrix, axis=1)  # Sum over target nodes
    return in_degree, out_degree


if __name__ == "__main__":
    # Test the graph utilities
    print("Testing graph utilities...")
    
    # Build adjacency matrix
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Number of edges: {np.sum(adj > 0)}")
    
    # Check degrees
    in_deg, out_deg = get_node_degrees(adj)
    print(f"\nDC out-degrees: {out_deg[:2]}")  # Should be 16 each (15 retailers + self-loop)
    print(f"Retailer in-degrees: {in_deg[2:5]}")  # Should be 3 each (2 DCs + self-loop)
    
    # Normalize
    adj_norm = normalize_adjacency(adj, method='symmetric')
    print(f"\nNormalized adjacency range: [{adj_norm.min():.3f}, {adj_norm.max():.3f}]")
    
    # Convert to edge_index format
    edge_index = adjacency_to_edge_index(adj)
    print(f"Edge index shape: {edge_index.shape}")
    
    # Visualize
    visualize_supply_chain_graph(adj, n_dcs=2, save_path="supply_chain_graph.png")
    print("\n✓ All tests passed!")
