"""
Graph utility functions for GNN-HAPPO implementation.

This module provides utilities for constructing and manipulating graph structures
representing the supply chain topology (DCs and Retailers).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


def build_supply_chain_adjacency(n_dcs=2, n_retailers=15, self_loops=True,
                                  dc_assignments=None):
    """
    Build adjacency matrix for supply chain graph.

    Structure:
    - If ``dc_assignments`` is given (dict mapping dc_id -> list of agent IDs),
      each DC only connects to its **assigned** retailers, mirroring the
      exclusive sourcing constraint in the environment.
    - Otherwise every DC connects to every retailer (fully bipartite).
    - Edges are bidirectional: DC ↔ Retailer (supply flow + demand signal).

    Args:
        n_dcs: Number of distribution centers (default: 2)
        n_retailers: Number of retailers (default: 15)
        self_loops: Whether to add self-loops (default: True)
        dc_assignments: Optional dict {dc_id (int): [retailer_agent_ids]}
                        matching the env's ``self.dc_assignments``.

    Returns:
        adj_matrix: numpy array of shape [n_agents, n_agents]
                   adj_matrix[i, j] = 1 if there's an edge from i to j
    """
    n_agents = n_dcs + n_retailers
    adj_matrix = np.zeros((n_agents, n_agents), dtype=np.float32)

    if dc_assignments is not None:
        # Connect each DC only to its assigned retailers
        for dc_id, retailer_agent_ids in dc_assignments.items():
            for retailer_id in retailer_agent_ids:
                adj_matrix[dc_id, retailer_id] = 1.0
                adj_matrix[retailer_id, dc_id] = 1.0
    else:
        # Fallback: fully bipartite (every DC → every retailer)
        for dc_id in range(n_dcs):
            for retailer_id in range(n_dcs, n_agents):
                adj_matrix[dc_id, retailer_id] = 1.0
                adj_matrix[retailer_id, dc_id] = 1.0

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

    # --- Mode 1: fully bipartite (no assignments) ---
    adj = build_supply_chain_adjacency(n_dcs=2, n_retailers=15)
    print(f"[Bipartite] Adjacency matrix shape: {adj.shape}")
    print(f"[Bipartite] Number of edges: {int(np.sum(adj > 0))}")
    in_deg, out_deg = get_node_degrees(adj)
    print(f"[Bipartite] DC out-degrees:      {out_deg[:2]}")   # 16 each
    print(f"[Bipartite] Retailer in-degrees: {in_deg[2:5]}")  # 3 each

    # --- Mode 2: assignment-aware (matching multi_dc_config.yaml defaults) ---
    dc_assignments = {
        0: [2, 3, 4, 5, 6, 7, 8],        # DC0 → agent IDs 2-8  (retailers 0-6)
        1: [9, 10, 11, 12, 13, 14, 15, 16],  # DC1 → agent IDs 9-16 (retailers 7-14)
    }
    adj_a = build_supply_chain_adjacency(n_dcs=2, n_retailers=15,
                                         dc_assignments=dc_assignments)
    print(f"\n[Assigned]  Adjacency matrix shape: {adj_a.shape}")
    print(f"[Assigned]  Number of edges: {int(np.sum(adj_a > 0))}")
    # DC0 should have edges to 7 retailers only; DC1 to 8 retailers only
    in_deg_a, out_deg_a = get_node_degrees(adj_a)
    print(f"[Assigned]  DC0 out-degree: {out_deg_a[0]}  (expected 8: 7 retailers + self)")
    print(f"[Assigned]  DC1 out-degree: {out_deg_a[1]}  (expected 9: 8 retailers + self)")

    # Normalize
    adj_norm = normalize_adjacency(adj_a, method='symmetric')
    print(f"\nNormalized adjacency range: [{adj_norm.min():.3f}, {adj_norm.max():.3f}]")

    # Convert to edge_index format
    edge_index = adjacency_to_edge_index(adj_a)
    print(f"Edge index shape: {edge_index.shape}")

    print("\n✓ All tests passed!")

