import torch
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

def compute_apsp_heat_kernel(
    num_nodes: int,
    edge_index: torch.Tensor,
    sigma: float = 0.35,
    power: float = 2.0,
    disconnected_value: float = 0.0,
    diagonal_value: float = 1.0
) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0, 0))
    if edge_index.size(1) == 0:
        W = torch.full((num_nodes, num_nodes), disconnected_value)
        W.fill_diagonal_(diagonal_value)
        return W
        
    # Build adjacency
    row, col = edge_index.numpy()
    data = np.ones_like(row, dtype=float)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Compute shortest paths
    D = shortest_path(adj, directed=False, unweighted=True)
    
    # Handle infinite distances (disconnected components)
    finite_mask = np.isfinite(D)
    
    max_D = D[finite_mask].max() if np.any(finite_mask) else 1.0
    if max_D == 0:
        max_D = 1.0
        
    D_norm = D / max_D
    
    # Compute heat kernel
    W = np.exp(- (D_norm / sigma)**power)
    W[~finite_mask] = disconnected_value
    np.fill_diagonal(W, diagonal_value)
    
    return torch.from_numpy(W).float()
