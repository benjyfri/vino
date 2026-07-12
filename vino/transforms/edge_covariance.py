import torch
import math
from torch_geometric.utils import scatter

def compute_edge_covariance(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    A_prop: torch.Tensor,
    h_edge: int = 64,
    seed: int = 456,
    powers: list = [0, 1, 2],
    weights: list = [1.0, 0.5, 0.25],
    robust_quantile: float = 0.95,
    clip: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0,0))
        
    m, d_edge = edge_attr.size()
    
    gen = torch.Generator().manual_seed(seed + d_edge)
    C_edge = torch.randn((d_edge, h_edge), generator=gen, dtype=torch.float32) / math.sqrt(h_edge)
    
    if m > 0:
        Z_edge = edge_attr @ C_edge
    else:
        Z_edge = torch.zeros((0, h_edge))
        
    # Aggregate to nodes
    R_edge_node = torch.zeros((num_nodes, h_edge), dtype=torch.float32)
    if m > 0:
        # Sum incident edges
        row, col = edge_index
        # Both directions if graph is undirected, or just incident. Assuming PyG standard (both directions in edge_index)
        # We scatter over row
        R_edge_node = scatter(Z_edge, row, dim=0, dim_size=num_nodes, reduce="mean")
        
    K_edge = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    for p, beta in zip(powers, weights):
        if p == 0:
            R_curr = R_edge_node
        else:
            R_curr = torch.matrix_power(A_prop, p) @ R_edge_node
            
        R_c = R_curr - R_curr.mean(dim=0, keepdim=True)
        K_p = R_c @ R_c.T
        K_edge += beta * K_p
        
    # Robust normalization
    q = torch.quantile(torch.abs(K_edge), robust_quantile) + eps
    K_edge = K_edge / q
    
    K_edge = torch.clamp(K_edge, -clip, clip)
    K_edge = (K_edge + clip) / (2 * clip)
    return K_edge
