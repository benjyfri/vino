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
    powers: list | tuple = (0, 1, 2),
    weights: list | tuple = (1.0, 0.5, 0.25),
    robust_quantile: float = 0.95,
    clip: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    if len(powers) != len(weights) or not powers:
        raise ValueError("powers and weights must be non-empty and have equal length")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, M]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr rows must match edge_index columns")
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
        # Aggregate both incoming and outgoing incident edges. For the common
        # bidirectional PyG representation, divide duplicate directed entries
        # naturally through the mean reduction.
        row, col = edge_index
        incident_index = torch.cat([row, col])
        incident_values = torch.cat([Z_edge, Z_edge], dim=0)
        R_edge_node = scatter(incident_values, incident_index, dim=0, dim_size=num_nodes, reduce="mean")
        
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
