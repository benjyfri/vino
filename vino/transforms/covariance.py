import torch
import math

def compute_node_covariance(
    X: torch.Tensor,
    A_prop: torch.Tensor,
    h_node: int = 64,
    seed: int = 123,
    powers: list | tuple = (0, 1, 2),
    weights: list | tuple = (1.0, 0.5, 0.25),
    robust_quantile: float = 0.95,
    clip: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"X must have shape [N, D], got {tuple(X.shape)}")
    if len(powers) != len(weights) or not powers:
        raise ValueError("powers and weights must be non-empty and have equal length")
    if any(int(p) < 0 for p in powers):
        raise ValueError("propagation powers must be non-negative")
    n, d_node = X.size()
    if n == 0:
        return torch.zeros((0,0))
        
    gen = torch.Generator().manual_seed(seed + d_node)
    
    # Projection C_node ~ N(0, 1/h_node)
    C_node = torch.randn((d_node, h_node), generator=gen, dtype=torch.float32) / math.sqrt(h_node)
    
    R0 = X @ C_node
    
    K_node = torch.zeros((n, n), dtype=torch.float32)
    for p, alpha in zip(powers, weights):
        if p == 0:
            R_curr = R0
        else:
            R_curr = torch.matrix_power(A_prop, p) @ R0
            
        R_c = R_curr - R_curr.mean(dim=0, keepdim=True)
        K_p = R_c @ R_c.T
        K_node += alpha * K_p
        
    # Robust normalization
    q = torch.quantile(torch.abs(K_node), robust_quantile) + eps
    K_node = K_node / q
    
    K_node = torch.clamp(K_node, -clip, clip)
    
    # Map to [0,1]
    K_node = (K_node + clip) / (2 * clip)
    return K_node
