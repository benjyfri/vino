import torch
import math

def compute_node_covariance(
    X: torch.Tensor,
    A_prop: torch.Tensor,
    h_node: int = 64,
    seed: int = 123,
    powers: list = [0, 1, 2],
    weights: list = [1.0, 0.5, 0.25],
    robust_quantile: float = 0.95,
    clip: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    n, d_node = X.size()
    if n == 0:
        return torch.zeros((0,0))
        
    gen = torch.Generator().manual_seed(seed + d_node)
    
    # Projection C_node ~ N(0, 1/h_node)
    C_node = torch.randn((d_node, h_node), generator=gen, dtype=torch.float32) / math.sqrt(h_node)
    
    R0 = X @ C_node
    
    K_node = torch.zeros((n, n), dtype=torch.float32)
    R_p = R0
    
    for p, alpha in zip(powers, weights):
        if p == 0:
            R_curr = R0
        else:
            R_curr = R_p
            for _ in range(p - (powers[0] if powers[0] == 0 else 0)):
                # Simple loop to reach power p
                pass 
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
