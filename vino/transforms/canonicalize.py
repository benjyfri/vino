import torch
from typing import Tuple, Dict

def canonicalize_topology(
    W_top: torch.Tensor, 
    eigengap_tol: float = 1e-6,
    tie_tol: float = 1e-8,
    lex_round_decimals: int = 8,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict]:
    n = W_top.size(0)
    meta = {
        "canonicalization_unstable": False,
        "fiedler_eigengap": 0.0,
        "num_fiedler_ties": 0
    }
    
    if n <= 2:
        return torch.arange(n), meta
        
    # Symmetric normalized Laplacian
    deg = W_top.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=eps), -0.5)
    
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L_sym = torch.eye(n) - D_inv_sqrt @ W_top @ D_inv_sqrt
    
    # Eigendecomposition
    # torch.linalg.eigh expects symmetric matrix
    L_sym = (L_sym + L_sym.T) / 2
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
    except Exception:
        meta["canonicalization_unstable"] = True
        return torch.arange(n), meta
        
    v2 = eigenvectors[:, 1]
    meta["fiedler_eigengap"] = float((eigenvalues[2] - eigenvalues[1]).item()) if n > 2 else 0.0
    
    if meta["fiedler_eigengap"] < eigengap_tol:
        meta["canonicalization_unstable"] = True
        
    # Check for near-ties in Fiedler vector
    v2_diff = torch.abs(v2.unsqueeze(0) - v2.unsqueeze(1))
    v2_diff.fill_diagonal_(float('inf'))
    if (v2_diff < tie_tol).any():
        meta["num_fiedler_ties"] = int((v2_diff < tie_tol).sum().item() // 2)
        meta["canonicalization_unstable"] = True
        
    # Candidate permutations
    order_plus = torch.argsort(v2)
    order_minus = torch.argsort(-v2)
    
    W_plus = W_top[order_plus][:, order_plus]
    W_minus = W_top[order_minus][:, order_minus]
    
    # Rounding for robust lexicographic comparison
    W_plus_flat = torch.round(W_plus.flatten() * (10**lex_round_decimals))
    W_minus_flat = torch.round(W_minus.flatten() * (10**lex_round_decimals))
    
    # Python comparison of lists
    if W_plus_flat.tolist() < W_minus_flat.tolist():
        return order_plus, meta
    else:
        return order_minus, meta
