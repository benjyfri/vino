import torch

def patch_tokens_to_node_pair_grid(
    patch_tokens: torch.Tensor,
    token_grid_size: int,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """
    Reshapes the sequence of patch tokens into a 2D grid of token_grid_size x token_grid_size.
    
    Args:
        patch_tokens: [B, T, D] where T = 1 + token_grid_size**2 if has_cls_token else token_grid_size**2
                     or [B, Ht, Wt, D] if already reshaped.
        token_grid_size: Expected token grid dimension.
        has_cls_token: Whether the first token in sequence is a CLS token.
        
    Returns:
        [B, token_grid_size, token_grid_size, D] tensor of patch tokens.
    """
    if patch_tokens.dim() == 4:
        # Already a grid
        b, h, w, d = patch_tokens.shape
        if h != token_grid_size or w != token_grid_size:
            raise ValueError(f"Expected grid of size {token_grid_size}x{token_grid_size}, got {h}x{w}")
        return patch_tokens
        
    b, t, d = patch_tokens.shape
    expected_t = token_grid_size * token_grid_size
    if has_cls_token:
        expected_t += 1
        
    if t != expected_t:
        raise ValueError(f"Expected {expected_t} tokens, got {t}")
        
    if has_cls_token:
        # discard CLS token
        grid_tokens = patch_tokens[:, 1:, :]
    else:
        grid_tokens = patch_tokens
        
    return grid_tokens.view(b, token_grid_size, token_grid_size, d)

def extract_edge_tokens(
    token_grid: torch.Tensor,
    edge_index: torch.Tensor,
    n_valid: int,
) -> torch.Tensor:
    """
    Extracts patch tokens corresponding to specified canonical edge pairs.
    
    Args:
        token_grid: [B, Ntok, Ntok, D] grid of tokens from patch_tokens_to_node_pair_grid.
        edge_index: [2, M] canonical edge indices for a single graph.
                   If batched, this needs adaptation. For now assumes unbatched [2, M].
        n_valid: Number of valid nodes for this graph.
        
    Returns:
        [M, D] extracted tokens.
    """
    # Assuming token_grid is [B, Ntok, Ntok, D], we need to extract for B=0 or similar.
    # Usually edge extraction would be per-graph in a batch loop.
    # Let's assume unbatched or B=1 for this utility for now, or edge_index belongs to the first graph.
    
    if token_grid.dim() == 4:
        # Just take the first element if B=1
        if token_grid.size(0) != 1:
            raise NotImplementedError("Batch size > 1 edge extraction requires batch indices in edge_index")
        grid = token_grid[0]
    else:
        grid = token_grid
        
    src, dst = edge_index
    
    # Optional check to ensure we don't index beyond n_valid or the grid
    valid_mask = (src < n_valid) & (dst < n_valid)
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    edge_tokens = grid[src, dst]
    return edge_tokens
