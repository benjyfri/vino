import torch
import torch.nn.functional as F

def transform_graph_image_for_model(
    image: torch.Tensor,
    valid_node_mask: torch.Tensor,
    input_mode: str,
    input_size: int = 224,
    patch_size: int = 16,
    token_grid_size: int | None = None,
    resize_mode: str = "bilinear",
) -> dict:
    """
    Transforms a cached graph image [3, Nmax, Nmax] into a model-ready image.
    
    Args:
        image: [3, Nmax, Nmax] graph image tensor
        valid_node_mask: [Nmax] boolean tensor
        input_mode: "resize_bilinear" or "patch_aligned_repeat"
        input_size: target size for resize_bilinear (H=W=input_size)
        patch_size: patch size for patch_aligned_repeat
        token_grid_size: target grid size for patch_aligned_repeat
        resize_mode: "bilinear", "nearest", or "bicubic"
        
    Returns:
        dict containing transformed image and metadata.
    """
    # Assuming valid_node_mask is 1D [Nmax] where True means valid node
    # Or maybe it's just the number of nodes if we don't have the mask, but we are supposed to have valid_node_mask.
    # Let's count valid nodes
    n = int(valid_node_mask.sum().item())
    if n == 0:
        # Fallback to 1 if empty graph to prevent 0-size tensors
        n = 1
        
    crop = image[:, :n, :n]
    
    if input_mode == "resize_bilinear":
        # Add batch dim for interpolate
        crop_b = crop.unsqueeze(0)
        
        if resize_mode in ["bilinear", "bicubic"]:
            resized = F.interpolate(crop_b, size=(input_size, input_size), mode=resize_mode, align_corners=False)
        else:
            resized = F.interpolate(crop_b, size=(input_size, input_size), mode=resize_mode)
            
        resized = resized.squeeze(0)
        resized = torch.clamp(resized, 0.0, 1.0)
        
        return {
            "image": resized,
            "n_valid": n,
            "valid_token_mask": None,
            "input_mode": input_mode,
            "model_input_size": input_size
        }
        
    elif input_mode == "patch_aligned_repeat":
        if token_grid_size is None:
            raise ValueError("token_grid_size must be provided for patch_aligned_repeat")
            
        if n > token_grid_size:
            raise ValueError(
                f"patch_aligned_repeat requires n_valid <= token_grid_size, "
                f"but got n_valid={n}, token_grid_size={token_grid_size}. "
                f"Graph cannot be represented without truncation."
            )
            
        crop = image[:, :n, :n]
            
        # Repeat each pixel into patch_size x patch_size
        # crop shape: [3, n, n]
        repeated = crop.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
        
        target_size = token_grid_size * patch_size
        out_image = torch.zeros((image.size(0), target_size, target_size), dtype=image.dtype, device=image.device)
        
        h_repeated, w_repeated = repeated.shape[-2:]
        out_image[:, :h_repeated, :w_repeated] = repeated
        
        out_image = torch.clamp(out_image, 0.0, 1.0)
        
        valid_token_mask = torch.zeros((token_grid_size, token_grid_size), dtype=torch.bool, device=image.device)
        valid_token_mask[:n, :n] = True
        
        return {
            "image": out_image,
            "n_valid": n,
            "valid_token_mask": valid_token_mask,
            "input_mode": input_mode,
            "model_input_size": target_size,
            "token_grid_size": token_grid_size,
            "patch_size": patch_size
        }
        
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
