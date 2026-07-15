import torch
from .shortest_paths import compute_apsp_heat_kernel
from .canonicalize import canonicalize_topology
from .covariance import compute_node_covariance
from .edge_covariance import compute_edge_covariance
from ..data.graph_record import GraphRecord
from typing import Dict, Any

def get_prop_operator(num_nodes: int, edge_index: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes))
    if edge_index.size(1) > 0:
        A[edge_index[0], edge_index[1]] = 1.0
        A[edge_index[1], edge_index[0]] = 1.0
    A.fill_diagonal_(1.0)
    
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=eps), -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def make_graph_image(record: GraphRecord, config: Dict[str, Any]) -> Dict[str, Any]:
    record.validate()
    n = record.num_nodes
    img_cfg = config["image"]
    N_max = img_cfg["n_max"]
    
    if n > N_max:
        raise ValueError(f"Graph size {n} exceeds N_max {N_max}")
        
    # Topology
    topology_mode = img_cfg["topology"].get("mode", "apsp_heat")
    if topology_mode == "raw_adj":
        W_top = torch.zeros((n, n), dtype=torch.float32)
        if record.edge_index.numel():
            W_top[record.edge_index[0], record.edge_index[1]] = 1.0
        W_top.fill_diagonal_(1.0)
    elif topology_mode == "apsp_heat":
        W_top = compute_apsp_heat_kernel(
            n, record.edge_index, sigma=img_cfg["topology"]["sigma"],
            power=img_cfg["topology"]["power"],
            disconnected_value=img_cfg["topology"].get("disconnected_value", 0.0),
            diagonal_value=img_cfg["topology"].get("diagonal_value", 1.0),
        )
    else:
        raise ValueError(f"Unknown topology mode: {topology_mode}")
    
    canonical_mode = img_cfg.get("canonicalization", {}).get("mode", "fiedler_apsp")
    if canonical_mode == "random":
        generator = torch.Generator().manual_seed(int(img_cfg["canonicalization"].get("seed", 42)))
        order, meta = torch.randperm(n, generator=generator), {"canonicalization_unstable": False}
    elif canonical_mode == "fiedler_apsp":
        canonical_cfg = img_cfg.get("canonicalization", {})
        order, meta = canonicalize_topology(
            W_top,
            eigengap_tol=canonical_cfg.get("eigengap_tol", 1e-6),
            tie_tol=canonical_cfg.get("tie_tol", 1e-8),
            lex_round_decimals=canonical_cfg.get("lex_round_decimals", 8),
        )
    else:
        raise ValueError(f"Unknown canonicalization mode: {canonical_mode}")
    
    # A_prop
    A_prop = get_prop_operator(n, record.edge_index)
    
    enabled = set(img_cfg.get("channels", ["topology", "node_cov", "edge_cov"]))
    node_cfg = img_cfg.get("node_cov", {})
    if "node_cov" in enabled and node_cfg.get("enabled", True):
        x = record.x if record.x is not None else torch.zeros((n, 1))
        K_node = compute_node_covariance(
            x, A_prop, h_node=img_cfg["node_cov"]["h"], seed=img_cfg["node_cov"]["seed"],
            powers=img_cfg["propagation"]["powers"], weights=img_cfg["propagation"]["weights"],
            robust_quantile=node_cfg.get("robust_quantile", 0.95),
            clip=node_cfg.get("clip", 1.0),
        )
    else:
        K_node = torch.zeros_like(W_top)
    
    # Edge Covariance
    edge_cfg = img_cfg.get("edge_cov", {})
    if "edge_cov" in enabled and edge_cfg.get("enabled", True):
        edge_attr = record.edge_attr
        if edge_attr is None or edge_attr.size(1) == 0:
            edge_attr = torch.ones((record.edge_index.size(1), 1))
        K_edge = compute_edge_covariance(
            n, record.edge_index, edge_attr, A_prop, h_edge=img_cfg["edge_cov"]["h"],
            seed=img_cfg["edge_cov"]["seed"], powers=img_cfg["propagation"]["powers"],
            weights=img_cfg["propagation"]["weights"],
            robust_quantile=edge_cfg.get("robust_quantile", 0.95),
            clip=edge_cfg.get("clip", 1.0),
        )
    else:
        K_edge = torch.zeros_like(W_top)
    
    # Apply canonical order
    W_top = W_top[order][:, order]
    K_node = K_node[order][:, order]
    K_edge = K_edge[order][:, order]
    
    # Normalize topology to [0,1] if not already (it is by construction mostly)
    
    # Stack
    image = torch.stack([
        W_top if ("topology" in enabled or "raw_adj" in enabled) else torch.zeros_like(W_top),
        K_node if "node_cov" in enabled else torch.zeros_like(K_node),
        K_edge if "edge_cov" in enabled else torch.zeros_like(K_edge),
    ], dim=0)
    
    # Pad
    storage = img_cfg.get("storage", "padded")
    if storage == "cropped":
        padded_image = image
        mask_size = n
    else:
        padded_image = torch.full((3, N_max, N_max), img_cfg["pad_value"], dtype=torch.float32)
        padded_image[:, :n, :n] = image
        mask_size = N_max
    
    valid_node_mask = torch.zeros(mask_size, dtype=torch.bool)
    valid_node_mask[:n] = True
    
    valid_pixel_mask = valid_node_mask.unsqueeze(0) & valid_node_mask.unsqueeze(1)
    
    # Update meta
    meta["num_nodes"] = n
    meta["num_edges"] = record.edge_index.size(1)
    meta["cache_format_version"] = int(img_cfg.get("cache_format_version", 2))
    meta["storage"] = storage
    
    return {
        "image": padded_image,
        "valid_node_mask": valid_node_mask,
        "valid_pixel_mask": valid_pixel_mask,
        "y": record.y,
        "graph_id": record.graph_id,
        "metadata": meta
    }
