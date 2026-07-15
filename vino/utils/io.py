import yaml
from omegaconf import OmegaConf, DictConfig
import json
import torch

CACHED_GRAPH_KEYS = {"image", "valid_node_mask", "valid_pixel_mask", "y", "graph_id", "metadata"}
RESULT_KEYS = {"dataset", "task_type", "model_name", "seed", "best_epoch", "test_metric", "output_dir", "data_dir"}

def validate_cached_graph(data: dict) -> None:
    missing = CACHED_GRAPH_KEYS - data.keys()
    if missing:
        raise ValueError(f"Cached graph is missing fields: {sorted(missing)}")
    image = data["image"]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Cached image must have shape [3, H, W], got {tuple(image.shape)}")
    if data["valid_pixel_mask"].shape != image.shape[-2:]:
        raise ValueError("valid_pixel_mask shape does not match image")
    if image.shape[-1] != image.shape[-2]:
        raise ValueError("Cached graph image must be square")
    if not image.is_floating_point() or not image.isfinite().all():
        raise ValueError("Cached image must be a finite floating-point tensor")
    node_mask = data["valid_node_mask"]
    if node_mask.ndim != 1 or node_mask.shape[0] != image.shape[-1] or node_mask.dtype != torch.bool:
        raise ValueError("valid_node_mask must be boolean and match image side length")
    n = int(node_mask.sum())
    expected_prefix = torch.arange(node_mask.numel()) < n
    if not torch.equal(node_mask.cpu(), expected_prefix):
        raise ValueError("valid_node_mask must be a contiguous valid prefix")
    expected_pixels = node_mask[:, None] & node_mask[None, :]
    if not torch.equal(data["valid_pixel_mask"].cpu(), expected_pixels.cpu()):
        raise ValueError("valid_pixel_mask is inconsistent with valid_node_mask")
    if not isinstance(data["graph_id"], str) or not data["graph_id"]:
        raise ValueError("graph_id must be a non-empty string")
    if not data["y"].isfinite().logical_or(data["y"].isnan()).all():
        raise ValueError("y contains infinite values")

def validate_result(data: dict) -> None:
    missing = RESULT_KEYS - data.keys()
    if missing:
        raise ValueError(f"Result is missing fields: {sorted(missing)}")

def load_config(path: str) -> DictConfig:
    conf = OmegaConf.load(path)
    return conf

def save_json(data, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_jsonl(data_list, path: str):
    with open(path, 'a') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')
