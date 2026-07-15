import yaml
from omegaconf import OmegaConf, DictConfig
import json

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
