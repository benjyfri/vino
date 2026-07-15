import hashlib
import json
from omegaconf import OmegaConf

def hash_config(config_dict: dict) -> str:
    """Creates a deterministic hash of a config dict."""
    if OmegaConf.is_config(config_dict):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)
    encoded = json.dumps(config_dict, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:8]

def hash_preprocessing_config(config_dict: dict) -> str:
    """Hash only inputs that determine cached graph-image contents."""
    if OmegaConf.is_config(config_dict):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)
    return hash_config({
        "cache_schema": 2,
        **{key: config_dict.get(key, {}) for key in ("dataset", "image")},
    })
