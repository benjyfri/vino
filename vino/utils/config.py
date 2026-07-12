import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def load_resolved_config(config_path: str | Path) -> DictConfig:
    config_path = str(config_path)
    base_cfg = OmegaConf.load(config_path)
    
    repo_root = os.getcwd()
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    for key in ["dataset", "image", "model", "train"]:
        if key in base_cfg:
            val = base_cfg[key]
            if isinstance(val, str) and (val.endswith(".yaml") or val.endswith(".yml")):
                candidate_repo_root = os.path.join(repo_root, val)
                candidate_config_dir = os.path.join(config_dir, val)
                
                if os.path.exists(candidate_repo_root):
                    resolved_path = candidate_repo_root
                elif os.path.exists(candidate_config_dir):
                    resolved_path = candidate_config_dir
                else:
                    raise FileNotFoundError(f"Could not resolve config path {val} referenced in {config_path}")
                
                sub_cfg = OmegaConf.load(resolved_path)
                base_cfg[key] = sub_cfg
            elif isinstance(val, dict) or OmegaConf.is_dict(val):
                if "_base_" in val:
                    base_path = val.pop("_base_")
                    resolved_path = os.path.join(repo_root, base_path)
                    if os.path.exists(resolved_path):
                        default_cfg = OmegaConf.load(resolved_path)
                        base_cfg[key] = OmegaConf.merge(default_cfg, val)
        else:
            base_cfg[key] = OmegaConf.create({})
                
    return base_cfg
