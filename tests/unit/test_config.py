import pytest
import os
from omegaconf import OmegaConf, DictConfig
from vino.utils.config import load_resolved_config

def test_load_resolved_config():
    # Test loading BBBP experiment config
    config_path = "configs/experiments/bbbp_frozen_vits16.yaml"
    
    if not os.path.exists(config_path):
        pytest.skip(f"{config_path} does not exist.")
        
    cfg = load_resolved_config(config_path)
    
    assert isinstance(cfg.dataset, DictConfig), "cfg.dataset should be a dict-like DictConfig"
    assert cfg.dataset.dataset == "bbbp", "dataset name should be bbbp"
    assert isinstance(cfg.image, DictConfig), "cfg.image should be a DictConfig"
    assert isinstance(cfg.model, DictConfig), "cfg.model should be a DictConfig"
    assert isinstance(cfg.train, DictConfig), "cfg.train should be a DictConfig"

def test_load_resolved_config_no_throw():
    # A lightweight test that loading BBBP experiment config does not throw
    config_path = "configs/experiments/bbbp_frozen_vits16.yaml"
    
    if not os.path.exists(config_path):
        pytest.skip(f"{config_path} does not exist.")
        
    try:
        cfg = load_resolved_config(config_path)
    except Exception as e:
        pytest.fail(f"load_resolved_config threw an exception: {e}")

def test_bbbp_config_image_fields():
    config_path = "configs/experiments/bbbp_frozen_vits16.yaml"
    if not os.path.exists(config_path):
        pytest.skip(f"{config_path} does not exist.")
    cfg = load_resolved_config(config_path)
    
    assert "n_max" in cfg.image, "cfg.image missing n_max"
    assert "topology" in cfg.image, "cfg.image missing topology"
    assert "node_cov" in cfg.image, "cfg.image missing node_cov"
    assert "edge_cov" in cfg.image, "cfg.image missing edge_cov"
