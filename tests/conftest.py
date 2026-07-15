import pytest
import os
import sys
import subprocess
import torch
from omegaconf import OmegaConf

# Ensure vino is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def synthetic_run_inputs(tmp_path):
    cfg = OmegaConf.load("configs/experiments/smoke_synthetic.yaml")
    cfg.dataset.num_graphs = 12
    cfg.image.n_max = 32
    cfg.image.node_cov.h = 8
    cfg.image.edge_cov.h = 8
    cfg.model.input_mode = "resize_bilinear"
    cfg.model.input_size = 32
    cfg.train.epochs = 1
    cfg.train.num_workers = 0
    config_path = tmp_path / "smoke.yaml"
    OmegaConf.save(cfg, config_path)
    cache_root = tmp_path / "cache"
    result = subprocess.run([
        sys.executable, "scripts/build_graph_images.py", "--config", str(config_path),
        "--output-dir", str(cache_root),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache_dir = next((cache_root / "synthetic").iterdir())
    return config_path, cache_dir

@pytest.fixture
def tiny_cached_dataset(tmp_path):
    root = tmp_path / "cached"
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        for index in range(4):
            n = 40 if index == 3 else 8 + index
            mask = torch.zeros(64, dtype=torch.bool)
            mask[:n] = True
            torch.save({
                "image": torch.rand(3, 64, 64), "valid_node_mask": mask,
                "y": torch.tensor([index % 2], dtype=torch.float32),
                "graph_id": f"{split}_{index}",
            }, root / split / f"{index}.pt")
    return root
