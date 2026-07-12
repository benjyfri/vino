import os
import subprocess
import json
from omegaconf import OmegaConf
from vino.utils.hashing import hash_config

def test_train_tiny_frozen():
    # We assume test_build_synthetic_images runs first or we just call the training script
    # with the smoke config. It handles dataset building if not found (in this POC it mocks it).
    config_path = "configs/experiments/smoke_synthetic.yaml"
    result = subprocess.run([
        "python", "scripts/train.py", 
        "--config", config_path
    ], capture_output=True, text=True)
    assert result.returncode == 0
    
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    out_dir = os.path.join("outputs", "run_" + hash_config(config_dict))
    
    assert os.path.exists(out_dir)
    required_files = [
        "checkpoint_best.pt",
        "checkpoint_last.pt",
        "result.json",
        "metrics.jsonl",
        "config_resolved.yaml"
    ]
    for f in required_files:
        assert os.path.exists(os.path.join(out_dir, f))
        
    with open(os.path.join(out_dir, "result.json"), "r") as f:
        res = json.load(f)
        
    required_fields = [
        "dataset", "task_type", "model_name", "freeze_mode", "seed",
        "best_epoch", "final_epoch", "train_loss", "best_train_loss",
        "val_metric", "test_metric", "metric_for_best", "metric_mode",
        "num_trainable_params", "num_total_params", "output_dir",
        "data_dir", "timestamp"
    ]
    for field in required_fields:
        assert field in res
        
    assert res["model_name"] == "tiny_dummy"
