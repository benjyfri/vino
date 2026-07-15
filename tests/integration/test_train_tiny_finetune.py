import os
import subprocess
import json
from omegaconf import OmegaConf

def test_train_tiny_finetune(tmp_path, synthetic_run_inputs):
    config_path, data_dir = synthetic_run_inputs
    out_dir = tmp_path / "finetune_run"
    result = subprocess.run([
        "python", "scripts/train.py", 
        "--config", str(config_path), "--data_dir", str(data_dir), "--output_dir", str(out_dir)
    ], capture_output=True, text=True)
    assert result.returncode == 0
    
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
