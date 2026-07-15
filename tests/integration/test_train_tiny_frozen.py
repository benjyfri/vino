import os
import subprocess
import json
from omegaconf import OmegaConf

def test_train_tiny_frozen(tmp_path, synthetic_run_inputs):
    # We assume test_build_synthetic_images runs first or we just call the training script
    # with the smoke config. It handles dataset building if not found (in this POC it mocks it).
    config_path, data_dir = synthetic_run_inputs
    out_dir = tmp_path / "frozen_run"
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

    evaluation = subprocess.run([
        "python", "scripts/evaluate.py", "--config", str(config_path),
        "--checkpoint", str(out_dir / "checkpoint_best.pt"), "--data-dir", str(data_dir),
    ], capture_output=True, text=True)
    assert evaluation.returncode == 0, evaluation.stderr
    assert '"num_samples"' in evaluation.stdout

    collision = subprocess.run([
        "python", "scripts/train.py", "--config", str(config_path), "--data_dir", str(data_dir),
        "--output_dir", str(out_dir),
    ], capture_output=True, text=True)
    assert collision.returncode != 0
    assert "Output directory already exists" in collision.stderr
