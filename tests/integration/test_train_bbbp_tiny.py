import subprocess
import os
import json
import glob

def run_and_check(config_path, expected_mode, tmp_path, data_dir):
    out_dir = str(tmp_path / "run")
    result = subprocess.run([
        "python", "scripts/train.py", "--config", config_path, "--output_dir", out_dir,
        "--data_dir", str(data_dir), "--num-workers", "0"
    ], capture_output=True, text=True)
    
    if expected_mode == "patch_aligned_repeat":
        assert result.returncode != 0, "Expected patch_aligned_repeat to fail on bbbp_5.pt"
        assert "ValueError" in result.stderr
        assert "n_valid=" in result.stderr
        assert "token_grid_size=" in result.stderr
        return
        
    assert result.returncode == 0, f"Training failed for {config_path}:\n{result.stderr}\n{result.stdout}"
    
    result_json_path = os.path.join(out_dir, "result.json")
    assert os.path.exists(result_json_path)
    
    with open(result_json_path, "r") as f:
        data = json.load(f)
        
    assert data["dataset"] == "bbbp"
    assert data["model_name"] == "tiny_dummy"
    assert data["val_loss"] is not None
    assert data["test_loss"] is not None
    assert data["val_metric"] is not None
    assert data["input_mode"] == expected_mode
    if expected_mode == "resize_bilinear":
        assert data["model_input_size"] == 224
    elif expected_mode == "patch_aligned_repeat":
        assert data["model_input_size"] == 512
    
    assert data["metric_for_best"] in ["val_loss", "val_roc_auc"]
    
    assert os.path.exists(os.path.join(out_dir, "checkpoint_best.pt"))
    assert os.path.exists(os.path.join(out_dir, "checkpoint_last.pt"))
    assert os.path.exists(os.path.join(out_dir, "config_resolved.yaml"))
    
    metrics_jsonl_path = os.path.join(out_dir, "metrics.jsonl")
    assert os.path.exists(metrics_jsonl_path)
    
    with open(metrics_jsonl_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        last_metric = json.loads(lines[-1])
        assert "val_loss" in last_metric

def test_bbbp_debug_tiny(tmp_path, tiny_cached_dataset):
    run_and_check("configs/experiments/bbbp_debug_tiny.yaml", "resize_bilinear", tmp_path, tiny_cached_dataset)

def test_bbbp_debug_tiny_seed(tmp_path, tiny_cached_dataset):
    out_dir = str(tmp_path / "run")
    config_path = "configs/experiments/bbbp_debug_tiny.yaml"
    result = subprocess.run([
        "python", "scripts/train.py", "--config", config_path, "--output_dir", out_dir,
        "--data_dir", str(tiny_cached_dataset), "--num-workers", "0", "--seed", "123"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    
    result_json_path = os.path.join(out_dir, "result.json")
    with open(result_json_path, "r") as f:
        data = json.load(f)
    assert data["seed"] == 123

def test_bbbp_debug_tiny_resize(tmp_path, tiny_cached_dataset):
    run_and_check("configs/experiments/bbbp_debug_tiny_resize.yaml", "resize_bilinear", tmp_path, tiny_cached_dataset)

def test_bbbp_debug_tiny_patchalign(tmp_path, tiny_cached_dataset):
    run_and_check("configs/experiments/bbbp_debug_tiny_patchalign.yaml", "patch_aligned_repeat", tmp_path, tiny_cached_dataset)
