import os
import subprocess
import json
import pytest

def test_builder_manifest(tmp_path):
    config_path = "configs/experiments/smoke_synthetic.yaml"
    result = subprocess.run([
        "python", "scripts/build_graph_images.py", 
        "--config", config_path, "--output-dir", str(tmp_path)
    ], capture_output=True, text=True)
    assert result.returncode == 0
    
    # Check manifest
    # We don't easily know the output dir, but we can search outputs/ or data/processed
    # In smoke_synthetic, it writes to data/processed/synthetic/
    base_dir = str(tmp_path / "synthetic")
    assert os.path.exists(base_dir)
    
    # find newest manifest
    manifest_files = []
    for root, dirs, files in os.walk(base_dir):
        if "manifest.json" in files:
            manifest_files.append(os.path.join(root, "manifest.json"))
            
    assert len(manifest_files) > 0
    with open(manifest_files[-1], "r") as f:
        manifest = json.load(f)
        
    assert "num_total" in manifest
    assert "num_success" in manifest
    assert "num_failed" in manifest
    assert manifest["num_success"] > 0
    
def test_builder_fail(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("dataset:\n  dataset: unknown\nimage: {}\n")
    result = subprocess.run([
        "python", "scripts/build_graph_images.py", "--config", str(config),
        "--output-dir", str(tmp_path / "cache")
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Unknown dataset unknown" in result.stderr
