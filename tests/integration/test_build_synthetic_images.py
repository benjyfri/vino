import os
import subprocess

def test_build_synthetic_images(tmp_path):
    result = subprocess.run([
        "python", "scripts/build_graph_images.py", 
        "--config", "configs/experiments/smoke_synthetic.yaml", "--output-dir", str(tmp_path)
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("synthetic/*/manifest.json"))
