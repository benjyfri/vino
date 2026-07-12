import os
import subprocess

def test_build_synthetic_images():
    result = subprocess.run([
        "python", "scripts/build_graph_images.py", 
        "--config", "configs/experiments/smoke_synthetic.yaml"
    ], capture_output=True, text=True)
    assert result.returncode == 0
