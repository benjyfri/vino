import subprocess
import os

def test_cli_help():
    scripts = [
        "scripts/build_graph_images.py",
        "scripts/train.py",
        "scripts/download_dinov3.py"
    ]
    
    for script in scripts:
        result = subprocess.run(["python", script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
