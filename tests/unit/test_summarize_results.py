import json
import os
import subprocess

def test_summarize_results(tmp_path):
    # Create fake results
    res1 = {
        "input_stem_type": "residual_cnn",
        "freeze_mode": "frozen",
        "input_mode": "resize_bilinear",
        "seed": 42,
        "val_roc_auc": 0.8,
        "test_roc_auc": 0.85
    }
    res2 = {
        "input_stem_type": "residual_cnn",
        "freeze_mode": "frozen",
        "input_mode": "resize_bilinear",
        "seed": 43,
        "val_roc_auc": 0.82,
        "test_roc_auc": 0.87
    }
    
    os.makedirs(tmp_path / "run1")
    os.makedirs(tmp_path / "run2")
    
    with open(tmp_path / "run1" / "result.json", "w") as f:
        json.dump(res1, f)
    with open(tmp_path / "run2" / "result.json", "w") as f:
        json.dump(res2, f)
        
    csv_path = tmp_path / "summary.csv"
        
    result = subprocess.run([
        "python", "scripts/summarize_results.py", 
        "--glob", f"{tmp_path}/*/result.json",
        "--csv", str(csv_path)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Found 2 results" in result.stdout
    assert "0.8100" in result.stdout # val mean
    assert "0.8600" in result.stdout # test mean
    
    assert csv_path.exists()
