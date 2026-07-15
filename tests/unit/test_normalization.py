import torch
from vino.transforms.covariance import compute_node_covariance

def test_normalization():
    result = compute_node_covariance(torch.randn(8, 3) * 1000, torch.eye(8))
    assert torch.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1
