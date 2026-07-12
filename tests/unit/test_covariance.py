import torch
from vino.transforms.covariance import compute_node_covariance

def test_node_covariance():
    X = torch.randn(5, 10)
    A_prop = torch.eye(5)
    K = compute_node_covariance(X, A_prop)
    assert K.shape == (5, 5)
    # Symmetry
    assert torch.allclose(K, K.T, atol=1e-5)
