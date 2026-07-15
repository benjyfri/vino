import torch
from vino.transforms.covariance import compute_node_covariance


def test_random_projection_invariances():
    x = torch.randn(6, 4)
    adjacency = torch.eye(6)
    first = compute_node_covariance(x, adjacency, seed=7)
    second = compute_node_covariance(x, adjacency, seed=7)
    assert torch.equal(first, second)
    assert torch.allclose(first, first.T)
    assert torch.isfinite(first).all()
    assert first.min() >= 0 and first.max() <= 1
