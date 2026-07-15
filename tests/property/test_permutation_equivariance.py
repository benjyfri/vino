import torch
from vino.transforms.covariance import compute_node_covariance


def test_permutation_equivariance():
    x = torch.randn(7, 5)
    adjacency = torch.rand(7, 7)
    adjacency = (adjacency + adjacency.T) / 2
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    original = compute_node_covariance(x, adjacency, seed=11)
    permuted = compute_node_covariance(x[permutation], adjacency[permutation][:, permutation], seed=11)
    assert torch.allclose(permuted, original[permutation][:, permutation], atol=1e-5)
