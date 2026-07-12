import torch
from vino.transforms.edge_covariance import compute_edge_covariance

def test_edge_covariance():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.ones(2, 3)
    A_prop = torch.eye(2)
    K = compute_edge_covariance(2, edge_index, edge_attr, A_prop)
    assert K.shape == (2, 2)
    assert torch.allclose(K, K.T, atol=1e-5)
