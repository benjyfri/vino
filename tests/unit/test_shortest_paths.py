import torch
from vino.transforms.shortest_paths import compute_apsp_heat_kernel

def test_shortest_paths():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    W = compute_apsp_heat_kernel(3, edge_index)
    assert W.shape == (3, 3)
    assert W[0, 0] == 1.0
