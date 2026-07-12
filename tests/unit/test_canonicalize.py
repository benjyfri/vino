import torch
from vino.transforms.canonicalize import canonicalize_topology

def test_canonicalize_topology():
    W = torch.tensor([[1.0, 0.5],
                      [0.5, 1.0]])
    order, meta = canonicalize_topology(W)
    assert len(order) == 2
    assert not meta["canonicalization_unstable"]
