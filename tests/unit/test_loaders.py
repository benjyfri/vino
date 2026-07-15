from unittest.mock import patch
import torch
from vino.data.pyg_loaders import load_bbbp


class FakeGraph:
    def __init__(self, label):
        self.x = torch.ones(3, 2)
        self.edge_index = torch.tensor([[0, 1], [1, 0]])
        self.edge_attr = torch.ones(2, 1)
        self.y = torch.tensor([label])
        self.num_nodes = 3


class FakeMoleculeNet:
    def __init__(self, *args, **kwargs):
        self.items = [FakeGraph(i % 2) for i in range(40)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_load_bbbp_uses_reproducible_stratified_splits():
    with patch("torch_geometric.datasets.MoleculeNet", FakeMoleculeNet):
        first = load_bbbp(split_seed=17)
        second = load_bbbp(split_seed=17)
    assert [(r.graph_id, r.split) for r in first] == [(r.graph_id, r.split) for r in second]
    assert {r.split for r in first} == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert {int(r.y.item()) for r in first if r.split == split} == {0, 1}
