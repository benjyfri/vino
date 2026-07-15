import torch
from vino.transforms.graph_image import make_graph_image
from vino.data.graph_record import GraphRecord

def test_graph_image():
    record = GraphRecord(
        graph_id="test_0",
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([1]),
        num_nodes=3
    )
    config = {
        "image": {
            "n_max": 256, "pad_value": 0.0,
            "topology": {"sigma": 0.35, "power": 2.0},
            "node_cov": {"h": 64, "seed": 42},
            "edge_cov": {"h": 64, "seed": 42},
            "propagation": {"powers": [0, 1, 2], "weights": [1.0, 0.5, 0.25]}
        }
    }
    
    out = make_graph_image(record, config)
    assert out["image"].shape == (3, 256, 256)

def test_topology_only_ablation_keeps_three_channel_contract():
    record = GraphRecord("g", torch.randn(4, 2), torch.tensor([[0, 1], [1, 0]]), torch.tensor([1]), num_nodes=4)
    from vino.utils.config import load_resolved_config
    cfg = {"image": dict(load_resolved_config("configs/experiments/smoke_synthetic.yaml").image)}
    cfg["image"] = dict(load_resolved_config("configs/experiments/smoke_synthetic.yaml").image)
    cfg["image"]["channels"] = ["topology"]
    out = make_graph_image(record, cfg)
    assert out["image"].shape[0] == 3
    assert torch.count_nonzero(out["image"][1:]) == 0
    assert out["valid_node_mask"].sum() == 4
