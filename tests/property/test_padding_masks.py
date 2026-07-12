import torch
from vino.transforms.graph_image import make_graph_image
from vino.data.graph_record import GraphRecord

def test_padding_masks():
    record = GraphRecord("test", torch.randn(3, 2), torch.zeros((2,0), dtype=torch.long), torch.tensor([0]), num_nodes=3)
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
    
    assert out["valid_node_mask"].sum() == 3
    assert out["valid_pixel_mask"].sum() == 9
