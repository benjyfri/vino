import pytest
import torch
from vino.utils.io import validate_cached_graph


def test_saved_image_schema():
    data = {
        "image": torch.zeros(3, 4, 4), "valid_node_mask": torch.ones(4, dtype=torch.bool),
        "valid_pixel_mask": torch.ones(4, 4, dtype=torch.bool), "y": torch.tensor([1]),
        "graph_id": "g", "metadata": {},
    }
    validate_cached_graph(data)
    with pytest.raises(ValueError, match="shape"):
        validate_cached_graph({**data, "image": torch.zeros(2, 4, 4)})
