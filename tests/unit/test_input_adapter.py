import torch
from vino.models.input_adapter import Conv1x1Adapter

def test_input_adapter():
    adapter = Conv1x1Adapter()
    x = torch.randn(2, 3, 32, 32)
    out = adapter(x)
    assert out.shape == (2, 3, 32, 32)
