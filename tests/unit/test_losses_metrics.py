import torch
from vino.training.losses import binary_cross_entropy_with_logits

def test_bce_loss():
    preds = torch.tensor([0.0, 0.0])
    targets = torch.tensor([0, 1])
    loss = binary_cross_entropy_with_logits(preds, targets)
    assert loss.item() > 0
