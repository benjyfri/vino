import torch
from vino.training.losses import binary_cross_entropy_with_logits
from vino.training.metrics import compute_multitask_metrics

def test_bce_loss():
    preds = torch.tensor([0.0, 0.0])
    targets = torch.tensor([0, 1])
    loss = binary_cross_entropy_with_logits(preds, targets)
    assert loss.item() > 0

def test_multitask_metrics_mask_missing_labels():
    preds = torch.tensor([[4.0, 0.0], [-4.0, 2.0], [3.0, -2.0]])
    targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0], [1.0, 0.0]])
    metrics = compute_multitask_metrics(preds, targets)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["num_valid_auc_tasks"] == 2
