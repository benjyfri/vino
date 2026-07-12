import torch
import torch.nn.functional as F

def binary_cross_entropy_with_logits(preds: torch.Tensor, targets: torch.Tensor, pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=preds.device)
    return F.binary_cross_entropy_with_logits(preds, targets.float(), pos_weight=pos_weight)

def masked_multitask_bce(preds: torch.Tensor, targets: torch.Tensor):
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(preds[mask], targets[mask].float())

def mse_loss(preds: torch.Tensor, targets: torch.Tensor):
    return F.mse_loss(preds, targets.float())
