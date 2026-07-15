from __future__ import annotations

import torch

from .losses import binary_cross_entropy_with_logits, masked_multitask_bce, mse_loss
from .metrics import compute_binary_metrics, compute_multitask_metrics, compute_regression_metrics, compute_ogb_graph_metrics


def task_loss(predictions, targets, task_type: str, pos_weight=None):
    if task_type == "binary_classification":
        return binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight)
    if task_type == "multitask_classification":
        return masked_multitask_bce(predictions, targets)
    if task_type == "regression":
        return mse_loss(predictions, targets)
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def task_metrics(predictions, targets, task_type: str, dataset_name: str | None = None):
    if dataset_name in {"molhiv", "ogbg-molhiv"}:
        return compute_ogb_graph_metrics(dataset_name, predictions, targets)
    if task_type == "binary_classification":
        return compute_binary_metrics(predictions, targets)
    if task_type == "multitask_classification":
        return compute_multitask_metrics(predictions, targets)
    if task_type == "regression":
        return compute_regression_metrics(predictions, targets)
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def evaluate_model(model, loader, task_type: str, *, channel_transform=None, pos_weight=None,
                   dataset_name: str | None = None):
    device = next(model.parameters()).device
    model.eval()
    weighted_loss, sample_count = 0.0, 0
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if channel_transform is not None:
                images = channel_transform(images)
            target = batch["y"].to(device, non_blocking=True)
            valid_token_mask = batch.get("valid_token_mask")
            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.to(device, non_blocking=True)
            output = model(images, valid_token_mask=valid_token_mask)
            loss = task_loss(output, target, task_type, pos_weight=pos_weight)
            count = target.shape[0]
            weighted_loss += float(loss) * count
            sample_count += count
            predictions.append(output.detach().cpu())
            targets.append(target.detach().cpu())
    if sample_count == 0:
        return {"loss": float("inf")}
    metrics = task_metrics(torch.cat(predictions), torch.cat(targets), task_type, dataset_name)
    metrics["loss"] = weighted_loss / sample_count
    return metrics
