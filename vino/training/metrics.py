import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

def compute_binary_metrics(preds: torch.Tensor, targets: torch.Tensor):
    p = torch.sigmoid(preds).cpu().numpy()
    t = targets.cpu().numpy()
    
    acc = (np.round(p) == t).mean()
    if len(np.unique(t)) < 2:
        return {"roc_auc": None, "accuracy": float(acc)}
        
    auc = roc_auc_score(t, p)
    return {"roc_auc": float(auc), "accuracy": float(acc)}

def compute_multitask_metrics(preds: torch.Tensor, targets: torch.Tensor):
    p = torch.sigmoid(preds).cpu().numpy()
    t = targets.cpu().numpy()
    aucs, accuracies = [], []
    for task in range(t.shape[1]):
        mask = ~np.isnan(t[:, task])
        if not mask.any():
            continue
        truth, scores = t[mask, task], p[mask, task]
        accuracies.append(float((np.round(scores) == truth).mean()))
        if len(np.unique(truth)) >= 2:
            aucs.append(float(roc_auc_score(truth, scores)))
    return {
        "roc_auc": float(np.mean(aucs)) if aucs else None,
        "accuracy": float(np.mean(accuracies)) if accuracies else None,
        "num_valid_auc_tasks": len(aucs),
    }

def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor):
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    mse = mean_squared_error(t, p)
    mae = mean_absolute_error(t, p)
    return {"rmse": float(np.sqrt(mse)), "mae": float(mae), "mse": float(mse)}


def compute_ogb_graph_metrics(dataset_name: str, preds: torch.Tensor, targets: torch.Tensor):
    """Use the benchmark's official evaluator rather than reimplementing semantics."""
    try:
        from ogb.graphproppred import Evaluator
    except ImportError as exc:
        raise ImportError("ogb is required for official OGB evaluation") from exc
    ogb_name = dataset_name if dataset_name.startswith("ogbg-") else f"ogbg-{dataset_name}"
    evaluator = Evaluator(name=ogb_name)
    result = evaluator.eval({"y_true": targets.numpy(), "y_pred": preds.numpy()})
    return {key.replace("rocauc", "roc_auc"): float(value) for key, value in result.items()}
