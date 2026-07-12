import argparse
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from vino.models.graph_image_model import GraphImageModel
from vino.training.losses import binary_cross_entropy_with_logits, masked_multitask_bce, mse_loss
from vino.training.metrics import compute_binary_metrics, compute_regression_metrics
from vino.utils.seed import seed_everything
from vino.utils.hashing import hash_config

from vino.transforms.model_input import transform_graph_image_for_model

class CachedGraphDataset(Dataset):
    def __init__(self, data_dir, model_config=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.model_config = model_config or {}
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        input_mode = self.model_config.get("input_mode")
        if input_mode:
            if "valid_node_mask" in data:
                mask = data["valid_node_mask"]
            else:
                num_nodes = data.get("num_nodes", data["image"].size(1))
                mask = torch.zeros(data["image"].size(1), dtype=torch.bool)
                mask[:num_nodes] = True
                
            transformed = transform_graph_image_for_model(
                image=data["image"],
                valid_node_mask=mask,
                input_mode=input_mode,
                input_size=self.model_config.get("input_size", 224),
                patch_size=self.model_config.get("patch_size", 16),
                token_grid_size=self.model_config.get("token_grid_size", None),
                resize_mode=self.model_config.get("resize_mode", "bilinear")
            )
            
            data["image"] = transformed["image"]
            data["n_valid"] = transformed["n_valid"]
            if transformed["valid_token_mask"] is not None:
                data["valid_token_mask"] = transformed["valid_token_mask"]
            data["input_mode"] = transformed["input_mode"]
            data["model_input_size"] = transformed["model_input_size"]
            
        return data

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    y = torch.stack([b["y"] for b in batch]).view(len(batch), -1)
    ret = {"image": images, "y": y, "graph_id": [b["graph_id"] for b in batch]}
    if "valid_token_mask" in batch[0]:
        ret["valid_token_mask"] = torch.stack([b["valid_token_mask"] for b in batch])
    if "n_valid" in batch[0]:
        ret["n_valid"] = [b["n_valid"] for b in batch]
    if "input_mode" in batch[0]:
        ret["input_mode"] = batch[0]["input_mode"]
    if "model_input_size" in batch[0]:
        ret["model_input_size"] = batch[0]["model_input_size"]
    return ret

def evaluate(model, loader, config_dict):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            valid_token_mask = batch.get("valid_token_mask")
            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.to(device, non_blocking=True)
                
            out = model(images)
            task_type = config_dict.get("train", {}).get("task_type", "unknown")
            if task_type == "binary_classification":
                loss = binary_cross_entropy_with_logits(out, y)
            elif task_type == "multitask_classification":
                loss = masked_multitask_bce(out, y)
            else:
                loss = mse_loss(out, y)
            total_loss += loss.item()
            all_preds.append(out)
            all_targets.append(y)
            
    if len(loader) == 0:
        return {"loss": float("inf")}
        
    epoch_loss = total_loss / len(loader)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    task_type = config_dict.get("train", {}).get("task_type", "unknown")
    if task_type == "binary_classification":
        metrics = compute_binary_metrics(preds, targets)
    elif task_type == "multitask_classification":
        metrics = compute_binary_metrics(preds, targets)
    else:
        metrics = compute_regression_metrics(preds, targets)
        
    metrics["loss"] = epoch_loss
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory. Defaults to outputs/run_<hash> if not set.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    
    from vino.utils.config import load_resolved_config
    config = load_resolved_config(args.config)
    
    if args.seed is not None:
        seed = int(args.seed)
        config.train.seed = seed
    else:
        seed = int(getattr(config.train, "seed", 42))
        
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    seed_everything(seed)
    
    # Mock data dir logic for smoke test
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    dataset_val = config_dict.get("dataset", "unknown")
    if isinstance(dataset_val, dict):
        dataset_name = dataset_val.get("dataset", "unknown")
    else:
        dataset_name = dataset_val

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif "data_dir" in config_dict:
        data_dir = config_dict["data_dir"]
    else:
        conf_hash = hash_config(config_dict)
        data_dir = os.path.join("data/processed", dataset_name, conf_hash)
        
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Data not found at {train_dir}. Please run build_graph_images.py first.")
        return
        
    model_config = config_dict.get("model", {})
    
    train_dataset = CachedGraphDataset(train_dir, model_config=model_config)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dir = os.path.join(data_dir, "val")
    val_loader = None
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        val_loader = DataLoader(CachedGraphDataset(val_dir, model_config=model_config), batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
        
    test_dir = os.path.join(data_dir, "test")
    test_loader = None
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_loader = DataLoader(CachedGraphDataset(test_dir, model_config=model_config), batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model_cfg = config_dict.get("model", {})
    if "backbone_name" in model_cfg:
        model_name_val = model_cfg["backbone_name"]
    elif "name" in model_cfg:
        model_name_val = model_cfg["name"]
    elif "type" in model_cfg:
        model_name_val = model_cfg["type"]
    elif "smoke_synthetic.yaml" in args.config:
        model_name_val = "tiny_dummy"
    else:
        model_name_val = "unknown"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[device] cuda name: {torch.cuda.get_device_name(0)}", flush=True)

    model = GraphImageModel(config_dict)
    model = model.to(device)
    print("[debug] model device:", next(model.parameters()).device, flush=True)
    
    # Optim
    head_params = list(model.head.parameters()) + list(model.adapter.parameters())
    if hasattr(model, "input_stem"):
        head_params.extend([p for p in model.input_stem.parameters() if p.requires_grad])
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": config.train.lr_head},
        {"params": backbone_params, "lr": config.train.lr_backbone}
    ], weight_decay=config.train.weight_decay)
    
    epochs = config.train.epochs
    
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("outputs", "run_" + hash_config(config_dict))
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        OmegaConf.save(config, f)
        
    metrics_file = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    
    best_train_loss = float("inf")
    best_val_metric_val = None
    best_epoch = 0
    metric_for_best = "train_loss"
    metric_mode = "min"
    
    # Pre-save a best checkpoint in case epochs=0
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_best.pt"))
    
    printed_first_batch_debug = False
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            valid_token_mask = batch.get("valid_token_mask")
            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.to(device, non_blocking=True)
                
            optimizer.zero_grad()
            out = model(images)
            
            if not printed_first_batch_debug:
                printed_first_batch_debug = True
                print("[debug] first batch image:", images.device, tuple(images.shape), flush=True)
                print("[debug] first batch y:", y.device, tuple(y.shape), flush=True)
                print("[debug] first logits:", out.device, tuple(out.shape), flush=True)
                if torch.cuda.is_available() and model_name_val != "tiny_dummy":
                    assert next(model.parameters()).is_cuda, "Non-tiny model is not on CUDA"
                    assert images.is_cuda, "Input images are not on CUDA"
                    assert out.is_cuda, "Logits are not on CUDA"

            if config.train.task_type == "binary_classification":
                loss = binary_cross_entropy_with_logits(out, y)
            elif config.train.task_type == "multitask_classification":
                loss = masked_multitask_bce(out, y)
            else:
                loss = mse_loss(out, y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_loader)
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            
        log_entry = {"epoch": epoch, "train_loss": epoch_loss}
        print_msg = f"Epoch {epoch}: Train Loss {epoch_loss:.4f}"
        
        is_best = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, config_dict)
            log_entry["val_loss"] = val_metrics["loss"]
            if "accuracy" in val_metrics:
                log_entry["val_accuracy"] = val_metrics["accuracy"]
            if "roc_auc" in val_metrics:
                log_entry["val_roc_auc"] = val_metrics["roc_auc"]
                
            if "roc_auc" in val_metrics and val_metrics["roc_auc"] is not None:
                current_metric = val_metrics["roc_auc"]
                metric_for_best = "val_roc_auc"
                metric_mode = "max"
            else:
                current_metric = val_metrics["loss"]
                metric_for_best = "val_loss"
                metric_mode = "min"
                
            if best_val_metric_val is None:
                is_best = True
            else:
                if metric_mode == "max":
                    is_best = current_metric > best_val_metric_val
                else:
                    is_best = current_metric < best_val_metric_val
                    
            if is_best:
                best_val_metric_val = current_metric
                
            print_msg += f" | Val Loss: {val_metrics['loss']:.4f}"
            if "roc_auc" in val_metrics and val_metrics["roc_auc"] is not None:
                print_msg += f" | Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        else:
            if epoch_loss <= best_train_loss:
                is_best = True
                
        if is_best:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_best.pt"))
            
        print(print_msg, flush=True)
        metrics_file.write(json.dumps(log_entry) + "\n")
        metrics_file.flush()
        
    metrics_file.close()
        
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_last.pt"))
    
    import datetime
    
    if os.path.exists(os.path.join(out_dir, "checkpoint_best.pt")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "checkpoint_best.pt"), map_location=device, weights_only=False))
        
    val_final_metrics = None
    if val_loader is not None:
        val_final_metrics = evaluate(model, val_loader, config_dict)
        
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, config_dict)

    result = {
        "dataset": dataset_name,
        "task_type": config_dict.get("train", {}).get("task_type", "unknown"),
        "model_name": model_name_val,
        "freeze_mode": config_dict.get("model", {}).get("freeze_mode", "unknown"),
        "seed": seed,
        "best_epoch": best_epoch,
        "final_epoch": epochs - 1 if epochs > 0 else 0,
        "train_loss": epoch_loss if epochs > 0 else float('inf'),
        "best_train_loss": best_train_loss,
        "val_metric": val_final_metrics["roc_auc"] if val_final_metrics and "roc_auc" in val_final_metrics and val_final_metrics["roc_auc"] is not None else (val_final_metrics["loss"] if val_final_metrics else None),
        "test_metric": test_metrics["roc_auc"] if test_metrics and "roc_auc" in test_metrics and test_metrics["roc_auc"] is not None else (test_metrics["loss"] if test_metrics else None),
        "val_loss": val_final_metrics["loss"] if val_final_metrics else None,
        "test_loss": test_metrics["loss"] if test_metrics else None,
        "val_accuracy": val_final_metrics.get("accuracy") if val_final_metrics else None,
        "test_accuracy": test_metrics.get("accuracy") if test_metrics else None,
        "val_roc_auc": val_final_metrics.get("roc_auc") if val_final_metrics else None,
        "test_roc_auc": test_metrics.get("roc_auc") if test_metrics else None,
        "metric_for_best": metric_for_best,
        "metric_mode": metric_mode,
        "input_mode": model_cfg.get("input_mode"),
        "model_input_size": model_cfg.get("input_size") if model_cfg.get("input_mode") == "resize_bilinear" else (model_cfg.get("token_grid_size", 0) * model_cfg.get("patch_size", 16) if model_cfg.get("input_mode") == "patch_aligned_repeat" else None),
        "patch_size": model_cfg.get("patch_size"),
        "token_grid_size": model_cfg.get("token_grid_size"),
        "resize_mode": model_cfg.get("resize_mode"),
        "input_stem_type": model_cfg.get("input_stem", {}).get("type", "residual_cnn"),
        "input_stem_enabled": model_cfg.get("input_stem", {}).get("enabled", False),
        "input_stem_hidden_channels": model_cfg.get("input_stem", {}).get("hidden_channels", 32),
        "input_stem_depth": model_cfg.get("input_stem", {}).get("depth", 2),
        "input_stem_residual_scale_init": model_cfg.get("input_stem", {}).get("residual_scale_init", 0.0),
        "patch_embed_trainable": model_cfg.get("freeze_mode") in ["train_patch_embed", "train_stem_and_patch_embed"],
        "patch_embed_trainable_params": sum(p.numel() for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d) for p in m.parameters() if p.requires_grad) if hasattr(model, "backbone") else 0,
        "stem_trainable_params": sum(p.numel() for p in model.input_stem.parameters() if p.requires_grad) if hasattr(model, "input_stem") else 0,
        "adapter_trainable_params": sum(p.numel() for p in model.adapter.parameters() if p.requires_grad) if hasattr(model, "adapter") else 0,
        "head_trainable_params": sum(p.numel() for p in model.head.parameters() if p.requires_grad) if hasattr(model, "head") else 0,
        "backbone_trainable_params": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) if hasattr(model, "backbone") else 0,
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "num_total_params": sum(p.numel() for p in model.parameters()),
        "output_dir": out_dir,
        "data_dir": data_dir,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
        
if __name__ == "__main__":
    main()
