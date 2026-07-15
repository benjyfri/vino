import argparse
import os
import json
import platform
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from vino.models.graph_image_model import GraphImageModel
from vino.training.losses import binary_cross_entropy_with_logits, masked_multitask_bce, mse_loss
from vino.training.metrics import compute_binary_metrics, compute_multitask_metrics, compute_regression_metrics
from vino.utils.seed import seed_everything, seed_worker
from vino.utils.hashing import hash_config, hash_preprocessing_config

from vino.transforms.model_input import transform_graph_image_for_model
from vino.utils.io import validate_result
from vino.utils.sweep import apply_frozen_sweep_overrides

class CachedGraphDataset(Dataset):
    def __init__(self, data_dir, model_config=None):
        self.files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.pt')]
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
    model_cfg = _cfg_get(config_dict, "model", {})
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            images = _apply_channel_ablation(images, model_cfg)
            y = batch["y"].to(device, non_blocking=True)
            valid_token_mask = batch.get("valid_token_mask")
            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.to(device, non_blocking=True)
                
            out = model(images, valid_token_mask=valid_token_mask)
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
        metrics = compute_multitask_metrics(preds, targets)
    else:
        metrics = compute_regression_metrics(preds, targets)
        
    metrics["loss"] = epoch_loss
    return metrics

def _cfg_get(cfg, key, default=None):
    """Small helper supporting dict/OmegaConf-style configs."""
    try:
        return cfg.get(key, default)
    except Exception:
        try:
            return getattr(cfg, key)
        except Exception:
            return default


def _apply_channel_ablation(images, model_cfg):
    """
    Apply channel ablation while preserving [B, 3, H, W].

    Convention:
      channel 0 = topology / APSP heat kernel
      channel 1 = node-feature covariance
      channel 2 = edge-feature covariance

    Config options:
      model.channel_indices: e.g. [0], [0, 1], [0, 1, 2]
      model.channel_mask:    e.g. [1, 0, 0], [1, 1, 0], [1, 1, 1]
    """
    channel_indices = _cfg_get(model_cfg, "channel_indices", None)
    channel_mask = _cfg_get(model_cfg, "channel_mask", None)

    if channel_indices is None and channel_mask is None:
        return images

    c = images.shape[1]

    if channel_indices is not None:
        keep = [int(i) for i in list(channel_indices)]
        mask = images.new_zeros(c)
        for i in keep:
            if i < 0 or i >= c:
                raise ValueError(f"Invalid channel index {i} for image with {c} channels")
            mask[i] = 1.0
    else:
        vals = [float(x) for x in list(channel_mask)]
        if len(vals) != c:
            raise ValueError(f"channel_mask length {len(vals)} does not match image channels {c}")
        mask = images.new_tensor(vals)

    return images * mask.view(1, c, 1, 1)


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory. Defaults to outputs/run_<hash> if not set.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--image-config", type=str, default=None, help="Override image preprocessing config")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases")
    parser.add_argument("--wandb-project", default="vino_bbbp")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="frozen_dino_full_ablation")
    parser.add_argument("--output-root", default=None, help="Create a unique W&B run directory below this root")
    args = parser.parse_args()
    
    from vino.utils.config import load_resolved_config
    config = load_resolved_config(args.config)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            job_type="frozen_dino_ablation", config={"base_config": args.config},
        )
        apply_frozen_sweep_overrides(config, dict(wandb.config))
    if args.image_config:
        config.image = OmegaConf.load(args.image_config)
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    required = {
        "model": ["backbone_name", "freeze_mode", "normalize_lvd_imagenet", "input_adapter", "head"],
        "train": ["batch_size", "epochs", "lr_head", "lr_backbone", "weight_decay", "grad_clip", "task_type"],
    }
    missing = [f"{section}.{key}" for section, keys in required.items() for key in keys if key not in config[section]]
    if missing:
        raise ValueError("Incomplete experiment config; missing: " + ", ".join(missing))
    
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
        conf_hash = hash_preprocessing_config(config_dict)
        data_dir = os.path.join("data/processed", dataset_name, conf_hash)
        
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"Data not found at {train_dir}. Please run build_graph_images.py first.")
        raise FileNotFoundError(f"Data not found at {train_dir}. Please run build_graph_images.py first.")
        
    model_config = config_dict.get("model", {})
    
    train_dataset = CachedGraphDataset(train_dir, model_config=model_config)
    if not train_dataset:
        raise ValueError(f"Training cache is empty: {train_dir}")
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {"batch_size": config.train.batch_size, "collate_fn": collate_fn,
                     "num_workers": int(config.train.get("num_workers", 0)), "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    
    val_dir = os.path.join(data_dir, "val")
    val_loader = None
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        val_loader = DataLoader(CachedGraphDataset(val_dir, model_config=model_config), shuffle=False, **loader_kwargs)
        
    test_dir = os.path.join(data_dir, "test")
    test_loader = None
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_loader = DataLoader(CachedGraphDataset(test_dir, model_config=model_config), shuffle=False, **loader_kwargs)
    
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

    if args.output_dir is not None and args.output_root is not None:
        raise ValueError("Use only one of --output_dir and --output-root")
    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.output_root is not None:
        if wandb_run is None:
            raise ValueError("--output-root requires --wandb for a unique run ID")
        out_dir = os.path.join(args.output_root, f"{wandb_run.id}_{hash_config(config_dict)}")
    else:
        out_dir = os.path.join("outputs", "run_" + hash_config(config_dict))
    if os.path.exists(out_dir):
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[device] cuda name: {torch.cuda.get_device_name(0)}", flush=True)

    model = GraphImageModel(config_dict)
    model = model.to(device)
    os.makedirs(out_dir, exist_ok=False)
    print("[debug] model device:", next(model.parameters()).device, flush=True)
    if wandb_run is not None:
        wandb_run.name = "__".join(str(wandb_run.config.get(k, "na")) for k in (
            "channel_set", "resize_mode", "stem_variant", "head_variant", "seed"
        ))
        wandb_run.config.update(config_dict, allow_val_change=True)
    
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
    
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        OmegaConf.save(config, f)
        
    metrics_file = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    
    best_train_loss = float("inf")
    best_val_metric_val = None
    best_epoch = 0
    metric_for_best = str(config.train.get("metric_for_best", "val_loss" if val_loader else "train_loss"))
    metric_mode = str(config.train.get("metric_mode", "min"))
    if metric_mode not in {"min", "max"}:
        raise ValueError(f"metric_mode must be 'min' or 'max', got {metric_mode!r}")
    
    # Pre-save a best checkpoint in case epochs=0
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_best.pt"))
    
    printed_first_batch_debug = False
    
    for epoch in range(epochs):
        model.train()
        if model_cfg.get("freeze_mode") == "frozen":
            model.backbone.eval()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            images = _apply_channel_ablation(images, model_cfg)
            y = batch["y"].to(device, non_blocking=True)
            valid_token_mask = batch.get("valid_token_mask")
            if valid_token_mask is not None:
                valid_token_mask = valid_token_mask.to(device, non_blocking=True)
                
            optimizer.zero_grad()
            out = model(images, valid_token_mask=valid_token_mask)
            
            if not printed_first_batch_debug:
                printed_first_batch_debug = True
                print("[debug] first batch image:", images.device, tuple(images.shape), flush=True)
                print("[debug] first batch y:", y.device, tuple(y.shape), flush=True)
                print("[debug] first logits:", out.device, tuple(out.shape), flush=True)
                if torch.cuda.is_available() and model_name_val != "tiny_dummy":
                    assert next(model.parameters()).is_cuda, "Non-tiny model is not on CUDA"
                    assert images.is_cuda, "Input images are not on CUDA"
                    assert out.is_cuda, "Logits are not on CUDA"
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "examples/model_input": wandb.Image(
                            images[0].detach().cpu(), caption=f"channels={wandb_run.config.get('channel_set')}"
                        )
                    }, step=epoch)

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
            log_entry.update({f"val_{key}": value for key, value in val_metrics.items()})
            metric_key = metric_for_best.removeprefix("val_")
            if metric_key not in val_metrics:
                raise ValueError(f"Configured metric_for_best {metric_for_best!r} was not computed")
            current_metric = val_metrics[metric_key]
            if current_metric is None:
                current_metric = val_metrics["loss"]
                
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
        if wandb_run is not None:
            wandb_run.log(log_entry, step=epoch)
        
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
        "input_stem_type": (
            model_cfg.get("input_stem", {}).get("type", "identity")
            if bool(model_cfg.get("input_stem", {}).get("enabled", False))
            else "identity"
        ),
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
        "timestamp": datetime.datetime.now().isoformat(),
        "config_hash": hash_config(config_dict),
        "preprocessing_hash": hash_preprocessing_config(config_dict),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None,
    }
    validate_result(result)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    if wandb_run is not None:
        wandb_run.summary.update(result)
        wandb_run.summary["selection_metric"] = result.get("val_roc_auc")
        wandb_run.log({
            "final/val_roc_auc": result.get("val_roc_auc"),
            "final/test_roc_auc": result.get("test_roc_auc"),
            "final/best_epoch": result.get("best_epoch"),
        })
        wandb_run.finish()
        
if __name__ == "__main__":
    main()
