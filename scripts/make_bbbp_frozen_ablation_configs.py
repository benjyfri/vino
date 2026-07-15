#!/usr/bin/env python
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
from vino.utils.config import load_resolved_config


BASE_CONFIG = "configs/experiments/bbbp_full_resize_vits16_no_stem.yaml"
OUT_DIR = Path("configs/experiments/generated_ablation_configs")
MANIFEST = OUT_DIR / "bbbp_frozen_ablation_manifest.csv"


def to_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def write_config(name: str, cfg: dict) -> str:
    path = OUT_DIR / f"{name}.yaml"
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(path))
    return str(path)


def set_common_baseline(cfg: dict):
    model = cfg.setdefault("model", {})
    train = cfg.setdefault("train", {})

    # Core production baseline after interface ablation.
    model["model_name"] = "facebook/dinov3-vits16-pretrain-lvd1689m"
    model["backbone_type"] = "hf_auto"
    model["freeze_mode"] = "frozen"
    model["input_stem"] = {"enabled": False}
    model["input_mode"] = "resize_bilinear"
    model["input_size"] = 224
    model["patch_size"] = 16
    model["token_grid_size"] = 32
    model["resize_mode"] = "bilinear"

    # Conservative default. Best epochs were usually < 50.
    train["epochs"] = 50


def set_channel_ablation(cfg: dict, channels: list[int], name: str):
    """
    This assumes train/model_input code supports config.model.channel_indices
    or config.model.channel_mask.

    If not yet implemented, the job will still run but the channel ablation
    will not be real. See the preflight check below.
    """
    model = cfg.setdefault("model", {})
    model["channel_ablation_name"] = name
    model["channel_indices"] = channels
    mask = [0, 0, 0]
    for c in channels:
        mask[c] = 1
    model["channel_mask"] = mask


def set_head_cfg(cfg: dict, head_name: str, lr=None, weight_decay=None, dropout=None, hidden_dim=None):
    train = cfg.setdefault("train", {})
    model = cfg.setdefault("model", {})

    model["head_ablation_name"] = head_name

    if lr is not None:
        train["lr_head"] = lr
    if weight_decay is not None:
        train["weight_decay"] = weight_decay
    if dropout is not None:
        model["dropout"] = dropout
        model.setdefault("head", {})["dropout"] = dropout
        model["head_dropout"] = dropout
    if hidden_dim is not None:
        model.setdefault("head", {})["hidden_dim"] = hidden_dim
        model["head_hidden_dim"] = hidden_dim
        model["adapter_hidden_dim"] = hidden_dim


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = to_dict(load_resolved_config(BASE_CONFIG))
    assert "train" in base and "batch_size" in base["train"], base.get("train")
    assert "model" in base, base.keys()

    rows = []

    seeds = [42, 2025, 9001]

    # ------------------------------------------------------------------
    # Phase A: channel ablations
    # Channel convention:
    #   0 = topology / APSP heat kernel
    #   1 = node covariance
    #   2 = edge covariance
    # ------------------------------------------------------------------
    channel_variants = [
        ("ch_topology_only", [0]),
        ("ch_node_only", [1]),
        ("ch_edge_only", [2]),
        ("ch_topology_node", [0, 1]),
        ("ch_topology_edge", [0, 2]),
        ("ch_node_edge", [1, 2]),
        ("ch_all_three", [0, 1, 2]),
    ]

    for variant_name, channels in channel_variants:
        cfg = deepcopy(base)
        set_common_baseline(cfg)
        set_channel_ablation(cfg, channels, variant_name)

        path = write_config(f"bbbp_frozen_{variant_name}", cfg)

        for seed in seeds:
            rows.append({
                "phase": "channel",
                "variant": variant_name,
                "seed": seed,
                "config": path,
            })

    # ------------------------------------------------------------------
    # Phase B: resize/interpolation ablations
    # ------------------------------------------------------------------
    resize_variants = [
        ("resize_bilinear", "bilinear"),
        ("resize_nearest", "nearest"),
        ("resize_area", "area"),
    ]

    for variant_name, resize_mode in resize_variants:
        cfg = deepcopy(base)
        set_common_baseline(cfg)
        cfg["model"]["resize_mode"] = resize_mode
        cfg["model"]["resize_ablation_name"] = variant_name
        cfg["model"]["channel_indices"] = [0, 1, 2]
        cfg["model"]["channel_mask"] = [1, 1, 1]

        path = write_config(f"bbbp_frozen_{variant_name}", cfg)

        for seed in seeds:
            rows.append({
                "phase": "resize",
                "variant": variant_name,
                "seed": seed,
                "config": path,
            })

    # ------------------------------------------------------------------
    # Phase C: small training/head regularization sweep
    # Keep compact. Do one seed for all configs, then selected configs
    # can later be repeated with more seeds.
    # ------------------------------------------------------------------
    head_variants = [
        ("head_current_lr1e-3_wd1e-3_do0p1", 1e-3, 1e-3, 0.1, None),
        ("head_lr3e-4_wd1e-3_do0p1", 3e-4, 1e-3, 0.1, None),
        ("head_lr1e-4_wd1e-3_do0p1", 1e-4, 1e-3, 0.1, None),
        ("head_lr3e-4_wd1e-2_do0p1", 3e-4, 1e-2, 0.1, None),
        ("head_lr3e-4_wd1e-3_do0p3", 3e-4, 1e-3, 0.3, None),
        ("head_small64_lr3e-4_wd1e-2_do0p3", 3e-4, 1e-2, 0.3, 64),
        ("head_small128_lr3e-4_wd1e-2_do0p3", 3e-4, 1e-2, 0.3, 128),
    ]

    # Use seed 42 first for training sweep to avoid exploding job count.
    # If a setting wins, rerun with 3-5 seeds later.
    for variant_name, lr, wd, dropout, hidden_dim in head_variants:
        cfg = deepcopy(base)
        set_common_baseline(cfg)
        cfg["model"]["channel_indices"] = [0, 1, 2]
        cfg["model"]["channel_mask"] = [1, 1, 1]
        set_head_cfg(cfg, variant_name, lr=lr, weight_decay=wd, dropout=dropout, hidden_dim=hidden_dim)

        path = write_config(f"bbbp_frozen_{variant_name}", cfg)

        rows.append({
            "phase": "head_reg",
            "variant": variant_name,
            "seed": 42,
            "config": path,
        })

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "variant", "seed", "config"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {MANIFEST}")
    print(f"Num runs: {len(rows)}")
    print()
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
