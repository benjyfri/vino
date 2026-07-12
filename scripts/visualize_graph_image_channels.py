#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from vino.transforms.model_input import transform_graph_image_for_model

CHANNEL_NAMES = [
    "Topology / APSP heat",
    "Node covariance",
    "Edge covariance",
]


def load_graph_image(path: Path) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected cached .pt file to contain a dict, got {type(obj)}")
    required = {"graph_id", "image", "valid_node_mask", "valid_pixel_mask", "y", "metadata"}
    missing = required - set(obj.keys())
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")
    return obj


def to_numpy_image(obj: dict, crop: bool = True) -> tuple[np.ndarray, int]:
    image = obj["image"].detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got shape {tuple(image.shape)}")

    valid_node_mask = obj["valid_node_mask"].detach().cpu().bool()
    n_valid = int(valid_node_mask.sum().item())

    if crop:
        image = image[:, :n_valid, :n_valid]

    image_np = image.numpy()
    image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np, n_valid


def save_contact_sheet(obj: dict, image: np.ndarray, n_valid: int, out_path: Path) -> None:
    graph_id = obj["graph_id"]
    y = obj["y"]
    metadata = obj.get("metadata", {})

    rgb = np.moveaxis(image, 0, -1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

    for c in range(3):
        axes[c].imshow(image[c], vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[c].set_title(CHANNEL_NAMES[c], fontsize=10)
        axes[c].set_xticks([])
        axes[c].set_yticks([])

    axes[3].imshow(rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[3].set_title("RGB composite\nR=topology, G=node, B=edge", fontsize=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    fig.suptitle(
        f"{graph_id} | valid nodes={n_valid} | y={y.tolist()} | "
        f"eigengap={metadata.get('fiedler_eigengap', 'NA')}",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_individual_channels(obj: dict, image: np.ndarray, out_dir: Path) -> None:
    graph_id = obj["graph_id"]
    rgb = np.moveaxis(image, 0, -1)

    for c, name in enumerate(["topology", "node_cov", "edge_cov"]):
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(image[c], vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{graph_id} — {name}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(out_dir / f"{graph_id}_{name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"{graph_id} — RGB composite")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_dir / f"{graph_id}_rgb.png", dpi=180)
    plt.close(fig)


def save_model_view(obj: dict, transformed: dict, out_dir: Path, args) -> dict:
    graph_id = obj["graph_id"]
    image = transformed["image"].detach().cpu().numpy()
    rgb = np.moveaxis(image, 0, -1)
    
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"{graph_id} — {args.input_mode} model view\nSize: {image.shape}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    out_path = out_dir / f"{graph_id}_model_view_{args.input_mode}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    
    if args.input_mode == "patch_aligned_repeat":
        # Draw grid overlay
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(rgb, vmin=0.0, vmax=1.0, interpolation="nearest")
        
        # Draw grid
        h, w = rgb.shape[:2]
        ps = args.patch_size
        for x in range(0, w+1, ps):
            ax.axvline(x - 0.5, color='white', lw=0.5, alpha=0.5)
        for y in range(0, h+1, ps):
            ax.axhline(y - 0.5, color='white', lw=0.5, alpha=0.5)
            
        ax.set_title(f"{graph_id} — {args.input_mode} model view (grid)")
        ax.set_xticks([])
        ax.set_yticks([])
        grid_path = out_dir / f"{graph_id}_model_view_patch_grid.png"
        fig.savefig(grid_path, dpi=180)
        plt.close(fig)
        
    n_valid = transformed["n_valid"]
    summary = {
        "raw_valid_crop_shape": [3, n_valid, n_valid],
        "transformed_image_shape": list(image.shape),
        "input_mode": args.input_mode,
        "n_valid": n_valid
    }
    
    if transformed.get("valid_token_mask") is not None:
        summary["valid_token_mask_shape"] = list(transformed["valid_token_mask"].shape)
        if args.input_mode == "patch_aligned_repeat":
            summary["number_of_valid_tokens"] = n_valid * n_valid
            
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to cached graph .pt file")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--full", action="store_true", help="Show full padded 256x256 image instead of valid-node crop")
    
    parser.add_argument("--model_view", action="store_true")
    parser.add_argument("--input_mode", type=str, default="resize_bilinear")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--token_grid_size", type=int, default=32)
    parser.add_argument("--resize_mode", type=str, default="bilinear")
    args = parser.parse_args()

    obj = load_graph_image(args.input)
    image, n_valid = to_numpy_image(obj, crop=not args.full)

    graph_id = obj["graph_id"]
    out_dir = args.out_dir / graph_id
    out_dir.mkdir(parents=True, exist_ok=True)

    contact_path = out_dir / f"{graph_id}_channels_contact_sheet.png"
    save_contact_sheet(obj, image, n_valid, contact_path)
    save_individual_channels(obj, image, out_dir)

    summary = {
        "input": str(args.input),
        "graph_id": graph_id,
        "output_dir": str(out_dir),
        "contact_sheet": str(contact_path),
        "image_shape_visualized": list(image.shape),
        "valid_nodes": n_valid,
        "y": obj["y"].tolist(),
        "metadata": obj.get("metadata", {}),
    }
    
    if args.model_view:
        # Generate mask
        mask = torch.zeros(obj["image"].size(1), dtype=torch.bool)
        num_nodes = int(obj["valid_node_mask"].sum().item())
        mask[:num_nodes] = True
        
        transformed = transform_graph_image_for_model(
            image=obj["image"].float(),
            valid_node_mask=mask,
            input_mode=args.input_mode,
            input_size=args.input_size,
            patch_size=args.patch_size,
            token_grid_size=args.token_grid_size,
            resize_mode=args.resize_mode
        )
        model_summary = save_model_view(obj, transformed, out_dir, args)
        summary.update(model_summary)
        
        print(f"Raw valid crop shape: {summary['raw_valid_crop_shape']}")
        print(f"Transformed image shape: {summary['transformed_image_shape']}")
        print(f"Input mode: {summary['input_mode']}")
        print(f"n_valid: {summary['n_valid']}")
        if "valid_token_mask_shape" in summary:
            print(f"valid_token_mask shape: {summary['valid_token_mask_shape']}")
        if "number_of_valid_tokens" in summary:
            print(f"Number of valid tokens: {summary['number_of_valid_tokens']}")

    summary_path = out_dir / f"{graph_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if not args.model_view:
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
