#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vino.transforms.model_input import transform_graph_image_for_model


CHANNEL_NAMES = ["Topology", "Node covariance", "Edge covariance"]


def load_sample(path: Path) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    obj = torch.load(path, map_location="cpu")
    required = {"graph_id", "image", "valid_node_mask", "valid_pixel_mask", "y", "metadata"}
    missing = required - set(obj.keys())
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
    return obj


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def rgb_np(x: torch.Tensor) -> np.ndarray:
    x = clamp01(x)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(x.shape)}")
    return x.permute(1, 2, 0).cpu().numpy()


def make_resize(image: torch.Tensor, n: int, size: int, mode: str) -> torch.Tensor:
    crop = image[:, :n, :n]
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(
        crop.unsqueeze(0),
        size=(size, size),
        mode=mode,
        **kwargs,
    ).squeeze(0).clamp(0, 1)


def draw_patch_grid(ax, image_size: int, patch_size: int, n_valid: int | None = None) -> None:
    # Full DINO patch grid.
    for v in range(0, image_size + 1, patch_size):
        ax.axhline(v - 0.5, linewidth=0.15, alpha=0.25)
        ax.axvline(v - 0.5, linewidth=0.15, alpha=0.25)

    # Boundary between valid graph tokens and padding.
    if n_valid is not None:
        b = n_valid * patch_size - 0.5
        ax.axhline(b, linewidth=1.0, alpha=0.9)
        ax.axvline(b, linewidth=1.0, alpha=0.9)


def save_comparison_sheet(
    obj: dict,
    views: list[tuple[str, torch.Tensor, dict]],
    out_path: Path,
) -> None:
    graph_id = obj["graph_id"]
    n_rows = len(views)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 3.55 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (name, img, meta) in enumerate(views):
        img = clamp01(img)
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(img[c].cpu().numpy(), vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_title(f"{name}\n{CHANNEL_NAMES[c]} | {tuple(img.shape)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if meta.get("draw_grid", False):
                draw_patch_grid(
                    ax,
                    image_size=img.shape[-1],
                    patch_size=meta["patch_size"],
                    n_valid=meta.get("n_valid"),
                )

        ax = axes[r, 3]
        ax.imshow(rgb_np(img), vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{name}\nRGB: R=topology, G=node, B=edge", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        if meta.get("draw_grid", False):
            draw_patch_grid(
                ax,
                image_size=img.shape[-1],
                patch_size=meta["patch_size"],
                n_valid=meta.get("n_valid"),
            )

    fig.suptitle(
        f"DINO input-view comparison | {graph_id} | y={obj['y'].tolist()}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_individual_rgb(
    graph_id: str,
    views: list[tuple[str, torch.Tensor, dict]],
    out_dir: Path,
) -> None:
    for name, img, meta in views:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(rgb_np(img), vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{graph_id} | {name} | {tuple(img.shape)}")
        ax.set_xticks([])
        ax.set_yticks([])

        if meta.get("draw_grid", False):
            draw_patch_grid(
                ax,
                image_size=img.shape[-1],
                patch_size=meta["patch_size"],
                n_valid=meta.get("n_valid"),
            )

        fig.savefig(out_dir / f"{graph_id}_{name}_rgb.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--token_grid_size", type=int, default=32)
    args = parser.parse_args()

    obj = load_sample(args.input)

    graph_id = obj["graph_id"]
    image = clamp01(obj["image"])
    valid_node_mask = obj["valid_node_mask"].bool()
    n = int(valid_node_mask.sum().item())

    valid_crop = image[:, :n, :n]
    naive_padded = image

    resize_bilinear = transform_graph_image_for_model(
        image=image,
        valid_node_mask=valid_node_mask,
        input_mode="resize_bilinear",
        input_size=args.resize_size,
        patch_size=args.patch_size,
        token_grid_size=args.token_grid_size,
        resize_mode="bilinear",
    )["image"]

    resize_nearest = make_resize(image, n, args.resize_size, "nearest")

    try:
        patch_out = transform_graph_image_for_model(
            image=image,
            valid_node_mask=valid_node_mask,
            input_mode="patch_aligned_repeat",
            input_size=args.resize_size,
            patch_size=args.patch_size,
            token_grid_size=args.token_grid_size,
            resize_mode="bilinear",
        )
        patch_aligned = patch_out["image"]
        valid_token_mask = patch_out["valid_token_mask"]
        patch_error = None
    except ValueError as e:
        patch_aligned = torch.zeros((3, args.token_grid_size * args.patch_size, args.token_grid_size * args.patch_size), dtype=image.dtype)
        valid_token_mask = torch.zeros((args.token_grid_size, args.token_grid_size), dtype=torch.bool)
        patch_error = str(e)

    views = [
        (
            "01_valid_crop_raw",
            valid_crop,
            {"kind": "raw"},
        ),
        (
            "02_naive_padded_256",
            naive_padded,
            {"kind": "bad_baseline"},
        ),
        (
            f"03_resize_bilinear_{args.resize_size}",
            resize_bilinear,
            {"kind": "dino_resize"},
        ),
        (
            f"04_resize_nearest_{args.resize_size}",
            resize_nearest,
            {"kind": "dino_resize_discrete"},
        ),
        (
            f"05_patch_aligned_repeat_{args.token_grid_size * args.patch_size}" if not patch_error else f"patch_aligned_repeat invalid\nn_valid={n} > token_grid_size={args.token_grid_size}",
            patch_aligned,
            {
                "kind": "dino_patch_aligned",
                "draw_grid": patch_error is None,
                "patch_size": args.patch_size,
                "n_valid": n,
            },
        ),
    ]

    out_dir = args.out_dir / graph_id
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / f"{graph_id}_dino_input_comparison_channels.png"
    save_comparison_sheet(obj, views, comparison_path)
    save_individual_rgb(graph_id, views, out_dir)

    stats = {
        "input": str(args.input),
        "graph_id": graph_id,
        "n_valid": n,
        "raw_cache_shape": list(image.shape),
        "valid_crop_shape": list(valid_crop.shape),
        "resize_bilinear_shape": list(resize_bilinear.shape),
        "resize_nearest_shape": list(resize_nearest.shape),
        "patch_aligned_shape": list(patch_aligned.shape),
        "patch_size": args.patch_size,
        "token_grid_size": args.token_grid_size,
        "valid_token_mask_shape": list(valid_token_mask.shape),
        "valid_token_count": int(valid_token_mask.sum().item()),
        "expected_valid_token_count": n * n if patch_error is None else 0,
        "patch_aligned_valid": patch_error is None,
        "patch_aligned_error": patch_error,
        "comparison_path": str(comparison_path),
    }

    stats_path = out_dir / f"{graph_id}_dino_input_comparison_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
