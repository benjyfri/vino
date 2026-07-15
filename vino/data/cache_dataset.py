from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vino.transforms.model_input import transform_graph_image_for_model
from vino.utils.io import validate_cached_graph


class CachedGraphDataset(Dataset):
    """Validated, immutable graph-image cache reader."""

    def __init__(self, data_dir: str | Path, model_config: dict | None = None):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        self.model_config = model_config or {}

    def __len__(self) -> int:
        return len(self.files)

    def target(self, idx: int) -> torch.Tensor:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "y" not in data:
            raise ValueError(f"Cache entry has no target: {self.files[idx]}")
        return data["y"]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Cache entry must be a dict: {self.files[idx]}")
        validate_cached_graph(data)
        input_mode = self.model_config.get("input_mode")
        if input_mode:
            transformed = transform_graph_image_for_model(
                image=data["image"], valid_node_mask=data["valid_node_mask"], input_mode=input_mode,
                input_size=self.model_config.get("input_size", 224),
                patch_size=self.model_config.get("patch_size", 16),
                token_grid_size=self.model_config.get("token_grid_size"),
                resize_mode=self.model_config.get("resize_mode", "bilinear"),
            )
            data = {**data, **{key: value for key, value in transformed.items() if value is not None}}
        return data


def collate_cached_graphs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    shapes = [item["image"].shape[-2:] for item in batch]
    max_h, max_w = max(h for h, _ in shapes), max(w for _, w in shapes)
    images = []
    for item in batch:
        image = item["image"]
        if image.shape[-2:] != (max_h, max_w):
            padded = image.new_zeros((image.shape[0], max_h, max_w))
            padded[:, : image.shape[-2], : image.shape[-1]] = image
            image = padded
        images.append(image)
    result = {
        "image": torch.stack(images),
        "y": torch.stack([item["y"] for item in batch]).view(len(batch), -1),
        "graph_id": [item["graph_id"] for item in batch],
    }
    optional = ("valid_token_mask", "n_valid", "input_mode", "model_input_size")
    for key in optional:
        present = [key in item for item in batch]
        if any(present) and not all(present):
            raise ValueError(f"Mixed presence of optional batch field {key!r}")
        if all(present):
            if key == "valid_token_mask":
                result[key] = torch.stack([item[key] for item in batch])
            elif key == "n_valid":
                result[key] = [item[key] for item in batch]
            else:
                values = {item[key] for item in batch}
                if len(values) != 1:
                    raise ValueError(f"Inconsistent {key!r} values in batch: {values}")
                result[key] = next(iter(values))
    return result
