from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EarlyStopping:
    patience: int
    mode: str
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        improved = self.best is None or (
            value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        )
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop


def resolve_selection_value(metrics: dict, metric_for_best: str, metric_mode: str) -> float:
    key = metric_for_best.removeprefix("val_")
    if key not in metrics:
        raise ValueError(f"Configured selection metric {metric_for_best!r} was not computed")
    value = metrics[key]
    if value is None:
        raise ValueError(
            f"Selection metric {metric_for_best!r} is undefined for this validation split; "
            "choose a defined metric instead of silently changing metric direction"
        )
    if metric_mode not in {"min", "max"}:
        raise ValueError("metric_mode must be min or max")
    return float(value)


def infer_binary_pos_weight(dataset, requested):
    if requested is None:
        return None
    if requested != "auto":
        return float(requested)
    positives = negatives = 0
    for index in range(len(dataset)):
        target = (dataset.target(index) if hasattr(dataset, "target") else dataset[index]["y"]).reshape(-1)
        if target.numel() != 1 or target.isnan().any():
            raise ValueError("Automatic pos_weight requires scalar, non-missing binary targets")
        label = float(target.item())
        if label == 1.0:
            positives += 1
        elif label == 0.0:
            negatives += 1
        else:
            raise ValueError(f"Automatic pos_weight requires labels 0/1, got {label}")
    if positives == 0:
        raise ValueError("Cannot infer pos_weight without positive training examples")
    return negatives / positives


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")
