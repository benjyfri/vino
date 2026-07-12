#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def stratified_split(items, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for item in items:
        by_label[item["label"]].append(item)

    splits = {"train": [], "val": [], "test": []}

    for label, group in sorted(by_label.items()):
        rng.shuffle(group)

        n = len(group)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))

        # Make sure all splits get examples when possible.
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))

        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]

        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)

    for split in splits:
        rng.shuffle(splits[split])

    return splits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)

    if args.dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.dst} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.dst)

    files = sorted(args.src.glob("**/*.pt"))
    if not files:
        raise RuntimeError(f"No .pt files found under {args.src}")

    items = []
    for path in files:
        obj = torch.load(path, map_location="cpu")
        y = float(obj["y"].reshape(-1)[0])
        if y not in (0.0, 1.0):
            raise ValueError(f"Expected binary label 0/1, got {y} in {path}")

        graph_id = str(obj.get("graph_id", path.stem))
        items.append({
            "path": path,
            "label": int(y),
            "graph_id": graph_id,
        })

    splits = stratified_split(
        items,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    for split, split_items in splits.items():
        for item in split_items:
            src = item["path"]
            graph_id = item["graph_id"]
            dst = args.dst / split / f"{graph_id}.pt"
            link_or_copy(src, dst, args.mode)

    manifest = {
        "dataset": "bbbp",
        "source_cache": str(args.src),
        "output_dir": str(args.dst),
        "split_strategy": "stratified_random",
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "mode": args.mode,
        "splits": {k: len(v) for k, v in splits.items()},
        "label_counts": {
            k: dict(Counter(item["label"] for item in v))
            for k, v in splits.items()
        },
        "num_total": len(items),
        "num_success": len(items),
        "num_failed": 0,
        "timestamp": datetime.now().isoformat(),
    }

    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
