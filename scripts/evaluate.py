import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scripts.train import CachedGraphDataset, collate_fn, evaluate
from vino.models.graph_image_model import GraphImageModel
from vino.utils.config import load_resolved_config
from vino.utils.hashing import hash_preprocessing_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", help="Optional new JSON output path")
    args = parser.parse_args()

    cfg = load_resolved_config(args.config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    dataset_name = resolved["dataset"]["dataset"]
    data_dir = Path(args.data_dir or Path("data/processed") / dataset_name / hash_preprocessing_config(resolved))
    split_dir = data_dir / args.split
    dataset = CachedGraphDataset(split_dir, resolved["model"])
    if not dataset:
        raise ValueError(f"No cached samples in {split_dir}")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphImageModel(resolved).to(device)
    # Checkpoints store only trained parameters (frozen backbone stays pretrained from
    # construction), so load non-strictly and keep the pretrained weights for missing keys.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=False)
    result = {"split": args.split, "num_samples": len(dataset), **evaluate(model, loader, resolved)}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
