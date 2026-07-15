import argparse
import os
import platform
import subprocess
import json
import torch
from omegaconf import OmegaConf
from datetime import datetime
from vino.utils.hashing import hash_preprocessing_config
from vino.data.synthetic import generate_synthetic_graphs
from vino.transforms.graph_image import make_graph_image
from tqdm import tqdm
from vino.utils.io import validate_cached_graph

def main():
    parser = argparse.ArgumentParser(description="Build graph images")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    from vino.utils.config import load_resolved_config
    config = load_resolved_config(args.config)
    
    if "dataset" not in config:
        config = OmegaConf.create({"dataset": config})
    # Also load image-config if provided
    if args.image_config:
        img_conf = OmegaConf.load(args.image_config)
        # Merge img_conf into config under "image_config" or similar
        config = OmegaConf.merge(config, {"image_pipeline": img_conf})
        # Hack to map to the structure we used in code
        config = OmegaConf.merge(config, img_conf)
        
    config_dict = OmegaConf.to_container(config, resolve=True)
    conf_hash = hash_preprocessing_config(config_dict)
    
    dataset_cfg = config_dict.get("dataset", {})
    dataset_name = dataset_cfg.get("dataset", "unknown") if isinstance(dataset_cfg, dict) else dataset_cfg
        
    out_dir = os.path.join(args.output_dir, dataset_name, conf_hash)
    if args.overwrite:
        raise ValueError("--overwrite is disabled: cached datasets are immutable")
    if os.path.exists(out_dir):
        raise FileExistsError(f"Cache already exists: {out_dir}")
    os.makedirs(out_dir, exist_ok=False)
    
    if dataset_name == "synthetic":
        dconf = config_dict["dataset"] if "dataset" in config_dict else config_dict
        records = generate_synthetic_graphs(
            dconf.get("num_graphs", 100),
            dconf.get("min_nodes", 6),
            dconf.get("max_nodes", 20),
            dconf.get("node_dim", 10),
            dconf.get("edge_dim", 3),
            dconf.get("task_type", "binary_classification"),
            dconf.get("seed", 42)
        )
    elif dataset_name == "bbbp":
        from vino.data.pyg_loaders import load_bbbp
        dconf = config_dict["dataset"] if "dataset" in config_dict else config_dict
        limit = dconf.get("limit")
        records = load_bbbp(limit=limit, split_seed=int(dconf.get("split_seed", 42)))
    elif dataset_name == "molhiv":
        from vino.data.pyg_loaders import load_molhiv
        dconf = config_dict["dataset"]
        records = load_molhiv(root=dconf.get("root", "data/ogb"))
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    splits = {"train": 0, "val": 0, "test": 0}
    num_total = len(records)
    num_success = 0
    num_failed = 0
    first_failure = None
    
    for r in tqdm(records, desc="Processing graphs"):
        split_dir = os.path.join(out_dir, r.split)
        os.makedirs(split_dir, exist_ok=True)
        
        try:
            img_data = make_graph_image(r, config_dict)
            img_data["split"] = r.split
            img_data["source_metadata"] = r.metadata
            validate_cached_graph(img_data)
            torch.save(img_data, os.path.join(split_dir, f"{r.graph_id}.pt"))
            splits[r.split] += 1
            num_success += 1
        except Exception as e:
            num_failed += 1
            if first_failure is None:
                first_failure = str(e)
            with open(os.path.join(out_dir, "failures.jsonl"), "a") as f:
                f.write(json.dumps({"graph_id": r.graph_id, "error": str(e)}) + "\n")
                
    if num_success == 0 and num_total > 0:
        raise RuntimeError(f"All graph conversions failed for dataset {dataset_name} at {out_dir}. First failure: {first_failure}")
                
    manifest = {
        "dataset": dataset_name,
        "splits": splits,
        "num_total": num_total,
        "num_success": num_success,
        "num_failed": num_failed,
        "output_dir": out_dir,
        "config_hash": conf_hash,
        "split_strategy": config_dict.get("dataset", {}).get("split_strategy", "ogb_official" if dataset_name == "molhiv" else "generated"),
        "split_seed": config_dict.get("dataset", {}).get("split_seed"),
        "split_graph_ids": {s: sorted(r.graph_id for r in records if r.split == s) for s in splits},
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        OmegaConf.save(config, f)
        
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
