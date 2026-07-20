import argparse
import os
import platform
import subprocess
import json
import hashlib
import uuid
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
        
    final_out_dir = os.path.join(args.output_dir, dataset_name, conf_hash)
    if args.overwrite:
        raise ValueError("--overwrite is disabled: cached datasets are immutable")
    if os.path.exists(final_out_dir):
        raise FileExistsError(f"Cache already exists: {final_out_dir}")
    os.makedirs(os.path.dirname(final_out_dir), exist_ok=True)
    out_dir = f"{final_out_dir}.staging-{uuid.uuid4().hex}"
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
        records = load_bbbp(
            limit=limit, split_seed=int(dconf.get("split_seed", 42)),
            split_strategy=dconf.get("split_strategy", "scaffold"), root=dconf.get("root", "data/pyg"),
        )
    elif dataset_name == "molhiv":
        from vino.data.pyg_loaders import load_molhiv
        dconf = config_dict["dataset"]
        records = load_molhiv(root=dconf.get("root", "data/ogb"))
    elif dataset_name == "esol":
        from vino.data.pyg_loaders import load_esol
        dconf = config_dict["dataset"]
        records = load_esol(root=dconf.get("root", "data/pyg"), split_seed=int(dconf.get("split_seed", 42)))
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    splits = {"train": 0, "val": 0, "test": 0}
    num_total = len(records)
    num_success = 0
    num_failed = 0
    first_failure = None
    saved_graph_ids = {"train": [], "val": [], "test": []}
    sign_methods = []

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
            saved_graph_ids[r.split].append(r.graph_id)
            sign_methods.append(img_data.get("metadata", {}).get("sign_method", "unknown"))
            num_success += 1
        except Exception as e:
            num_failed += 1
            if first_failure is None:
                first_failure = str(e)
            with open(os.path.join(out_dir, "failures.jsonl"), "a") as f:
                f.write(json.dumps({"graph_id": r.graph_id, "error": str(e)}) + "\n")
                
    failure_fraction = num_failed / num_total if num_total else 0.0
    max_failure_fraction = float(config_dict.get("image", {}).get("max_failure_fraction", 0.0))
    if failure_fraction > max_failure_fraction:
        raise RuntimeError(
            f"Graph conversion failure fraction {failure_fraction:.6f} exceeds configured maximum "
            f"{max_failure_fraction:.6f}. Staging cache retained at {out_dir}. First failure: {first_failure}"
        )

    split_checksums = {
        split: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        for split, ids in saved_graph_ids.items()
    }
                
    from vino.transforms.fiedler_sign import (
        DEFAULT_PIPELINE, canonicalization_signature, summarize_sign_methods,
    )
    canon_cfg = config_dict.get("image", {}).get("canonicalization", {}) or {}
    try:
        sign_signature = canonicalization_signature(
            canon_cfg.get("sign_pipeline") or DEFAULT_PIPELINE,
            canon_cfg.get("sign_rule", "fiedler_cascade"),
        )
    except Exception:
        sign_signature = None

    manifest = {
        "dataset": dataset_name,
        "splits": splits,
        "num_total": num_total,
        "num_success": num_success,
        "num_failed": num_failed,
        "sign_signature": sign_signature,
        "sign_method_diagnostics": summarize_sign_methods(sign_methods),
        "output_dir": final_out_dir,
        "config_hash": conf_hash,
        "cache_format_version": int(config_dict.get("image", {}).get("cache_format_version", 2)),
        "split_strategy": config_dict.get("dataset", {}).get("split_strategy") or {
            "molhiv": "ogb_official",
            "bbbp": "scaffold",
            "esol": "scaffold",
            "synthetic": "generated",
        }.get(dataset_name, "generated"),
        "split_seed": config_dict.get("dataset", {}).get("split_seed"),
        "split_graph_ids": {s: sorted(ids) for s, ids in saved_graph_ids.items()},
        "split_checksums": split_checksums,
        "failure_fraction": failure_fraction,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        OmegaConf.save(config, f)
        
    os.replace(out_dir, final_out_dir)
    print(f"Done. Saved to {final_out_dir}")

if __name__ == "__main__":
    main()
