# VINO: Graph-to-Image Foundation-Model Transfer

A fast proof-of-concept for graph-to-image foundation-model transfer using DINOv3.

The current runtime validates every executable config field, uses immutable versioned cache
manifests, defaults BBBP to a deterministic Bemis-Murcko scaffold split, and refuses partial
preprocessing success unless an explicit failure tolerance is configured.

## Quick Start

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate vino
pip install -e .
```

2. Tiny smoke test:
```bash
pytest -q
python scripts/build_graph_images.py --config configs/experiments/smoke_synthetic.yaml
python scripts/train.py --config configs/experiments/smoke_synthetic.yaml
```

The synthetic smoke configuration selects checkpoints by validation loss because tiny splits
may contain only one class. Molecular classification experiments select ROC-AUC and fail if the
validation split cannot define it.

## DINOv3 Weights and Pretrained Cache

Pretrained weights are NOT committed to the repository. The DINOv3 wrapper will automatically download and cache them using Hugging Face's Hub mechanism.

Ensure your `HF_HOME` is set properly, or allow it to download to the default cache directory (typically `~/.cache/huggingface/hub`).
To explicitly download weights:
```bash
python scripts/download_dinov3.py --model-name facebook/dinov3-vits16-pretrain-lvd1689m
```

## Data Building
```bash
python scripts/build_graph_images.py --config configs/data/bbbp.yaml --image-config configs/image/graph_image_default.yaml
```

## Training Frozen
```bash
python scripts/train.py --config configs/experiments/bbbp_frozen_vits16.yaml
```

## Training Light Finetune
```bash
python scripts/train.py --config configs/experiments/bbbp_finetune_vits16plus.yaml
```

## Running Ablations
Check `configs/image/` for ablation configs and run them with the train script.

Before trusting an experiment config, audit its field-to-runtime mapping:

```bash
python scripts/audit_experiment_configs.py 'configs/experiments/*.yaml'
```

Historical output configs were audited in
`.ai/reviews/vino_existing_run_validity_audit_20260715.json`; historical results are exploratory
because AMP, early stopping, automatic positive weighting, and some transform settings were not
executed by the former runtime.

## SLURM Usage
Submit jobs from the `sbatch/` directory:
```bash
sbatch sbatch/smoke_synthetic.sbatch
```

Production BBBP launchers require `VINO_DATA_DIR` to identify an immutable scaffold-split cache.
They accept `VINO_ROOT` and `CONDA_SH` instead of embedding personal paths. Validate launchers
without submitting jobs using `bash .ai/scripts/preview_sbatch.sh`.
