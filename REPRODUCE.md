# Reproduce

To reproduce the experiments in VINO, follow these exact commands.

1. Setup environment:
```bash
conda env create -f environment.yml
conda activate vino
pip install -e .
```

2. Run smoke tests (no internet required):
```bash
pytest -q
python scripts/build_graph_images.py --config configs/experiments/smoke_synthetic.yaml
python scripts/train.py --config configs/experiments/smoke_synthetic.yaml
```

3. Build Datasets (e.g., BBBP) on cluster:
```bash
sbatch sbatch/build_bbbp_images.sbatch
```

4. Train models:
```bash
sbatch sbatch/train_bbbp_frozen_vits16.sbatch
```

5. Validate configuration and HPC launchers without running an experiment:

```bash
python scripts/audit_experiment_configs.py 'configs/experiments/*.yaml'
bash .ai/scripts/preview_sbatch.sh
```

6. For a BBBP production run, build a scaffold-split cache from the current code, inspect its
`manifest.json`, record `config_hash`, `cache_format_version`, and `split_checksums`, then export
its immutable path:

```bash
export VINO_DATA_DIR=/absolute/path/to/data/processed/bbbp/<hash>
export VINO_ROOT=/absolute/path/to/vino
sbatch sbatch/train_bbbp_frozen_vits16.sbatch
```

Do not reuse historical random/stratified caches for scaffold-split comparisons. Final paper
evaluation must follow `configs/experiments/PAPER_PROTOCOL.md`, aggregate at least five seeds,
and keep test metrics out of architecture selection.

## W&B ablation

The Bayesian discovery sweep is capped operationally by the SLURM launcher: eight agents with
40 assignments each (320 trials). Before submission, export `VINO_DATA_DIR`, create the sweep,
and pass the returned path as `WANDB_SWEEP_ID`. The assistant does not submit jobs:

```bash
python scripts/create_wandb_frozen_sweep.py --dry-run
python scripts/create_wandb_frozen_sweep.py --entity <team>
bash .ai/scripts/preview_sbatch.sh sbatch/wandb_bbbp_frozen_dino_sweep.sbatch

# Human submission only:
# sbatch --export=ALL,WANDB_SWEEP_ID=<entity>/vino_bbbp/<id>,VINO_DATA_DIR="$VINO_DATA_DIR" \
#   sbatch/wandb_bbbp_frozen_dino_sweep.sbatch
```

Use `scripts/select_best_wandb_ablation.py` on validation ROC-AUC, then create the locked
five-seed confirmation grid with `scripts/create_wandb_confirmation_sweep.py`. Do not select the
winner by test ROC-AUC.
