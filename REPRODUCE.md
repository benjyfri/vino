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
