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
