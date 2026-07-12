#!/bin/bash
#SBATCH --job-name=vino_bbbp_tiny
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "VINO full BBBP tiny sanity"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Partition: ${SLURM_JOB_PARTITION:-none}"
echo "Account: dym-lab"
echo "QOS: normal"
echo "Node list: ${SLURM_JOB_NODELIST:-none}"
echo "Repo: /home/benjamin.fri/PycharmProjects/vino"
echo "============================================================"

source /home/benjamin.fri/miniconda3/etc/profile.d/conda.sh
conda activate vino

cd /home/benjamin.fri/PycharmProjects/vino

mkdir -p logs outputs

echo "[python]"
which python
python --version

echo "[slurm]"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-none}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-none}"

echo "[tests]"
python -m compileall scripts vino tests
python -m pytest -q -rs

echo "[cache check]"
python - <<'PY'
import json
from pathlib import Path

manifest = Path("data/processed/bbbp/d0ae770c/manifest.json")
if not manifest.exists():
    raise FileNotFoundError(f"Missing full BBBP manifest: {manifest}")

d = json.loads(manifest.read_text())
print(json.dumps(d, indent=2))

assert d["dataset"] == "bbbp"
assert d["num_success"] == 2039
assert d["num_failed"] == 0
PY

OUT_DIR="outputs/bbbp_full_tiny_${SLURM_JOB_ID}"

echo "[train]"
python scripts/train.py \
  --config configs/experiments/bbbp_full_debug_tiny.yaml \
  --output_dir "${OUT_DIR}"

echo "[result]"
cat "${OUT_DIR}/result.json"

echo "============================================================"
echo "Done: $(date)"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"
