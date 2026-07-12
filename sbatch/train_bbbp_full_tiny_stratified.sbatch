#!/bin/bash
#SBATCH --job-name=vino_bbbp_tiny_strat
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
echo "VINO full BBBP tiny sanity — stratified split"
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

DATA_DIR="data/processed/bbbp/d0ae770c_stratified_s42"
OUT_DIR="outputs/bbbp_full_tiny_stratified_${SLURM_JOB_ID}"

echo "[python]"
which python
python --version

echo "[tests]"
python -m compileall scripts vino tests
python -m pytest -q -rs

echo "[manifest]"
cat "${DATA_DIR}/manifest.json"

echo "[train]"
python scripts/train.py \
  --config configs/experiments/bbbp_full_debug_tiny.yaml \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}"

echo "[result]"
cat "${OUT_DIR}/result.json"

echo "============================================================"
echo "Done: $(date)"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"
