#!/bin/bash
#SBATCH --job-name=vino_bbbp_dino_5s
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodelist=dym-lab[2-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source /home/benjamin.fri/miniconda3/etc/profile.d/conda.sh
conda activate vino

cd /home/benjamin.fri/PycharmProjects/vino

SEEDS=(42 1337 2025 3407 9001)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="data/processed/bbbp/d0ae770c_stratified_s42"
OUT_DIR="outputs/bbbp_full_dino_resize_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "============================================================"
echo "VINO DINOv3 resize-bilinear BBBP"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Seed: ${SEED}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"

echo "[python]"
which python
python --version

echo "[gpu]"
nvidia-smi

echo "[train]"
python -u scripts/train.py \
  --config configs/experiments/bbbp_full_resize_vits16.yaml \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}"

echo "[result]"
cat "${OUT_DIR}/result.json"

echo "============================================================"
echo "Done: $(date)"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"
