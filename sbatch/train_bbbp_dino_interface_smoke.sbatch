#!/bin/bash
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodelist=dym-lab[2-3]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=bbbp_smoke
#SBATCH --output=logs/bbbp_smoke_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vino

cd /home/benjamin.fri/PycharmProjects/vino

CONFIG="configs/experiments/bbbp_full_resize_vits16_residual_stem.yaml"
DATA_DIR="data/processed/bbbp/d0ae770c_stratified_s42"
OUT_DIR="outputs/bbbp_dino_residual_stem_smoke_seed42_${SLURM_JOB_ID}"
SEED=42

echo "============================================================"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi
echo "Config Path: $CONFIG"
echo "Seed: $SEED"
echo "Output Dir: $OUT_DIR"
echo "============================================================"

python -u scripts/train.py \
    --config "${CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --seed "${SEED}"

echo "============================================================"
echo "Result summary:"
cat "${OUT_DIR}/result.json" || echo "result.json not found."
echo "============================================================"
