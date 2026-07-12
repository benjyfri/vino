#!/bin/bash
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=dym-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%1
#SBATCH --job-name=bbbp_dino_interface
#SBATCH --output=logs/bbbp_dino_interface_%A_%a.out
#SBATCH --error=logs/bbbp_dino_interface_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "VINO BBBP DINO interface ablation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-NA}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "============================================================"

cd /home/benjamin.fri/PycharmProjects/vino

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vino

mkdir -p outputs

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi || true

VARIANTS=(
  no_stem
  residual_stem
  train_patch_embed
  residual_stem_train_patch_embed
)

SEEDS=(42 2025 9001)

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

VARIANT="${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

CONFIG="configs/experiments/bbbp_full_resize_vits16_${VARIANT}.yaml"
DATA_DIR="data/processed/bbbp/d0ae770c_stratified_s42"
OUT_DIR="outputs/bbbp_dino_interface_${VARIANT}_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo
echo "Resolved run:"
echo "  VARIANT_IDX=${VARIANT_IDX}"
echo "  SEED_IDX=${SEED_IDX}"
echo "  VARIANT=${VARIANT}"
echo "  SEED=${SEED}"
echo "  CONFIG=${CONFIG}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  OUT_DIR=${OUT_DIR}"

echo
echo "Checking required paths..."
test -f "${CONFIG}"
test -d "${DATA_DIR}"

echo
echo "Resolved config preview:"
python - <<PY
from vino.utils.config import load_resolved_config

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

cfg = load_resolved_config("${CONFIG}")
if OmegaConf is not None and not isinstance(cfg, dict):
    d = OmegaConf.to_container(cfg, resolve=True)
else:
    d = cfg

model = d.get("model", {})
train = d.get("train", d.get("training", {}))

print("model_name:", model.get("model_name") or model.get("backbone_name"))
print("freeze_mode:", model.get("freeze_mode"))
print("input_mode:", model.get("input_mode"))
print("input_size:", model.get("input_size") or model.get("model_input_size"))
print("patch_size:", model.get("patch_size"))
print("token_grid_size:", model.get("token_grid_size"))
print("resize_mode:", model.get("resize_mode"))
print("input_stem:", model.get("input_stem"))
print("train:", train)
PY

echo
echo "Starting training..."
python -u scripts/train.py \
  --config "${CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}"

echo
echo "Training finished."
echo "Result:"
cat "${OUT_DIR}/result.json"

echo
echo "Done: $(date)"
