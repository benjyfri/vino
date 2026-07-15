#!/bin/bash
#SBATCH --job-name=vino_bbbp_dino_resize
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
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "VINO full BBBP DINOv3 resize-bilinear"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Partition: ${SLURM_JOB_PARTITION:-none}"
echo "Account: dym-lab"
echo "QOS: normal"
echo "Node list: ${SLURM_JOB_NODELIST:-none}"
echo "Repo: ${VINO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "============================================================"

VINO_ROOT="${VINO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vino

cd "${VINO_ROOT}"

: "${VINO_DATA_DIR:?Export VINO_DATA_DIR with the immutable scaffold-split cache path}"
DATA_DIR="${VINO_DATA_DIR}"
OUT_DIR="outputs/bbbp_full_dino_resize_${SLURM_JOB_ID}"

echo "[python]"
which python
python --version

echo "[gpu]"
nvidia-smi

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("num devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available despite requesting a GPU.")
print("device:", torch.cuda.get_device_name(0))
PY

echo "[DINOv3 access check]"
python - <<'PY'
from huggingface_hub import HfApi

model = "facebook/dinov3-vits16-pretrain-lvd1689m"
api = HfApi()

try:
    print("HF user:", api.whoami().get("name"))
    info = api.model_info(model)
    print("model_info ok:", info.modelId)
except Exception as e:
    raise RuntimeError(
        "DINOv3 Hugging Face access check failed. "
        "You are either not logged in or not yet approved for the gated DINOv3 repo. "
        f"Original error: {type(e).__name__}: {e}"
    )
PY

echo "[tests]"
python -m compileall scripts vino tests
python -m pytest -q -rs

echo "[manifest]"
cat "${DATA_DIR}/manifest.json"

echo "[train]"
python -u scripts/train.py \
  --config configs/experiments/bbbp_full_resize_vits16.yaml \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}"

echo "[result]"
cat "${OUT_DIR}/result.json"

echo "============================================================"
echo "Done: $(date)"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"
