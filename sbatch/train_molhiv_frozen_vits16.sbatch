#!/bin/bash
#SBATCH --partition=dym
#SBATCH --nodelist=dym-lab2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR
eval "$(conda shell.bash hook)"
conda activate vino

export PYTHONPATH=$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm

python scripts/train.py --config configs/experiments/molhiv_frozen_vits16.yaml | tee logs/slurm/train_molhiv_frozen_vits16_${SLURM_JOB_ID}.log
