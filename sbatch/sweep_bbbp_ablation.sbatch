#!/bin/bash
#SBATCH --partition=dym
#SBATCH --nodelist=dym-lab2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --array=0-4

cd $SLURM_SUBMIT_DIR
eval "$(conda shell.bash hook)"
conda activate vino

export PYTHONPATH=$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm

VARIANTS=("default" "ablate_topology_only" "ablate_topology_node" "ablate_raw_adj" "ablate_random_order")
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Running variant: $VARIANT"

OUT_DIR="outputs/bbbp_ablation_${VARIANT}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python scripts/train.py \
  --config configs/experiments/ablation_bbbp.yaml \
  --image-config "configs/image/graph_image_${VARIANT}.yaml" \
  --output_dir "${OUT_DIR}" | tee "logs/slurm/sweep_bbbp_${VARIANT}_${SLURM_JOB_ID}.log"
