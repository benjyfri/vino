#!/bin/bash
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=dym-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=bbbp_frozen_abls
#SBATCH --output=logs/bbbp_frozen_abls_%j.out
#SBATCH --error=logs/bbbp_frozen_abls_%j.err

set -euo pipefail

echo "============================================================"
echo "VINO BBBP frozen-DINO ablation suite"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "============================================================"

VINO_ROOT="${VINO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
cd "${VINO_ROOT}"
source "${CONDA_SH}"
conda activate vino

mkdir -p logs outputs reports configs/experiments/generated_ablation_configs

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi || true

echo
echo "Generating ablation configs and manifest..."
python scripts/make_bbbp_frozen_ablation_configs.py

MANIFEST="configs/experiments/generated_ablation_configs/bbbp_frozen_ablation_manifest.csv"
: "${VINO_DATA_DIR:?Export VINO_DATA_DIR with the immutable scaffold-split cache path}"
export VINO_DATA_DIR
DATA_DIR="${VINO_DATA_DIR}"
SUMMARY_CSV="reports/bbbp_frozen_ablation_suite_${SLURM_JOB_ID}.csv"

test -f "${MANIFEST}"
test -d "${DATA_DIR}"

echo
echo "Manifest:"
cat "${MANIFEST}"

echo
echo "Preflight: check whether channel-mask keys appear in code."
echo "This must be non-empty for real channel ablations."
grep -R "channel_indices\|channel_mask\|channel_ablation" -n scripts vino || true

echo
echo "Starting sequential ablation runs..."

python - <<'PY'
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MANIFEST = Path("configs/experiments/generated_ablation_configs/bbbp_frozen_ablation_manifest.csv")
DATA_DIR = os.environ["VINO_DATA_DIR"]
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
SUMMARY_CSV = Path(f"reports/bbbp_frozen_ablation_suite_{SLURM_JOB_ID}.csv")

rows = list(csv.DictReader(MANIFEST.open()))
summary_rows = []

for idx, row in enumerate(rows):
    phase = row["phase"]
    variant = row["variant"]
    seed = int(row["seed"])
    config = row["config"]

    out_dir = Path(f"outputs/bbbp_frozen_ablation_{phase}_{variant}_seed{seed}_{SLURM_JOB_ID}_{idx}")
    result_path = out_dir / "result.json"

    print("\n" + "=" * 120, flush=True)
    print(f"[{idx+1}/{len(rows)}] phase={phase} variant={variant} seed={seed}", flush=True)
    print(f"config={config}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print("=" * 120, flush=True)

    if result_path.exists():
        print(f"Skipping existing completed run: {result_path}", flush=True)
    else:
        cmd = [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--config", config,
            "--data_dir", DATA_DIR,
            "--output_dir", str(out_dir),
            "--seed", str(seed),
        ]

        start = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - start

        if rc != 0:
            print(f"FAILED run idx={idx} phase={phase} variant={variant} seed={seed} rc={rc}", flush=True)
            raise SystemExit(rc)

        print(f"Finished run in {elapsed/60:.2f} minutes", flush=True)

    if not result_path.exists():
        raise FileNotFoundError(result_path)

    d = json.loads(result_path.read_text())

    summary_rows.append({
        "idx": idx,
        "phase": phase,
        "variant": variant,
        "seed": seed,
        "config": config,
        "out_dir": str(out_dir),
        "model_name": d.get("model_name"),
        "freeze_mode": d.get("freeze_mode"),
        "input_mode": d.get("input_mode"),
        "resize_mode": d.get("resize_mode"),
        "input_stem_enabled": d.get("input_stem_enabled"),
        "input_stem_type": d.get("input_stem_type"),
        "patch_embed_trainable": d.get("patch_embed_trainable"),
        "num_trainable_params": d.get("num_trainable_params"),
        "best_epoch": d.get("best_epoch"),
        "val_roc_auc": d.get("val_roc_auc"),
        "test_roc_auc": d.get("test_roc_auc"),
        "val_loss": d.get("val_loss"),
        "test_loss": d.get("test_loss"),
    })

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Updated summary: {SUMMARY_CSV}", flush=True)

print("\nAll sequential ablations completed.", flush=True)
print(f"Summary CSV: {SUMMARY_CSV}", flush=True)
PY

echo
echo "Sequential suite finished. Final summary:"
cat "${SUMMARY_CSV}"

echo
echo "Grouped summary by phase/variant:"
python - <<'PY'
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path
import os

p = Path(f"reports/bbbp_frozen_ablation_suite_{os.environ.get('SLURM_JOB_ID')}.csv")
rows = list(csv.DictReader(p.open()))

groups = defaultdict(list)
for r in rows:
    if r["test_roc_auc"] not in ["", "None", "null"]:
        groups[(r["phase"], r["variant"])].append(float(r["test_roc_auc"]))

print(f"{'phase':<14} {'variant':<42} {'n':>3} {'test_mean':>10} {'test_std':>10}")
for (phase, variant), vals in sorted(groups.items()):
    mean = stats.mean(vals)
    std = stats.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{phase:<14} {variant:<42} {len(vals):>3} {mean:>10.4f} {std:>10.4f}")
PY

echo
echo "Done: $(date)"
