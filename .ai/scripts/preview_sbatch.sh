#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="${1:-}"

if [ -z "$SBATCH_FILE" ] || [ ! -f "$SBATCH_FILE" ]; then
  echo "Usage: bash .ai/scripts/preview_sbatch.sh path/to/job.sbatch"
  exit 2
fi

echo "== SBATCH PREVIEW ONLY =="
echo "File: $SBATCH_FILE"
echo

sed -n '1,260p' "$SBATCH_FILE"

echo
echo "== Extracted resources =="
grep -E '^#SBATCH --partition=|^#SBATCH --account=|^#SBATCH --qos=|^#SBATCH --gres=|^#SBATCH --time=|^#SBATCH --mem=|^#SBATCH --cpus-per-task=|^#SBATCH --nodelist=|^#SBATCH --array=' "$SBATCH_FILE" || true

echo
echo "This script does NOT submit."
echo "Manual submission only:"
echo "  sbatch $SBATCH_FILE"
