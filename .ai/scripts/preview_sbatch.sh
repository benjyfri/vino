#!/usr/bin/env bash
set -euo pipefail

show_preview=false
if (($# == 0)); then
  set -- sbatch/*.sbatch
else
  show_preview=true
fi

status=0
for file in "$@"; do
  echo "checking ${file}"
  bash -n "${file}"
  if ${show_preview}; then
    echo "== SBATCH PREVIEW ONLY: ${file} =="
    sed -n '1,260p' "${file}"
    echo "== Extracted resources =="
    grep -E '^#SBATCH --partition=|^#SBATCH --account=|^#SBATCH --qos=|^#SBATCH --gres=|^#SBATCH --time=|^#SBATCH --mem=|^#SBATCH --cpus-per-task=|^#SBATCH --nodelist=|^#SBATCH --array=' "${file}" || true
  fi
  if grep -nE 'cd /home/|source /home/|DATA_DIR="data/processed/.+/[0-9a-f]{8}' "${file}"; then
    echo "non-portable path or cache hash found in ${file}" >&2
    status=1
  fi
done
echo "This script validates/previews only; it never submits jobs."
exit "${status}"
