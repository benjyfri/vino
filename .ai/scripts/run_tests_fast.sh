#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ai/logs
LOG=".ai/logs/test_fast_$(date +%Y%m%d_%H%M%S).log"

{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "python=$(which python)"
  python --version

  if [ -d tests/unit ]; then
    pytest -q tests/unit
  elif [ -d tests ]; then
    pytest -q tests
  else
    echo "No tests directory found."
  fi
} 2>&1 | tee "$LOG"

echo "log=$LOG"
