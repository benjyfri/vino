#!/usr/bin/env bash
set -euo pipefail

echo "== repo =="
pwd

echo
echo "== git =="
git status --short || true
git branch --show-current || true
git rev-parse --short HEAD || true

echo
echo "== python =="
which python || true
python --version || true

echo
echo "== imports =="
python - <<'PY'
mods = ["vino", "torch", "numpy", "pytest", "networkx", "sklearn"]
for m in mods:
    try:
        __import__(m)
        print(f"{m}: ok")
    except Exception as e:
        print(f"{m}: FAIL: {e}")
PY

echo
echo "== tests discovered =="
find tests -maxdepth 4 -type f \( -name 'test*.py' -o -name '*test*.py' \) | sort | head -120 || true
