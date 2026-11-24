#!/bin/bash
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "=============================================="
echo " RUNNING BENCHMARKS"
echo "=============================================="

python3 benchmarks/benchmark.py
python3 benchmarks/benchmark_hetero_composition.py

echo "=============================================="
echo " BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "=============================================="
