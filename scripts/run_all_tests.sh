#!/bin/bash
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "=============================================="
echo " RUNNING ALL TESTS IN $REPO_DIR/tests"
echo "=============================================="

for t in tests/test_*.py; do
    echo "=============================================="
    echo " RUNNING $t"
    echo "=============================================="
    python3 "$t"
done

echo "=============================================="
echo " ALL TESTS COMPLETED SUCCESSFULLY!"
echo "=============================================="
