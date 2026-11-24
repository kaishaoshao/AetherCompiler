#!/bin/bash
set -e
cd /home/kaikai/learn/tiny_graph_compiler
for t in test*.py; do
    echo "=============================================="
    echo " RUNNING $t"
    echo "=============================================="
    python3 "$t"
done
echo "=============================================="
echo " ALL TESTS COMPLETED SUCCESSFULLY!"
echo "=============================================="
