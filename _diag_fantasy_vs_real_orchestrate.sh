#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Hartmann_6D" "Ackley_10D")
SEEDS=(42 43 44)

mkdir -p /tmp/diag_fantasy_vs_real_logs

for bm in "${BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        logfile="/tmp/diag_fantasy_vs_real_logs/${bm}_seed${seed}.log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup .venv/bin/python3 _diag_fantasy_vs_real_conditioning.py "$bm" "$seed" \
            > "$logfile" 2>&1 &
    done
done
wait

echo "=== ALL FANTASY-VS-REAL DIAGNOSTIC JOBS COMPLETE ==="
