#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Currin_2D" "Hartmann_6D" "Borehole_8D")
SEEDS=(42 43 44 45 46)
METHODS=("SF-DRO" "MF-GP-UCB" "MF-MI-Greedy" "Greedy-MES" "MF-DRO")

mkdir -p /tmp/stage2_logs

for method in "${METHODS[@]}"; do
    echo "=== PHASE START: $method ===" | tee -a /tmp/stage2_logs/orchestrate.log
    for bm in "${BENCHMARKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            safe_method=$(echo "$method" | tr ' ' '_')
            logfile="/tmp/stage2_logs/${safe_method}_${bm}_seed${seed}.log"
            nohup .venv/bin/python3 _stage2_worker.py "$method" "$bm" "$seed" \
                > "$logfile" 2>&1 &
        done
    done
    wait
    echo "=== PHASE DONE: $method ===" | tee -a /tmp/stage2_logs/orchestrate.log
done

echo "=== ALL STAGE 2 PHASES COMPLETE ===" | tee -a /tmp/stage2_logs/orchestrate.log
