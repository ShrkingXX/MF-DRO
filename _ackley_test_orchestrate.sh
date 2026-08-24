#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

SEEDS=(42 43 44)
# Sequential by method (slowest, MF-DRO, last); seeds within a method run
# in parallel. Skip-if-done (built into _ackley_test_worker.py) makes the
# already-completed MF-GP-UCB/seed42 and in-progress MF-DRO/seed42 smoke
# tests safe to include here unchanged.
METHODS=("MF-GP-UCB" "MF-MI-Greedy" "Greedy-MES" "MF-DRO")

mkdir -p /tmp/ackley_test_logs

for method in "${METHODS[@]}"; do
    echo "=== PHASE START: $method ===" | tee -a /tmp/ackley_test_logs/orchestrate.log
    for seed in "${SEEDS[@]}"; do
        logfile="/tmp/ackley_test_logs/${method}_seed${seed}.log"
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup .venv/bin/python3 _ackley_test_worker.py "$method" "$seed" \
            > "$logfile" 2>&1 &
    done
    wait
    echo "=== PHASE DONE: $method ===" | tee -a /tmp/ackley_test_logs/orchestrate.log
done

echo "=== ALL ACKLEY TEST JOBS COMPLETE ===" | tee -a /tmp/ackley_test_logs/orchestrate.log
