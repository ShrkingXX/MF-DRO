#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Currin_2D" "Hartmann_6D" "Borehole_8D")
MULTS=("0.5" "1.0" "2.0")
SEEDS=(42 43 44)

mkdir -p /tmp/init_size_tuning_logs

JOBLIST=$(mktemp)
for bm in "${BENCHMARKS[@]}"; do
    for mult in "${MULTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$bm $mult $seed" >> "$JOBLIST"
        done
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 10 -n 3 bash -c '
    bm="$0"; mult="$1"; seed="$2"
    logfile="/tmp/init_size_tuning_logs/${bm}_mult${mult}_seed${seed}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 _init_size_tuning_worker.py "$bm" "$mult" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL INIT_SIZE_TUNING JOBS COMPLETE ==="
