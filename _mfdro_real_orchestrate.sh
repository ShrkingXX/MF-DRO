#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

SEEDS=(42 43 44)
VARIANTS=("Simulated" "Real")

mkdir -p /tmp/mfdro_real_logs

JOBLIST=$(mktemp)
for v in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "$v $seed" >> "$JOBLIST"
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 6 -n 2 bash -c '
    v="$0"; seed="$1"
    logfile="/tmp/mfdro_real_logs/${v}_seed${seed}.log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python3 _mfdro_real_worker.py "$v" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL MF-DRO-REAL COMPARISON JOBS COMPLETE ==="
