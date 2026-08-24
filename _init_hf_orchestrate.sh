#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

VARIANTS=("A" "B")
SEEDS=(42 43 44)

mkdir -p /tmp/init_hf_logs

JOBLIST=$(mktemp)
for v in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "$v $seed" >> "$JOBLIST"
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 4 -n 2 bash -c '
    v="$0"; seed="$1"
    logfile="/tmp/init_hf_logs/variant${v}_seed${seed}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 _init_hf_worker.py "$v" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL INIT_HF TEST JOBS COMPLETE ==="
