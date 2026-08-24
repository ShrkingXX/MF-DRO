#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Hartmann_6D" "Currin_2D" "Borehole_8D")
EPOCHS=(10 20 30 100)
SEEDS=(42 43 44)

mkdir -p /tmp/num_epochs_v2_logs

JOBLIST=$(mktemp)
for bm in "${BENCHMARKS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$bm $ep $seed" >> "$JOBLIST"
        done
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 8 -n 3 bash -c '
    bm="$0"; ep="$1"; seed="$2"
    logfile="/tmp/num_epochs_v2_logs/${bm}_epochs${ep}_seed${seed}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 _num_epochs_ablation_worker_v2.py "$bm" "$ep" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL NUM_EPOCHS ABLATION V2 JOBS COMPLETE ==="
