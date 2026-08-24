#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Hartmann_6D" "Borehole_8D")
VARIANTS=("off" "on")
SEEDS=(42)

mkdir -p /tmp/lf_screened_pilot_logs

JOBLIST=$(mktemp)
for bm in "${BENCHMARKS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "$v $bm $seed" >> "$JOBLIST"
        done
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 2 -n 3 bash -c '
    v="$0"; bm="$1"; seed="$2"
    logfile="/tmp/lf_screened_pilot_logs/${v}_${bm}_seed${seed}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 _lf_screened_pilot_worker.py "$v" "$bm" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL LF-SCREENED PILOT JOBS COMPLETE ==="
