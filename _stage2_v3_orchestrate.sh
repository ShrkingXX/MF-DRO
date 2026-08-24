#!/bin/bash
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

BENCHMARKS=("Currin_2D" "Hartmann_6D" "Borehole_8D" "Ackley_10D")
SEEDS=(42 43 44 45 46)
# MF-DRO + SF-DRO already complete (40/40, skip-if-done will no-op them).
# Greedy-MES added: the logregret-vs-cost plot showed it (at its OLD 1x-
# sizing/cost_budget-terminated numbers) dramatically outperforming MF-DRO
# on Currin_2D/Borehole_8D -- and Greedy-MES directly shares KennedyOHaganGP
# with MF-DRO, so it's affected by this session's lognormal-prior fix,
# making the old numbers not a fair comparison. Rerunning at the current
# protocol (2x init sizing, fixed 100 iterations, cost tracked not budget-
# gated) for an apples-to-apples read. MF-GP-UCB/MF-MI-Greedy still held
# back at their OLD numbers for now (not requested this pass) -- MF-DRO
# +LFScreen also still held back. Revisit before any FINAL (non-
# intermediate) Stage 2 table -- see the worker's docstring for the full
# 6-method design if/when that's needed.
METHODS=("MF-DRO" "SF-DRO" "Greedy-MES")

mkdir -p /tmp/stage2_v3_logs

# Benchmark/seed-major, method-innermost -- interleaves the two methods
# throughout the sweep rather than running all MF-DRO jobs (slow) before
# any SF-DRO (comparatively fast) result appears.
JOBLIST=$(mktemp)
for bm in "${BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
            echo "$method $bm $seed" >> "$JOBLIST"
        done
    done
done

echo "Total jobs: $(wc -l < "$JOBLIST")"

cat "$JOBLIST" | xargs -P 15 -n 3 bash -c '
    method="$0"; bm="$1"; seed="$2"
    safe_method=$(echo "$method" | tr " " "_")
    logfile="/tmp/stage2_v3_logs/${safe_method}_${bm}_seed${seed}.log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 _stage2_v3_worker.py "$method" "$bm" "$seed" \
        > "$logfile" 2>&1
'

rm -f "$JOBLIST"
echo "=== ALL STAGE 2 V3 JOBS COMPLETE ==="
