#!/bin/bash
# Orchestrator for the KO-MES paper experiment (exp_name "ko_mes_paper").
#
#   ./_ko_mes_orchestrate.sh [MAX_PARALLEL]
#
# Runs every (method, benchmark, seed) combination through _ko_mes_worker.py.
# The worker skips any run whose checkpoint already exists and writes results
# atomically, so this script is safe to interrupt and re-run -- it resumes
# rather than restarting.
#
# Job-pool rather than phase barriers: Stage 2's orchestrator waited for a
# whole method to finish across all benchmarks and seeds before starting the
# next, which idles most slots whenever one run is much slower than its
# siblings (Hartmann_6D runs are far longer than Currin_2D ones). Here every
# job goes into one pool with a fixed number of slots.
set -uo pipefail
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission

MAX_PARALLEL="${1:-6}"          # 15 cores available; each run is single-threaded
                                # but BLAS inside torch is not, so leave headroom
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

BENCHMARKS=("Currin_2D" "Hartmann_6D" "Borehole_8D")
SEEDS=(42 43 44 45 46)
METHODS=("KO-MES" "Additive-MES" "Additive-MES-Song" "SF-MES" "MF-GP-UCB" "MF-MI-Greedy")

LOGDIR="results/ko_mes_paper/logs"
mkdir -p "$LOGDIR" "results/ko_mes_paper/checkpoints"
PROGRESS="$LOGDIR/progress.log"

echo "=== START $(date '+%F %T') max_parallel=$MAX_PARALLEL ===" | tee -a "$PROGRESS"

for method in "${METHODS[@]}"; do
    for bm in "${BENCHMARKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Block until a slot frees up.
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
                wait -n 2>/dev/null || sleep 1
            done
            logfile="$LOGDIR/${method}_${bm}_seed${seed}.log"
            (
                .venv/bin/python3 _ko_mes_worker.py "$method" "$bm" "$seed" \
                    > "$logfile" 2>&1
                tail -n 1 "$logfile" >> "$PROGRESS"
            ) &
        done
    done
done

wait
echo "=== ALL COMPLETE $(date '+%F %T') ===" | tee -a "$PROGRESS"
