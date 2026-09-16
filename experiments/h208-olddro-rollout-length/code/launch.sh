#!/bin/bash
# h208 launcher: 6 arms x 10 seeds = 60 jobs through a 13-wide queue, longest
# arm first (h207 lesson: longest-first keeps every core busy to the end).
# 13 + h207's one running process = 14 <= 15.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h208-olddro-rollout-length
mkdir -p $R/logs $R/results
n=$(bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -gt 2 ]; then echo "ABORT: $n workers already running"; exit 1; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
JOBS=$R/logs/jobs.txt; : > $JOBS
for A in L8 L8-TRUNC L4 ESON-L4 L2 L1; do
  for s in 42 43 44 45 46 47 48 49 50 51; do
    echo "$A $s" >> $JOBS
  done
done
echo "[$(date '+%H:%M:%S')] launching $(wc -l < $JOBS | tr -d ' ') jobs, 13-wide"
xargs -P 13 -L 1 sh -c '.venv/bin/python3 '"$R"'/code/worker.py "$0" "$1" > '"$R"'/logs/"$0"_seed"$1".log 2>&1' < $JOBS
echo "[$(date '+%H:%M:%S')] queue drained"
