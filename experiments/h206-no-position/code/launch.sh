#!/bin/bash
# h206 Stage 1 launcher. 2 arms x 5 seeds = 10 workers x 1 thread = 10 <= 15.
# Stage 0 must have PASSED and been committed before this runs.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h206-no-position
mkdir -p $R/logs $R/results/ckpt
n=$(bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -gt 5 ]; then echo "ABORT: $n workers already running, cap is 15"; exit 1; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for A in N P; do
  for s in 42 43 44 45 46; do
    nohup .venv/bin/python $R/code/worker_$A.py Borehole_8D $s \
      > $R/logs/${A}_seed${s}.log 2>&1 &
    sleep 1
  done
done
sleep 20
echo "launched: $(bash tools/count_workers.sh | wc -l | tr -d ' ') workers"
