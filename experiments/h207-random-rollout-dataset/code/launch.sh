#!/bin/bash
# h207 Stage 1 launcher, LONGEST-FIRST queue so all 15 cores stay busy.
#
# 20 jobs on 15 slots. Expected per-seed wall (60 MES rollouts/iter ~ 85 min on
# h206N): W ~2x (300 rollouts generated), MIX ~1.5x (120), NIR ~1x (60 MES + IR
# label), R <1x (60 random, no MES argmax). Launching the slow arm LAST would
# put its whole duration after the fast arms finish -- 5 workers on 15 cores
# for ~3 h. Launching it FIRST overlaps it with everything else:
#     last-first:  wall ~ T_NIR + T_W  ~ 85 + 170 = 255 min, tail 10 cores idle
#     longest-first: wall ~ max(T_W, T_NIR + T_R) ~ max(170, 155) = 170 min
# Threads stay at 1/worker: torch.set_num_threads mid-run changes reduction
# order and would break bit-reproducibility against a 1-thread run.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h207-random-rollout-dataset
mkdir -p $R/logs $R/results/ckpt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
nrun(){ bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' '; }
ORDER="${ORDER:-MIXO MIXR NIR}"
say "queue order (longest first): $ORDER"
for A in $ORDER; do
  for s in 42 43 44 45 46; do
    while [ "$(nrun)" -ge 15 ]; do sleep 60; done
    nohup .venv/bin/python $R/code/worker_$A.py Borehole_8D $s > $R/logs/${A}_seed${s}.log 2>&1 &
    sleep 3
    say "launched $A seed $s  (running: $(nrun)/15)"
  done
done
say "all launched; $(nrun)/15 running"
