#!/bin/bash
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h221-rtg-credit-assignment; mkdir -p $R/logs $R/results/ckpt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
nrun(){ bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' '; }
launch(){ while [ "$(nrun)" -ge 15 ]; do sleep 120; done
  nohup .venv/bin/python $R/code/worker_$1.py Borehole_8D $2 > $R/logs/${1}_seed$2.log 2>&1 &
  sleep 3; say "launched $1 seed $2  (running: $(nrun)/15)"; }
for s in 42 43 44 45 46; do launch CRN $s; done
for s in 42 43 44 45 46; do launch BOTH $s; done
for s in 42 43 44 45 46; do launch ADV $s; done
say "all 15 launched; $(nrun)/15 running"
