#!/bin/bash
# h209: 25 runs, cap 15, longest-first (Hartmann first), remainder queued as slots free.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h209-mixr-replication
mkdir -p $R/logs $R/results/ckpt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
nrun(){ bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' '; }
launch(){ # arm bench seed
  while [ "$(nrun)" -ge 15 ]; do sleep 60; done
  nohup .venv/bin/python $R/code/worker_$1.py $2 $3 > $R/logs/${1}_${2%%_*}_seed$3.log 2>&1 &
  sleep 3; say "launched $1 $2 seed $3  (running: $(nrun)/15)"
}
for s in 42 43 44 45 46; do launch MIXR Hartmann_6D $s; launch NIR Hartmann_6D $s; done
for s in 42 43 44 45 46; do launch NIR120 Borehole_8D $s; done
for s in 47 48 49 50 51; do launch MIXR Borehole_8D $s; launch NIR Borehole_8D $s; done
say "all 25 launched; $(nrun)/15 running"
