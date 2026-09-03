#!/bin/bash
# h201 supervisor: wait for slots, run the SC1 readout GATE, launch only if it passes.
# Sleeps while waiting, so it is not a compute worker and does not count to the cap.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h201-oracle-teacher-window
export SCRATCH=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/066b0360-2a64-4497-9920-3b47bbf67828/scratchpad
say(){ echo "[$(date '+%H:%M:%S')] $*"; }

nrun(){ bash tools/count_workers.sh 2>/dev/null | wc -l | tr -d ' '; }
wait_for_slots(){   # $1 = how many free slots are needed
  while [ "$(( 15 - $(nrun) ))" -lt "$1" ]; do sleep 60; done
}

say "waiting for $1 free slots to run the SC1 gate..."
wait_for_slots 1
say "running SC1 readout gate"
.venv/bin/python $R/code/sc_readout.py > $R/logs/sc_readout.log 2>&1
if ! grep -q "STAGE 0: PASS" $R/logs/sc_readout.log; then
  say "GATE MISS -- h201 premise not confirmed. NOT launching. See logs/sc_readout.log"
  grep -E "max index|GATE|STAGE 0|tau7|tau=0" $R/logs/sc_readout.log
  exit 1
fi
say "SC1 PASS -- the window moves the readout. launching arm A (K=8)"
grep -E "max index|tau7_max_dev|tau=0 across" $R/logs/sc_readout.log

wait_for_slots 5
for s in 42 43 44 45 46; do
  nohup .venv/bin/python $R/code/worker_A.py Borehole_8D $s > $R/logs/A_seed$s.log 2>&1 &
done
say "arm A launched ($(nrun) workers)"

while [ "$(ls $R/results/ 2>/dev/null | grep -c 'H201A.*\.json$')" -lt 5 ]; do sleep 120; done
say "arm A complete. launching arm B (K=1, matched control)"
wait_for_slots 5
for s in 42 43 44 45 46; do
  nohup .venv/bin/python $R/code/worker_B.py Borehole_8D $s > $R/logs/B_seed$s.log 2>&1 &
done
say "arm B launched ($(nrun) workers)"
while [ "$(ls $R/results/ 2>/dev/null | grep -c 'H201B.*\.json$')" -lt 5 ]; do sleep 120; done
say "BOTH ARMS COMPLETE"
grep -h "^\[done\]" $R/logs/*.log
