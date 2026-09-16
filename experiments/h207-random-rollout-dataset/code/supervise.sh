#!/bin/bash
# h207 supervisor v3: gate on Stage 0 v3 PASS, then run Stage 0c (SC9, oracle half),
# gate on that, then launch NIR / MIXR / MIXO = 15 workers in one go.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h207-random-rollout-dataset
export SCRATCH=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/75c6514b-93bc-4c9a-8b80-662a37c15284/scratchpad
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
say "waiting for Stage 0 v3 (sanity.py) to exit..."
while pgrep -f "h207-random-rollout-dataset/code/sanity.py" > /dev/null; do sleep 20; done
if ! grep -q "STAGE 0: PASS" $R/logs/stage0.log 2>/dev/null; then
  say "GATE MISS -- Stage 0 v3 did not PASS. NOT launching."; grep -E "^SC|->|STAGE" $R/logs/stage0.log | head -30; exit 1
fi
say "Stage 0 v3 PASS -- running Stage 0c (SC9, oracle half)"
.venv/bin/python $R/code/stage0c_oracle.py > $R/logs/stage0c.log 2>&1
if ! grep -q "STAGE 0c: PASS" $R/logs/stage0c.log 2>/dev/null; then
  say "GATE MISS -- Stage 0c (SC9) did not PASS. NOT launching."; grep -vE "Warning|_fwd|optimum" $R/logs/stage0c.log | tail -12; exit 1
fi
say "Stage 0c PASS -- launching NIR, MIXR, MIXO (15 workers)"
grep -E "SC9|d per member" $R/logs/stage0c.log
ORDER="MIXO MIXR NIR" bash $R/code/launch.sh
