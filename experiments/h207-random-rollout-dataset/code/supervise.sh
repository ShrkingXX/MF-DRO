#!/bin/bash
# h207 supervisor v4: Stage 0 v3 already PASSED on this core (no core change since).
# Run Stage 0c (SC9 v2, oracle half with true-f labels), gate on it, then launch.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h207-random-rollout-dataset
export SCRATCH=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/75c6514b-93bc-4c9a-8b80-662a37c15284/scratchpad
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
grep -q "STAGE 0: PASS" $R/logs/stage0.log || { say "Stage 0 v3 log does not say PASS; refusing"; exit 1; }
say "running Stage 0c v3 (SC9 v3, oracle half, true-f labels)"
.venv/bin/python $R/code/stage0c_oracle.py > $R/logs/stage0c.log 2>&1
if ! grep -q "STAGE 0c: PASS" $R/logs/stage0c.log 2>/dev/null; then
  say "GATE MISS -- Stage 0c v3 (SC9) did not PASS. NOT launching."; grep -vE "Warning|_fwd|optimum" $R/logs/stage0c.log | tail -12; exit 1
fi
say "Stage 0c v3 PASS -- launching NIR, MIXR, MIXO into free slots (cap 15, h208 running)"
grep -E "SC9|d per member" $R/logs/stage0c.log
ORDER="MIXO MIXR NIR" bash $R/code/launch.sh
