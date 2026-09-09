#!/bin/bash
# h206 supervisor: wait for Stage 0 to exit, GATE on "STAGE 0: PASS", launch only
# if it passes. Sleeps while waiting, so it is not a compute worker and does not
# count against the 15-worker cap. Same pattern as h201/code/supervise.sh.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h206-no-position
OUT=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/75c6514b-93bc-4c9a-8b80-662a37c15284/tasks/by8jsnt39.output
say(){ echo "[$(date '+%H:%M:%S')] $*"; }

say "waiting for Stage 0 (sanity.py) to exit..."
while pgrep -f "h206-no-position/code/sanity.py" > /dev/null; do sleep 20; done
say "sanity.py exited; reading gate"
cp "$OUT" $R/logs/stage0.log 2>/dev/null

if ! grep -q "STAGE 0: PASS" $R/logs/stage0.log 2>/dev/null; then
  say "GATE MISS -- Stage 0 did not PASS. NOT launching."
  grep -E "^SC|STAGE 0|max\|dW\||seq_len" $R/logs/stage0.log 2>/dev/null | head -20
  exit 1
fi
say "STAGE 0: PASS -- launching 10 workers"
grep -E "^SC|STAGE 0|max\|dW\|" $R/logs/stage0.log | head -20
bash $R/code/launch.sh
