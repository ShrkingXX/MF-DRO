#!/bin/bash
# h207 supervisor: wait for Stage 0 to exit, GATE on "STAGE 0: PASS", then run the
# queued launcher. Sleeps while waiting; not a compute worker.
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h207-random-rollout-dataset
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
say "waiting for Stage 0 (sanity.py) to exit..."
while pgrep -f "h207-random-rollout-dataset/code/sanity.py" > /dev/null; do sleep 20; done
say "sanity.py exited; reading gate"
if ! grep -q "STAGE 0: PASS" $R/logs/stage0.log 2>/dev/null; then
  say "GATE MISS -- Stage 0 did not PASS. NOT launching."
  grep -E "^SC|STAGE 0|->" $R/logs/stage0.log | head -30; exit 1
fi
say "STAGE 0: PASS -- longest-first queued launch"
grep -E "^SC|STAGE 0" $R/logs/stage0.log
# Order arms by SC7's measured per-iteration wall (desc) so the queue is
# longest-first by MEASUREMENT, not by my estimate. Falls back to W MIX NIR R.
ORDER="MIX NIR R W"   # v3: W last (Stage 0b: its dataset never beats the incumbent)
say "order: $ORDER"
ORDER="$ORDER" bash $R/code/launch.sh
