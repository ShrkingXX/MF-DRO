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
ORDER=$(grep "^SC7" $R/logs/stage0.log | .venv/bin/python -c '
import sys,re
t=sys.stdin.read()
m=re.search(r"MES-20/member ([\d.]+)s\s+random-100/member ([\d.]+)s\s+random-100->top20/member ([\d.]+)s",t)
if not m: print("W MIX NIR R"); sys.exit()
mes,rnd,w=map(float,m.groups())
est={"W":w,"MIX":mes+rnd*0.2,"NIR":mes,"R":rnd*0.2}
print(" ".join(sorted(est,key=lambda k:-est[k])))')
say "order from SC7 timing: $ORDER"
ORDER="$ORDER" bash $R/code/launch.sh
