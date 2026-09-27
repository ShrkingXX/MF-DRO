#!/bin/bash
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h213-label-rng-matched
export SCRATCH=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/75c6514b-93bc-4c9a-8b80-662a37c15284/scratchpad
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
say "waiting for h212 smoke to finish before adding load"
while pgrep -f "h212-rtg-target-schema/code/smoke.py" > /dev/null; do sleep 30; done
say "running h213 smoke (parity gate)"; .venv/bin/python $R/code/smoke.py > $R/logs/smoke.log 2>&1
if ! grep -q "SMOKE: PASS" $R/logs/smoke.log; then say "SMOKE FAIL -- not launching"; grep -E "^TI-PAR|^CTRL|^GATE|SMOKE|Traceback|Error" $R/logs/smoke.log | tail -10; exit 1; fi
say "smoke PASS -- parity confirmed"; grep -E "^TI-PAR|^CTRL|^GATE|SMOKE|first 3" $R/logs/smoke.log; bash $R/code/launch.sh
