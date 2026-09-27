#!/bin/bash
cd /Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission
R=experiments/h211-mixr-on-k1
export SCRATCH=/private/tmp/claude-501/-Users-yurucui-Desktop-DRO-Code-DRO-aistats-submission/75c6514b-93bc-4c9a-8b80-662a37c15284/scratchpad
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
say "running smoke"; .venv/bin/python $R/code/smoke.py > $R/logs/smoke.log 2>&1
if ! grep -q "SMOKE: PASS" $R/logs/smoke.log; then say "SMOKE FAIL -- not launching"; grep -E "^MIXRK1|^K1TI|SMOKE|Traceback|Error" $R/logs/smoke.log | tail -8; exit 1; fi
say "smoke PASS"; grep -E "^MIXRK1|^K1TI|SMOKE" $R/logs/smoke.log; bash $R/code/launch.sh
