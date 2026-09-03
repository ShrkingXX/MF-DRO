#!/bin/sh
# Count running experiment workers WITHOUT counting the counting command itself.
#
# WHY THIS EXISTS
#   Two separate sessions have now miscounted the fleet by matching their own
#   monitoring process. A shell invoked as `zsh -c '... pgrep -f ROI-Q30 ...'`
#   carries that pattern in its own argv, so any `pgrep -f`/`ps | grep` for the
#   pattern matches the shell running the query. The peer reported 27 workers on
#   15 cores this way; I reported h127 as 10/10 dispatched when it was 9/10 and
#   one seed had never launched.
#
#   Two compounding defects in my version:
#     1. the query matched its own shell, and
#     2. `grep -o 'ARM [0-9]*'` uses zero-or-more, so the literal text
#        "ARM [0-9]*" inside that shell's argv matched as "ARM " with no digits
#        and counted as a distinct seed.
#
#   Fixes: match only the worker.py path, exclude our own pid and any shell,
#   and require one-or-more digits.
set -e
# DEFAULT WIDENED from "code/worker.py" to "code/worker" on 2026-09-02.
#   h194 ran two arms from code/worker.py and code/worker_ctrl.py. The old default
#   matched only the first and reported 5 when 10 were live -- an UNDER-count, the
#   dangerous direction, since acting on it means launching past the 15-worker cap.
#   The same session had documented this hazard one tick earlier and repeated it
#   immediately, which is why the fix is in the tool rather than in a note.
#   "code/worker" still matches every real worker (worker.py, worker_*.py) and no
#   analysis script, which are named analyse.py / readout.py / stage0.py.
PAT="${1:-code/worker}"
ps -o pid=,args= -ax 2>/dev/null \
  | grep "$PAT" \
  | grep -v -e ' -c ' -e 'zsh' -e 'bash' -e 'sh -c' -e 'grep' \
  | grep -v "^ *$$ " \
  | sed 's/^ *//'
