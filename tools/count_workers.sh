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
PAT="${1:-code/worker.py}"
ps -o pid=,args= -ax 2>/dev/null \
  | grep "$PAT" \
  | grep -v -e ' -c ' -e 'zsh' -e 'bash' -e 'sh -c' -e 'grep' \
  | grep -v "^ *$$ " \
  | sed 's/^ *//'
