#!/bin/sh
# Correct worker count. `pgrep -f 'code/worker'` OVER-COUNTS: a launcher whose
# argv carries job strings (e.g. run_all.py Currin_2D:MF-DRO:56 ...) matches the
# pattern too, and per-PID greps on the launcher's argument list can attribute
# one process as many. That produced a false 27-worker reading against a real 15
# on 2026-08-27, reported as a cap violation by another session before it was
# checked.
#
# A real worker is a process whose command invokes worker*.py WITH job arguments.
n=0
for pid in $(pgrep -f "code/worker" 2>/dev/null); do
  c=$(ps -o command= -p "$pid" 2>/dev/null)
  echo "$c" | grep -qE "worker(_mes)?\.py +[A-Za-z_0-9]+ +[A-Za-z0-9-]+ +[0-9]+" && n=$((n+1))
done
echo "$n"
