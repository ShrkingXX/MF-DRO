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

# --list: print the enumeration this count summarises, one worker per line.
# A bare total cannot be reconciled; an enumeration can. Added 2026-08-28 after
# a peer session's count read 10 where the truth was 9 -- two defects compounding
# (a `ps|grep` matching the inspecting shell's own argv, and a zero-or-more digit
# class matching the pattern text itself) produced a number exactly one too large,
# which looked like the completion being waited for. The genus, which is what
# matters: ANY process inspection can match the process doing the inspecting.
if [ "$1" = "--list" ]; then
  ps -eo args 2>/dev/null \
    | grep -oE "experiments/h[0-9]+[^/]*/code/worker(_mes)?\.py +[A-Za-z_0-9]+ +[A-Za-z0-9-]+ +[0-9]+" \
    | sed 's|experiments/||; s|/code/worker\(_mes\)\?\.py||' | sort
fi
echo "$n"
