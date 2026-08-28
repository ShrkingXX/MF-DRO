#!/bin/sh
# Claim the next free experiment ID atomically. Usage: tools/claim_id.sh <slug>
#
# WHY THIS EXISTS
#   Three ID collisions in one day (h88, h89, h97), all from the same cause: a
#   reservation written inside one session's protocol file is invisible to the
#   other session at the moment it picks a number. Any scheme based on reading a
#   note has a race between "read the note" and "create the directory".
#
#   `mkdir` does not. It is atomic and it fails if the directory exists, so the
#   directory IS the registry and the claim IS the creation. Two sessions racing
#   for the same number cannot both succeed; the loser retries and gets the next.
#
# USAGE
#   tools/claim_id.sh roi-tightness   -> creates experiments/h102-roi-tightness
#                                        and prints the path
#   Then write protocol.md into it BEFORE running anything, as usual.
set -e
[ -n "$1" ] || { echo "usage: $0 <slug>   (e.g. roi-tightness)" >&2; exit 2; }
SLUG="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/experiments"
# Highest existing hNNN, from directory names only -- no file needs reading.
N=$(ls "$EXP" 2>/dev/null | sed -n 's/^h\([0-9][0-9]*\)-.*/\1/p' | sort -n | tail -1)
N=${N:-0}
i=$((N + 1))
# Race-safe: mkdir fails if another session claimed it between our scan and now.
while [ $i -lt 1000 ]; do
  if mkdir "$EXP/h$i-$SLUG" 2>/dev/null; then
    mkdir -p "$EXP/h$i-$SLUG/code" "$EXP/h$i-$SLUG/results/ckpt"
    echo "experiments/h$i-$SLUG"
    exit 0
  fi
  # Either this number exists under a different slug, or we lost the race.
  i=$((i + 1))
done
echo "no free id below 1000" >&2; exit 1
