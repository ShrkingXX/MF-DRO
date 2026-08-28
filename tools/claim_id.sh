#!/bin/sh
# Claim the next free experiment ID atomically. Usage: tools/claim_id.sh <slug>
#
# WHY THIS EXISTS
#   Three ID collisions in one day (h88, h89, h97), all from the same cause: a
#   reservation written inside one session's protocol file is invisible to the
#   other session at the moment it picks a number. Any scheme based on reading a
#   note has a race between "read the note" and "create the directory".
#
#   `mkdir` does not. It is atomic and fails if the target exists.
#
# WHY THE CLAIM IS ON THE NUMBER, NOT ON "hNNN-slug"
#   The first version of this script claimed `experiments/hNNN-$SLUG`. That does
#   NOT close the race, because mkdir only fails on an EXACT name match: two
#   sessions racing for h102 with different slugs both succeed and both get 102.
#   Verified by direct reproduction -- h91-alpha and h91-beta were both created.
#   And different slugs is precisely the real case: every collision we have had
#   was two sessions naming DIFFERENT experiments with the same number.
#
#   So the lock is a slug-independent marker under experiments/.ids/hNNN. The
#   experiment directory is created only after that marker is won.
set -e
[ -n "$1" ] || { echo "usage: $0 <slug>   (e.g. roi-tightness)" >&2; exit 2; }
SLUG="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/experiments"
IDS="$EXP/.ids"
mkdir -p "$IDS"
# Highest number already taken, from BOTH materialised experiments and pending
# claim markers -- so a number claimed but not yet populated is never reused.
A=$(ls "$EXP" 2>/dev/null | sed -n 's/^h\([0-9][0-9]*\)-.*/\1/p' | sort -n | tail -1)
B=$(ls "$IDS" 2>/dev/null | sed -n 's/^h\([0-9][0-9]*\)$/\1/p'   | sort -n | tail -1)
N=$(printf '%s\n%s\n' "${A:-0}" "${B:-0}" | sort -n | tail -1)
i=$((N + 1))
while [ $i -lt 1000 ]; do
  if mkdir "$IDS/h$i" 2>/dev/null; then      # <- the atomic claim, slug-independent
    mkdir -p "$EXP/h$i-$SLUG/code" "$EXP/h$i-$SLUG/results/ckpt"
    echo "experiments/h$i-$SLUG"
    exit 0
  fi
  i=$((i + 1))                                # lost the race, or number in use
done
echo "no free id below 1000" >&2; exit 1
