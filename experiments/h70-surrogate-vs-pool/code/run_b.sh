#!/bin/zsh
cd "$(dirname "$0")/../../.."
for b in Hartmann_6D Borehole_8D; do
  for a in SF-EI ALTGP; do
    for s in 42 43 44 45 46 47 48 49 50 51; do
      f="experiments/h70-surrogate-vs-pool/results/${b}__${a}__seed${s}.json"
      [[ -f "$f" ]] && continue
      .venv/bin/python experiments/h70-surrogate-vs-pool/code/worker.py "$b" "$a" "$s" 2>/dev/null | tail -1
    done
  done
done
