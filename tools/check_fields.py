#!/usr/bin/env python3
"""Verify a protocol's required result fields EXIST before the protocol is locked.

WHY THIS EXISTS
  Five registered predictions in this project were written against data the
  pipeline does not serialise, and each was discovered only after runs were spent
  or an analysis refused:

    h140  actions_x_var_per_iter   appended in memory, never written
    h139  per-record roi_stats     worker collapses it to means (5 runs spent)
    h145  SC5 states[0]            per-trajectory states not serialised
    h145  SC8 teacher actions      not serialised
    h148  realised rtg[0]          only rtg_target + aggregate fractions exist

  Every one was avoidable in two seconds by opening a result file first. Writing
  the rule down did not work -- it was written down after the second instance and
  three more followed.

USAGE
  tools/check_fields.py <result.json> field [field ...]
  tools/check_fields.py experiments/h83-main-comparison/results/Borehole_8D__MF-DRO__seed42.json rtg_target neg_rtg_frac_per_iter
"""
import json, sys, os

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, fields = sys.argv[1], sys.argv[2:]
    if not os.path.exists(path):
        sys.exit(f"MISSING RESULT FILE: {path}")
    r = json.load(open(path))
    print(f"== {os.path.basename(path)} ==")
    bad = []
    for f in fields:
        if f in r:
            v = r[f]
            kind = type(v).__name__
            extra = (f"len={len(v)}" if isinstance(v, (list, dict)) else str(v)[:40])
            print(f"  PRESENT  {f:34s} {kind:6s} {extra}")
        else:
            print(f"  ABSENT   {f:34s} <- protocol cannot be evaluated on this field")
            bad.append(f)
    if bad:
        print(f"\n  {len(bad)} field(s) ABSENT. Do NOT lock a protocol that depends on them:")
        print(f"    either serialise them first, or write the prediction against what exists.")
        print(f"  Available top-level keys ({len(r)}):")
        for k in sorted(r):
            print(f"    {k}")
    else:
        print("\n  ALL PRESENT -- the protocol can be evaluated on this run.")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
