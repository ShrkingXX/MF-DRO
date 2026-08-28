#!/usr/bin/env python
"""Compare two run files' query traces for bit-identity.

WHY THIS EXISTS
  Reproduction controls in this project ask one question: does re-running a
  stored configuration on current code produce the SAME trace? That question has
  been answered three different ways in three experiments, and once it was
  answered with an identity (comparing stored data to itself, h106's Q3). This
  makes the comparison explicit and refuses the degenerate case.

USAGE
  tools/compare_traces.py <stored.json> <rerun.json>

REFUSES when both paths resolve to the same file -- comparing a run to itself
always "passes" and proves nothing.
"""
import sys, os, json
import numpy as np

def load(p):
    q = json.load(open(p))["queries"]
    post = [e for e in q if not e.get("is_init")]
    return (np.array([np.asarray(e["x"], float) for e in post]),
            np.array([float(e["y"]) for e in post]),
            np.array([int(e["fid"]) for e in post]))

def main(a, b):
    if os.path.realpath(a) == os.path.realpath(b):
        print("REFUSED: both arguments resolve to the same file. Comparing a run "
              "to itself is an identity, not a reproduction check.")
        return 2
    xa, ya, fa = load(a); xb, yb, fb = load(b)
    print(f"  stored : {a}\n  rerun  : {b}\n")
    print(f"  post-init queries: {len(ya)} vs {len(yb)}")
    if len(ya) != len(yb):
        print(f"  *** LENGTH DIFFERS by {abs(len(ya)-len(yb))} -- NOT bit-identical ***")
        n = min(len(ya), len(yb))
    else:
        n = len(ya)
    dx = np.abs(xa[:n] - xb[:n]).max() if n else float('nan')
    dy = np.abs(ya[:n] - yb[:n]).max() if n else float('nan')
    df = int((fa[:n] != fb[:n]).sum()) if n else -1
    print(f"  max |dx|          : {dx:.3e}")
    print(f"  max |dy|          : {dy:.3e}")
    print(f"  fidelity mismatches: {df}")
    same = (len(ya) == len(yb)) and dx == 0.0 and dy == 0.0 and df == 0
    print(f"\n  BIT-IDENTICAL: {same}")
    if not same and n:
        i = int(np.argmax(np.abs(ya[:n] - yb[:n])))
        for j in range(n):
            if not np.array_equal(xa[j], xb[j]) or ya[j] != yb[j] or fa[j] != fb[j]:
                print(f"  first divergence at post-init query {j}: "
                      f"fid {fa[j]}->{fb[j]}, y {ya[j]:.6f}->{yb[j]:.6f}")
                break
    return 0 if same else 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
