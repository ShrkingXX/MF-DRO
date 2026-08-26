"""Is an incumbent stall recoverable, and does stall length predict regret?

Stall is counted in HF-OPPORTUNITY units: LF queries do NOT count toward a stall.

WHY THIS MATTERS -- counting stalls over ALL queries is CONFOUNDED. A run with 3
HF queries in 179 (Hartmann MF-DRO seed 46) is forced into a huge stall by
construction, purely from LF interleaving. Counted that way MF-DRO looks like a
uniquely bad staller (terminal 50.0 vs MF-MES 22.4); counted in HF units it ties
MI-Greedy exactly (34% vs 34%). The all-query version was written first in this
file and was WRONG. Do not reintroduce it.

CONCLUSION (h57, n=3/cell): terminal stall fraction does NOT track regret.
MI-Greedy has the LOWEST terminal stall on Hartmann (2%, still climbing at
budget exhaustion) and the WORST regret there. Stall length is therefore not an
actionable trigger for an adaptive intervention -- it does not separate 'stuck'
from 'converged'."""
import json, glob, os
from collections import defaultdict
import numpy as np
RES = "experiments/h57-baseline-comparison/results"

def hf_stalls(path):
    d = json.load(open(path))
    q = [e for e in d["queries"] if not e.get("is_init")]
    hf = [e["y"] for e in q if e["fid"] == 1]
    best, run, broken = -np.inf, 0, []
    for y in hf:
        if y > best:
            best = y
            if run: broken.append(run)
            run = 0
        else: run += 1
    return broken, run, len(hf), d["final_regret"]

rows = defaultdict(list)
for p in sorted(glob.glob(f"{RES}/*.json")):
    b = os.path.basename(p)[:-5]
    if b.count("__") != 2: continue
    bench, arm, seed = b.split("__")
    rows[(bench, arm)].append(hf_stalls(p))

print("  Stall in HF-OPPORTUNITY units (LF queries excluded)   n=3/cell, descriptive\n")
print(f"  {'benchmark':<13}{'arm':<14}{'nHF':>6}{'brk':>5}{'medbrk':>8}{'maxbrk':>8}{'TERM':>7}{'term/nHF':>10}")
byarm = defaultdict(lambda: ([], [], []))
for (bench, arm), rs in sorted(rows.items()):
    allb = [x for r in rs for x in r[0]]
    term = [r[1] for r in rs]; nhf = [r[2] for r in rs]
    frac = np.mean([t/n if n else np.nan for t, n in zip(term, nhf)])
    print(f"  {bench:<13}{arm:<14}{np.mean(nhf):>6.0f}{len(allb):>5}"
          f"{(np.median(allb) if allb else 0):>8.0f}{(max(allb) if allb else 0):>8.0f}"
          f"{np.mean(term):>7.1f}{frac:>10.0%}")
    byarm[arm][0].extend(allb); byarm[arm][1].extend(term); byarm[arm][2].extend(nhf)

print("\n  Pooled over benchmarks (MF-GP-UCB excluded: structurally all-LF, 0 HF queries):")
for arm, (b, t, n) in sorted(byarm.items()):
    if arm == "MF-GP-UCB": continue
    print(f"    {arm:<14} nHF/run {np.mean(n):>5.1f}  recoveries {len(b):>3}  "
          f"median {np.median(b) if b else 0:>4.1f}  max {max(b) if b else 0:>3.0f}   "
          f"|  terminal {np.mean(t):>5.1f} = {np.mean(t)/np.mean(n):>4.0%} of HF budget")
print(f"\n    MF-GP-UCB      nHF/run {np.mean(byarm['MF-GP-UCB'][2]):>5.1f}  (no HF queries at all -- excluded above)")
