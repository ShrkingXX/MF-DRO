"""Freeze watch for MF-DRO runs. Reads h57 finals AND live checkpoints.

Distinguishes the two failure modes this project has confused before:
  QUERY FREEZE      distinct < n_queries -- the same point proposed repeatedly.
  INCUMBENT STALL   distinct == n_queries but improvements stop -- every
                    proposal fresh, none ever beating the incumbent. This is
                    what h45 seeds 49/50 did (144/144 and 74/74 distinct, ZERO
                    improvements) and it is NOT a freeze.
Spread is normalised by the domain span; Borehole's raw span is ~52530 on one
axis, so an unnormalised spread is meaningless across benchmarks.
"""
import sys, os, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "experiments", "h57-baseline-comparison", "results")

def _bounds(bench):
    from benchmarks import get_benchmark
    b = get_benchmark(f"{bench}_HF")
    return np.array(b["domain_min"], float), np.array(b["domain_max"], float)

def scan(path, live, method="MF-DRO"):
    d = json.load(open(path))
    m = d.get("_meta") or d
    if m.get("method") != method:
        return None
    q = [e for e in d.get("queries", []) if not e.get("is_init")]
    if not q:
        return None
    lo, hi = _bounds(m["bench"]); span = np.where(hi - lo > 0, hi - lo, 1.0)
    X = (np.array([e["x"] for e in q], float) - lo) / span
    F = np.array([e["fid"] for e in q])
    distinct = len({tuple(np.round(x, 9)) for x in X})
    best, imp, last_imp = -np.inf, 0, -1
    for i, e in enumerate(q):
        if e["fid"] == 1 and e["y"] > best:
            best, imp, last_imp = e["y"], imp + 1, i
    Z = X[-40:]
    spread = float(np.mean([np.linalg.norm(Z[i] - Z[j])
                            for i in range(len(Z)) for j in range(i + 1, len(Z))])) \
        if len(Z) > 1 else float("nan")
    stall = (len(q) - 1 - last_imp) if last_imp >= 0 else len(q)
    return dict(bench=m["bench"], seed=m["seed"], n=len(q), distinct=distinct,
                query_freeze=distinct < len(q), improv=imp, spread=spread,
                since_last_improv=stall, live=live,
                nhf=int((F == 1).sum()), nlf=int((F == 0).sum()))

def main(method="MF-DRO"):
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(RES, f"*{method}*.json"))):
        r = scan(f, False, method)
        if r: rows.append(r); seen.add((r["bench"], r["seed"]))
    for f in sorted(glob.glob(os.path.join(RES, "ckpt", f"*{method}*.json"))):
        r = scan(f, True, method)
        if r and (r["bench"], r["seed"]) not in seen: rows.append(r)
    if not rows:
        print(f"  no {method} queries recorded yet"); return
    rows.sort(key=lambda r: (r["bench"], r["seed"]))
    print(f"  {'benchmark':<13}{'seed':>5}{'src':>7}{'nq':>5}{'distinct':>9}"
          f"{'QFREEZE':>9}{'improv':>7}{'stall':>7}{'spread':>8}{'HF/LF':>9}")
    for r in rows:
        print(f"  {r['bench']:<13}{r['seed']:>5}{'live' if r['live'] else 'final':>7}"
              f"{r['n']:>5}{r['distinct']:>9}{'YES' if r['query_freeze'] else 'no':>9}"
              f"{r['improv']:>7}{r['since_last_improv']:>7}{r['spread']:>8.4f}"
              f"{str(r['nhf'])+'/'+str(r['nlf']):>9}")
    print("\n  QFREEZE = same point re-proposed.  stall = queries since the last "
          "incumbent improvement.\n  A large stall with QFREEZE=no is the h45 "
          "seeds-49/50 mode: all-distinct, never improving.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MF-DRO")
