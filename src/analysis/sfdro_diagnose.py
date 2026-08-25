"""Diagnose SF-DRO runs from h62's traces (single-fidelity: every query is HF).

Separates the same two failure modes as freeze_watch, and adds the value-based
check from lesson 23 -- where a query sits in the domain's VALUE distribution,
not its geometry.
"""
import os, sys, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "experiments", "h62-sfdro-traces", "results")

def main():
    from benchmarks import get_benchmark
    from src.analysis.value_reference import pct
    print(f"  {'benchmark':<13}{'seed':>5}{'nq':>5}{'distinct':>9}{'QFREEZE':>9}"
          f"{'improv':>7}{'stall':>7}{'regret':>10}{'q pctile':>10}{'inc pctile':>12}")
    for b in ("Currin_2D", "Hartmann_6D", "Borehole_8D"):
        h = get_benchmark(f"{b}_HF")
        lo = np.array(h["domain_min"], float); hi = np.array(h["domain_max"], float)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        for s in (44, 46, 48):
            f = os.path.join(RES, f"{b}__SF-DRO__seed{s}.json")
            if not os.path.exists(f): continue
            d = json.load(open(f))
            X = np.array(d["query_x"], float); Y = np.array(d["query_y"], float)
            if X.size == 0: continue
            U = (X - lo) / span
            distinct = len({tuple(np.round(u, 9)) for u in U})
            best, imp, last = -np.inf, 0, -1
            for i, y in enumerate(Y):
                if y > best: best, imp, last = y, imp + 1, i
            stall = len(Y) - 1 - last
            qp = float(np.mean([pct(b, v) for v in Y[-40:]]))
            ip = pct(b, float(Y.max()))
            print(f"  {b:<13}{s:>5}{len(Y):>5}{distinct:>9}"
                  f"{('YES' if distinct < len(Y) else 'no'):>9}{imp:>7}{stall:>7}"
                  f"{d['final_regret']:>10.4f}{qp:>10.1%}{ip:>12.1%}")
    print("\n  Single fidelity: every query is HF, so the incumbent can move on any of them.")
    print("  q pctile = mean percentile of the LAST 40 query values vs a 20k Sobol reference.")

if __name__ == "__main__":
    main()
