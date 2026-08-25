"""H62's primary locked prediction: the re-run must reproduce h59's SF-DRO
regret to within 1e-9. The trace fix touched only post-run_optimization
bookkeeping (reading dro.data_x/data_y instead of a key that never existed), so
a divergence means the fix perturbed the run and h59's SF-DRO numbers become
suspect.

Reports the gate result FIRST and unconditionally. Only if it passes are the
traces declared usable for freeze_watch / stall_diagnose.
"""
import os, sys, json, glob
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
H59 = os.path.join(REPO, "experiments", "h59-sfdro-baseline", "results")
H62 = os.path.join(REPO, "experiments", "h62-sfdro-traces", "results")
TOL = 1e-9

def main():
    rows, missing = [], []
    for b in ("Currin_2D", "Hartmann_6D", "Borehole_8D"):
        for s in (44, 46, 48):
            f59 = os.path.join(H59, f"{b}__SF-DRO__seed{s}.json")
            f62 = os.path.join(H62, f"{b}__SF-DRO__seed{s}.json")
            if not (os.path.exists(f59) and os.path.exists(f62)):
                missing.append((b, s)); continue
            a, c = json.load(open(f59)), json.load(open(f62))
            rows.append((b, s, a["final_regret"], c["final_regret"],
                         len(c.get("query_x") or []), c.get("n_queries", 0)))
    if missing:
        print(f"  INCOMPLETE — {len(missing)} cell(s) missing: {missing}")
        print("  Gate not evaluated. Nothing reported.")
        return 2
    print(f"  {'benchmark':<13}{'seed':>5}{'h59 regret':>14}{'h62 regret':>14}{'|diff|':>12}{'trace n':>9}")
    bad = 0
    for b, s, r59, r62, nq, nq2 in rows:
        d = abs(r59 - r62)
        if d > TOL: bad += 1
        print(f"  {b:<13}{s:>5}{r59:>14.9f}{r62:>14.9f}{d:>12.2e}{nq:>9}"
              + ("   <-- DIVERGES" if d > TOL else ""))
    print()
    if bad:
        print(f"  GATE FAILED: {bad}/{len(rows)} cells diverge beyond {TOL:g}.")
        print("  h59's SF-DRO regret numbers are now SUSPECT. Report the divergence;")
        print("  do NOT silently adopt h62's numbers in their place.")
        return 1
    ntr = sum(1 for r in rows if r[4] > 0)
    print(f"  GATE PASSED: all {len(rows)} cells match within {TOL:g}.")
    print(f"  Traces present in {ntr}/{len(rows)} cells -> usable for freeze_watch / stall_diagnose.")
    print("  h59's regret conclusions stand unchanged; H62 adds instrumentation only.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
