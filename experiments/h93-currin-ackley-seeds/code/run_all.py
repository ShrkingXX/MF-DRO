"""H93 launcher: MF-DRO vs each benchmark's OWN best baseline, seeds 52-56.

Comparator per benchmark, taken from h83 rather than defaulting to MF-MES (which
wins NEITHER of these -- using it would test an easier question):
    Currin_2D    MF-MI-Greedy  (0.00% at 42-46)
    Ackley_10D   SF-DRO        (3.43  at 42-46)
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
H = os.path.dirname(os.path.abspath(__file__))
PY_ = os.path.join(H, "..", "..", "..", ".venv", "bin", "python")
SEEDS = (52, 53, 54, 55, 56)
BEST = {"Currin_2D": "MF-MI-Greedy", "Ackley_10D": "SF-DRO"}
# MF-DRO arms first: they dominate wall-clock (Currin ~115m, Ackley ~35m), and
# MI-Greedy finishes in under a minute.
JOBS = ([(b, "MF-DRO", s) for b in BEST for s in SEEDS] +
        [(b, m, s) for b, m in BEST.items() for s in SEEDS])
SKIP = {t for t in sys.argv[1:]}
if SKIP:
    JOBS = [j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
    print(f"[launcher] excluding {len(SKIP)} in-flight; {len(JOBS)} remain", flush=True)
def run(j):
    b, m, s = j
    out = os.path.join(H, "..", "results", f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}", flush=True); return
    r = subprocess.run([PY_, "-u", os.path.join(H, "worker.py"), b, m, str(s)],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}", flush=True)
    else:
        print((r.stdout.strip().splitlines() or ["[no output]"])[-1], flush=True)
with ThreadPoolExecutor(max_workers=15) as ex: list(ex.map(run, JOBS))
print("ALL DONE", flush=True)
