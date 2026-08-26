import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
# Arms from argv ALWAYS -- the duplicate-job defect (h56/h57/h59) came from a
# launcher that hardcoded its arm list and re-ran jobs already in flight.
ARMS=sys.argv[1:] or ["SOFTPLUS"]
# Borehole first: it is the PRIMARY test (true slope 1.2566 is outside the
# sigmoid's representable range). Hartmann is the CONTROL and carries the
# discriminating prediction -- softplus must be approximately INERT there.
JOBS=[(b,a,s) for b in ("Borehole_8D","Hartmann_6D") for a in ARMS for s in (44,46,48)]
print(f"[launcher] arms={ARMS} jobs={len(JOBS)}",flush=True)
def run(j):
    b,a,s=j; out=os.path.join(H,"..","results",f"{b}__{a}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {a} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,a,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {a} s{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print(r.stdout.strip().splitlines()[-1],flush=True)
with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
