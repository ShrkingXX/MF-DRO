import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(H,"..","..","..",".venv","bin","python")
# Arms come from argv. The results-file guard only skips FINISHED jobs, so a
# second launcher with the full list restarts everything still RUNNING -- that
# happened here and produced 9 duplicate workers (20 total against a 15 cap).
ARMS=sys.argv[1:] or ["SF-DRO"]
JOBS=[(b,a,s) for b in ("Borehole_8D","Currin_2D","Hartmann_6D")
              for a in ARMS for s in (44,46,48)]
print(f"[launcher] arms={ARMS}  jobs={len(JOBS)}",flush=True)
def run(j):
    b,a,s=j; out=os.path.join(H,"..","results",f"{b}__{a}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {a} {s}",flush=True); return
    r=subprocess.run([PY,"-u",os.path.join(H,"worker.py"),b,a,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {a} s{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print(r.stdout.strip().splitlines()[-1],flush=True)
with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
