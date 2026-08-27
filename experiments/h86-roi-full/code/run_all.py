import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
# Only the two benchmarks the ROI has never been tested on. h84 already ran
# Hartmann and Borehole under this exact configuration at these seeds.
JOBS=[(b,"ROI-Q10",s) for b in ("Ackley_10D","Currin_2D") for s in (42,43,44,45,46)]
SKIP={t for t in (sys.argv[1:] if len(sys.argv)>1 else [])}
if SKIP: JOBS=[j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=10) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
