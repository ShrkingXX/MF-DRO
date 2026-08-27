import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=(47,48,49,50,51)          # never used in h83/h84/h86
# MF-MES first: it is ~2 min/run and is the comparator every MF-DRO run is
# measured against. A comparator that lands last is a comparator you cannot
# check early (Lesson 21).
JOBS=[("Hartmann_6D","MF-MES",s) for s in SEEDS]+ \
     [("Hartmann_6D","MF-DRO",s) for s in SEEDS]
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
