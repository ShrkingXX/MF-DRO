import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=(42,43,44,45,46); BENCH=("Borehole_8D","Hartmann_6D")
ARMS=("REFINE-100","HF-FLOOR")
JOBS=[(b,a,s) for a in ARMS for b in BENCH for s in SEEDS]
JOBS+=[(b,"REFINE-0",s) for b in BENCH for s in (42,43)]   # reproduction control
SKIP={t for t in (sys.argv[1:] if len(sys.argv)>1 else [])}
if SKIP:
    JOBS=[j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
    print(f"[launcher] excluding {len(SKIP)}; {len(JOBS)} remain",flush=True)
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=15) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
