import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
JOBS=[("Borehole_8D","MF-MES",s) for s in (52,53,54,55,56)]
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=5) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
