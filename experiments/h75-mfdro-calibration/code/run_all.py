import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
METHODS=sys.argv[1:] or ["MF-DRO"]        # methods ALWAYS from argv
SEEDS=[42,43,45,47,49,50,51]              # 44/46/48 reused from h57
JOBS=[("Borehole_8D",m,s) for m in METHODS for s in SEEDS]
print(f"[launcher] methods={METHODS} seeds={SEEDS} jobs={len(JOBS)}",flush=True)
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {m} s{s}\n{r.stderr.strip()[-700:]}",flush=True)
    else: print((r.stdout.strip().splitlines() or ["(no output)"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=7) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
