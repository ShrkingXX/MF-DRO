import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
METHODS=sys.argv[1:] or ["MF-MI-Greedy","MF-GP-UCB","SF-MES","SF-EI"]
JOBS=[(b,m,s) for b in ("Currin_2D","Hartmann_6D","Borehole_8D")
      for m in METHODS for s in range(42,52)]
print(f"[launcher] methods={METHODS} jobs={len(JOBS)}",flush=True)
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL] {b} {m} s{s}: {r.stderr.strip()[-300:]}",flush=True)
with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
