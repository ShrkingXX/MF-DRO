import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
# Independent of h83's 42-46. MF-DRO first: it dominates wall-clock (83 min vs 5).
JOBS=[("Borehole_8D","MF-DRO",s) for s in (52,53,54,55,56)] + \
     [("Borehole_8D","MF-MES",s) for s in (52,53,54,55,56)]
SKIP={t for t in sys.argv[1:]}
if SKIP:
    JOBS=[j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
    print(f"[launcher] excluding {len(SKIP)} in-flight; {len(JOBS)} remain",flush=True)
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    print((f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}") if r.returncode
          else (r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
# Cap: 4 here + 1 identity-gate worker = 5, alongside 10 for the peer's h113 = 15.
with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
