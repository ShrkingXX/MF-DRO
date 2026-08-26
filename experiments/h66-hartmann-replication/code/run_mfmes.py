import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=[42,43,45,47,49,50,51]          # 44/46/48 reused from h57
print(f"[launcher] MF-MES seeds={SEEDS}",flush=True)
def run(s):
    out=os.path.join(H,"..","results",f"Hartmann_6D__MF-MES__seed{s}.json")
    if os.path.exists(out): print(f"[skip] MF-MES {s}",flush=True); return
    r=subprocess.run([PY,"-u",os.path.join(H,"worker_mfmes.py"),"Hartmann_6D","MF-MES",str(s)],
                     capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] MF-MES s{s}\n{r.stderr.strip()[-800:]}",flush=True)
    else: print(r.stdout.strip().splitlines()[-1],flush=True)
with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(run,SEEDS))
print("MF-MES DONE",flush=True)
