import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=(44,46,48)
# Longest-first: MF-DRO dominates wall-clock, Hartmann most of all.
ORDER=[("Hartmann_6D","MF-DRO"),("Borehole_8D","MF-DRO"),("Currin_2D","MF-DRO"),
       ("Hartmann_6D","MF-MES"),("Borehole_8D","MF-MES"),("Currin_2D","MF-MES"),
       ("Hartmann_6D","MF-MI-Greedy"),("Borehole_8D","MF-MI-Greedy"),("Currin_2D","MF-MI-Greedy"),
       ("Hartmann_6D","MF-GP-UCB"),("Borehole_8D","MF-GP-UCB"),("Currin_2D","MF-GP-UCB")]
JOBS=[(b,m,s) for (b,m) in ORDER for s in SEEDS]
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY,"-u",os.path.join(H,"worker.py"),b,m,str(s)],
                     capture_output=True,text=True)
    if r.returncode:
        print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else:
        print(r.stdout.strip().splitlines()[-1],flush=True)
with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
