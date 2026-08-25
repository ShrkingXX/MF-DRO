import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(H,"..","..","..",".venv","bin","python")
# Arms may be launched separately; pass them as argv so a second launcher can
# never re-start a job that is still RUNNING (the results-file guard only sees
# FINISHED jobs, which duplicated 3 workers when the MESROI arm was added).
ARMS=sys.argv[1:] or ["ROI","GLOBAL","MESROI"]
JOBS=[(a,s) for s in (44,46,48) for a in ARMS]
def run(j):
    a,s=j; out=os.path.join(H,"..","results",f"{a}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {a} {s}",flush=True); return
    print(f"[start] {a} seed{s}",flush=True)
    r=subprocess.run([PY,"-u",os.path.join(H,"worker.py"),a,str(s)],capture_output=True,text=True)
    print(f"[exit {r.returncode}] {a} seed{s} "+"\n".join(r.stdout.strip().splitlines()[-1:]),flush=True)
    if r.returncode: print(r.stderr.strip()[-1200:],flush=True)
with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
