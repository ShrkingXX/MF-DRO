import os,sys,subprocess,itertools
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
JOBS=[(a,s) for a in ("T","S","R") for s in (42,43,44)]   # T first: it calibrates the measure
def run(j):
    a,s=j
    out=os.path.join(HERE,"..","results",f"{a}__seed{s}.json")
    if os.path.exists(out):
        print(f"[skip] {a} seed{s}",flush=True); return
    print(f"[start] {a} seed{s}",flush=True)
    r=subprocess.run([PY,"-u",os.path.join(HERE,"worker.py"),a,str(s)],
                     capture_output=True,text=True)
    tail="\n".join(r.stdout.strip().splitlines()[-2:])
    print(f"[exit {r.returncode}] {a} seed{s}\n{tail}",flush=True)
    if r.returncode: print(r.stderr.strip()[-1500:],flush=True)
with ThreadPoolExecutor(max_workers=7) as ex:
    list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
