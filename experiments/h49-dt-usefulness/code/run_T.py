import os,subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
def run(s):
    out=os.path.join(HERE,"..","results",f"T__seed{s}.json")
    if os.path.exists(out): print(f"[skip] T seed{s}",flush=True); return
    print(f"[start] T seed{s}",flush=True)
    r=subprocess.run([PY,"-u",os.path.join(HERE,"worker.py"),"T",str(s)],
                     capture_output=True,text=True)
    print(f"[exit {r.returncode}] T seed{s}\n"+"\n".join(r.stdout.strip().splitlines()[-1:]),flush=True)
    if r.returncode: print(r.stderr.strip()[-1200:],flush=True)
with ThreadPoolExecutor(max_workers=2) as ex: list(ex.map(run,(42,43,44)))
print("T DONE",flush=True)
