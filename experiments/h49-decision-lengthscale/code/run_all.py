"""DRIVER -- long-lived parent. This is the launch pattern that survives
(h17/h31/h29 all completed this way); bare `nohup worker.py &` does not."""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
RES=os.path.join(HERE,"..","results")
SEEDS=list(range(42,52))
def job(s):
    ck=os.path.join(RES,f"nat__seed{s}.json")
    if os.path.exists(ck):
        print(f"[skip] seed{s} already checkpointed",flush=True); return json.load(open(ck))
    o=subprocess.run([PY,os.path.join(HERE,"worker.py"),str(s)],capture_output=True,text=True)
    print(o.stdout.strip()[-200:] or f"[FAIL] seed{s}",flush=True)
    if o.returncode!=0: sys.stderr.write(f"FAIL seed{s}\n{o.stderr[-600:]}\n")
    return json.load(open(ck)) if os.path.exists(ck) else None
if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=10) as ex:
        res=[r for r in ex.map(job,SEEDS) if r]
    print(f"{len(res)}/{len(SEEDS)} complete",flush=True)
