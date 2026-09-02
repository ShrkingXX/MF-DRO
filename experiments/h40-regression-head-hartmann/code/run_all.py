import os,sys,json,subprocess,itertools
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
JOBS=[(a,s) for s in range(42,52) for a in ("cs","reg")]  # interleaved
def job(a):
    arm,s=a
    # skip work already checkpointed (makes the grid resumable)
    done=os.path.join(HERE,"..","results",f"{arm}__seed{s}.json")
    if os.path.exists(done):
        return json.load(open(done))
    o=subprocess.run([PY,os.path.join(HERE,"worker.py"),arm,str(s)],capture_output=True,text=True)
    for l in reversed(o.stdout.strip().splitlines()):
        if l.startswith("{"): return json.loads(l)
    sys.stderr.write(f"FAIL {a}\n{o.stdout[-400:]}{o.stderr[-500:]}\n"); return None
# 4 workers. Two silent kills already: 0-byte log, no traceback = external
# termination, not a Python exception. Swap was at 3.18/4.00 GB (78%) when the
# 8-worker run died after 3 jobs. Raw "free RAM" was misleading here; swap is the
# real signal. Slower, but the per-job checkpoints make it resumable either way.
if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=4) as ex:
        res=[r for r in ex.map(job,JOBS) if r]
    json.dump(res,open(os.path.join(HERE,"..","results","h40.json"),"w"),indent=2)
    print(f"{len(res)}/{len(JOBS)} done")
