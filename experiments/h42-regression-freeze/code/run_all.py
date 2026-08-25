import os,sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
def job(s):
    done=os.path.join(HERE,"..","results",f"reg__seed{s}.json")
    if os.path.exists(done): return json.load(open(done))
    o=subprocess.run([PY,os.path.join(HERE,"worker.py"),str(s)],capture_output=True,text=True)
    for l in reversed(o.stdout.strip().splitlines()):
        if l.startswith("{"): return json.loads(l)
    sys.stderr.write(f"FAIL seed{s}\n{o.stdout[-500:]}{o.stderr[-800:]}\n"); return None
if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        res=[r for r in ex.map(job,[42,43,44]) if r]
    print(f"{len(res)}/3 done")
