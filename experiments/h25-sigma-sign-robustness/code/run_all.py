import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
def job(s):
    o=subprocess.run([PY,os.path.join(HERE,"worker.py"),str(s)],capture_output=True,text=True)
    for l in reversed(o.stdout.strip().splitlines()):
        if l.startswith("{"): return json.loads(l)
    sys.stderr.write(f"FAIL seed {s}\n{o.stdout[-500:]}{o.stderr[-500:]}\n"); return None
if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=10) as ex:
        res=[r for r in ex.map(job,range(42,52)) if r]
    json.dump(res,open(os.path.join(HERE,"..","results","h25.json"),"w"),indent=2)
    print(f"{len(res)}/10 done")
