import os,sys,json,subprocess,itertools
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
PY=os.path.join(HERE,"..","..","..",".venv","bin","python")
CELLS=list(itertools.product([2,4,8,16],[5,10,50]))
def job(c):
    r,K=c
    o=subprocess.run([PY,os.path.join(HERE,"worker.py"),str(r),str(K)],capture_output=True,text=True)
    for l in reversed(o.stdout.strip().splitlines()):
        if l.startswith("{"): return json.loads(l)
    sys.stderr.write(f"FAIL {c}\n{o.stdout[-400:]}{o.stderr[-400:]}\n"); return None
if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=12) as ex:
        res=[r for r in ex.map(job,CELLS) if r]
    json.dump(res,open(os.path.join(HERE,"..","results","h29.json"),"w"),indent=2)
    print(f"{len(res)}/12 cells done")
