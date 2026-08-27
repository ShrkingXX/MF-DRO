import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=(47,48,49,50,51)          # never used for Borehole
# INTERLEAVED BY SEED so that PAIRS complete together. A paired comparison needs
# both arms of a seed; running arm A's five seeds first would give five useless
# half-pairs before any usable data. This is the paired-experiment form of
# Lesson 21 -- order the queue so partial results are interpretable.
JOBS=[("Borehole_8D",arm,s) for s in SEEDS for arm in ("NO-ROI","ROI-Q10")]
SKIP={t for t in (sys.argv[1:] if len(sys.argv)>1 else [])}
if SKIP: JOBS=[j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],capture_output=True,text=True)
    if r.returncode: print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else: print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=10) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
