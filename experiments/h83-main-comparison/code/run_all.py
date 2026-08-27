import os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
SEEDS=(42,43,44,45,46)
BENCH=("Hartmann_6D","Ackley_10D","Borehole_8D","Currin_2D")
# Longest-first: the DRO arms dominate wall-clock, Hartmann most of all.
METHODS=("MF-DRO","SF-DRO","MF-MES","MF-MI-Greedy","MF-GP-UCB")
JOBS=[(b,m,s) for m in METHODS for b in BENCH for s in SEEDS]
# A second launcher must never restart a job that is still RUNNING: the
# results-file guard only sees FINISHED jobs, which duplicated three workers on
# h56 when an arm was added mid-flight. Pass "bench:method:seed" triples to skip.
SKIP={t for t in (sys.argv[1:] if len(sys.argv)>1 else [])}
if SKIP:
    JOBS=[j for j in JOBS if f"{j[0]}:{j[1]}:{j[2]}" not in SKIP]
    print(f"[launcher] excluding {len(SKIP)} in-flight jobs; {len(JOBS)} remain",flush=True)
def run(j):
    b,m,s=j; out=os.path.join(H,"..","results",f"{b}__{m}__seed{s}.json")
    if os.path.exists(out): print(f"[skip] {b} {m} {s}",flush=True); return
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),b,m,str(s)],
                     capture_output=True,text=True)
    if r.returncode:
        print(f"[FAIL {r.returncode}] {b} {m} seed{s}\n{r.stderr.strip()[-900:]}",flush=True)
    else:
        print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
with ThreadPoolExecutor(max_workers=15) as ex: list(ex.map(run,JOBS))
print("ALL DONE",flush=True)
