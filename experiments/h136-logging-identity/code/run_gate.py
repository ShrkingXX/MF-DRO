"""H136 gate: Ackley ROI-Q10 seed42 on the patched tree vs h86's stored trace.
Waits for a free slot (global cap 15, shared) before spawning."""
import os, subprocess, sys, time, json
H=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(H,"..","..",".."))
PY=os.path.join(REPO,".venv","bin","python"); CAP=15
def workers():
    try:
        r=subprocess.run(["bash",os.path.join(REPO,"src/analysis/worker_count.sh")],
                         capture_output=True,text=True,timeout=30,cwd=REPO)
        return int(r.stdout.strip().split()[-1])
    except Exception as e:
        print(f"[gate] worker count failed ({e}); refusing to spawn",flush=True); return CAP
out=os.path.join(H,"..","results","Ackley_10D__ROI-Q10__seed42.json")
if not os.path.exists(out):
    while workers()>=CAP:
        time.sleep(60)
    print(f"[gate] launching (workers before spawn: {workers()})",flush=True)
    r=subprocess.run([PY,"-u",os.path.join(H,"worker.py"),"Ackley_10D","ROI-Q10","42"],
                     capture_output=True,text=True,cwd=REPO)
    if r.returncode: print(f"[gate FAIL rc={r.returncode}]\n{r.stderr[-1200:]}",flush=True); sys.exit(2)
    print((r.stdout.strip().splitlines() or ["[no output]"])[-1],flush=True)
a=json.load(open(out)); b=json.load(open(os.path.join(REPO,"experiments/h86-roi-full/results/Ackley_10D__ROI-Q10__seed42.json")))
qa,qb=a["queries"],b["queries"]
if len(qa)!=len(qb):
    print(f"[GATE FAIL] length {len(qa)} vs {len(qb)}"); sys.exit(1)
bad=[i for i,(u,v) in enumerate(zip(qa,qb)) if u["fid"]!=v["fid"] or u["x"]!=v["x"] or u["y"]!=v["y"]]
print(f"[GATE {'PASS' if not bad else 'FAIL'}] {len(qa)} queries, {len(bad)} differing"
      + (f"; first at i={bad[0]}" if bad else ""))
rs=a.get("roi_summary") or {}
print(f"  roi_summary n_records={rs.get('n_records')}  (h86 stored: {(b.get('roi_summary') or {}).get('n_records')})")
sys.exit(0 if not bad else 1)
