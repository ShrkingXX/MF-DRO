"""Complete h84 Hartmann ROI-OFF control at seeds 44, 45, 46.

Slot discipline: the 15-worker cap is GLOBAL and shared with the peer session.
This launcher counts real workers via src/analysis/worker_count.sh before every
spawn and waits rather than assuming a fixed allocation, so it fills in behind
h113 as that drains without ever taking the total past 15.
"""
import os, subprocess, sys, time
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")
CAP = 15
JOBS = [("Hartmann_6D", "ROI-OFF", s) for s in (44, 45, 46)]

def workers():
    try:
        r = subprocess.run(["bash", os.path.join(REPO, "src/analysis/worker_count.sh")],
                           capture_output=True, text=True, timeout=30, cwd=REPO)
        return int(r.stdout.strip().split()[-1])
    except Exception as e:
        print(f"[launcher] worker count failed ({e}); refusing to spawn", flush=True)
        return CAP  # fail closed

procs = []
for b, m, s in JOBS:
    out = os.path.join(H, "..", "results", f"{b}__{m}__seed{s}.json")
    if os.path.exists(out):
        print(f"[skip] {b} {m} {s}", flush=True); continue
    while workers() >= CAP:
        time.sleep(60)
    print(f"[launch] {b} {m} seed{s} (workers before spawn: {workers()})", flush=True)
    procs.append(((b, m, s), subprocess.Popen(
        [PY, "-u", os.path.join(H, "worker.py"), b, m, str(s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)))
    time.sleep(20)  # let the new worker register before the next count

for (b, m, s), p in procs:
    o, e = p.communicate()
    print(f"[FAIL {p.returncode}] {b} {m} seed{s}\n{e.strip()[-800:]}" if p.returncode
          else (o.strip().splitlines() or ["[no output]"])[-1], flush=True)
print("CONTROL ARM DONE", flush=True)
