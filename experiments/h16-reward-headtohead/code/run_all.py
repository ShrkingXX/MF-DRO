"""H16 driver: 20 jobs, 15 workers x 1 thread (compute rule)."""
import os, sys, json, subprocess, itertools
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), "..", "..", ".venv", "bin", "python")
SEEDS = list(range(42, 52))
REWARDS = ["improvement", "mes_entropy"]

def job(a):
    s, r = a
    out = subprocess.run([PY, os.path.join(HERE, "worker.py"), str(s), r],
                         capture_output=True, text=True)
    for line in reversed(out.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    sys.stderr.write(f"FAIL seed={s} {r}\n{out.stdout[-800:]}{out.stderr[-800:]}\n")
    return None

if __name__ == "__main__":
    jobs = list(itertools.product(SEEDS, REWARDS))
    with ThreadPoolExecutor(max_workers=15) as ex:
        res = [r for r in ex.map(job, jobs) if r]
    json.dump(res, open(os.path.join(HERE, "..", "results", "h16.json"), "w"),
              indent=2, default=float)
    print(f"{len(res)}/{len(jobs)} jobs done")
