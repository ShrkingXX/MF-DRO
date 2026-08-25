import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
PY_ = os.path.join(HERE, "..", "..", "..", ".venv", "bin", "python")
SEEDS = [49, 50, 42, 44]      # FAIL, FAIL, PASS, PASS
def job(s):
    out = os.path.join(HERE, "..", "results", f"mc__seed{s}.json")
    if os.path.exists(out):
        print(f"[skip] seed{s}", flush=True); return s
    o = subprocess.run([PY_, os.path.join(HERE, "worker.py"), str(s)],
                       capture_output=True, text=True)
    print(o.stdout.strip() or o.stderr.strip()[-800:], flush=True)
    return s if o.returncode == 0 else None
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = [r for r in ex.map(job, SEEDS) if r]
    print(f"{len(res)}/{len(SEEDS)} complete", flush=True)
