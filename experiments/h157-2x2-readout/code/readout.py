"""h157 2x2 readout. Committed BEFORE h153/h155 produced any result."""
import os, sys, json, importlib.util
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); os.chdir(REPO)
def load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
a83 = load("a83", "experiments/h83-main-comparison/code/analyse.py")
from benchmarks import get_benchmark

BENCH, SEEDS = "Borehole_8D", [42, 43, 44, 45, 46]
OPT = float(get_benchmark(f"{BENCH}_HF")["known_optimal_value"]); A = abs(OPT)
G = np.linspace(0, 200, 201)
ARMS = {
 "control MES (closed)":  "experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json",
 "h153 MES-FROZEN (open)":"experiments/h153-mes-frozen/results/{b}__MES-FROZEN__seed{s}.json",
 "h155 UCB-LOC (closed)": "experiments/h155-ucb-loc/results/{b}__UCB-LOC__seed{s}.json",
 "ORACLE (open)":         "experiments/h145-oracle-expert-ceiling/results/{b}__ORACLE-EXPERT__seed{s}.json",
 "RANDOM-POOL (open)":    "experiments/h149-forced-vs-teacher-quality/results/{b}__RANDOM-POOL__seed{s}.json",
}
def stats(p):
    r = json.load(open(p))
    c, s = a83.sr_curve(r, OPT); rel = a83.grid(c, s, G)[200] / A * 100
    q = r["queries"]
    bi = max([e["y"] for e in q if e.get("is_init") and e["fid"] == 1], default=-1e18)
    po = [e["y"] for e in q if not e.get("is_init") and e["fid"] == 1]
    imp = (max(po) if po else -1e18) > bi + 1e-9
    hf = np.mean([e["fid"] for e in q if not e.get("is_init")])
    return rel, imp, float(np.mean(r.get("rtg_target", [np.nan]))), hf, r

print(f"\n{'arm':26s} {'rel%':>7s} {'improves':>9s} {'rtg_target':>11s} {'HF frac':>8s}  n")
rows = {}
for name, pat in ARMS.items():
    v = [stats(pat.format(b=BENCH, s=s)) for s in SEEDS if os.path.exists(pat.format(b=BENCH, s=s))]
    if not v: print(f"{name:26s} {'-- not finished --':>40s}"); continue
    rel = np.mean([x[0] for x in v]); imp = sum(x[1] for x in v)
    rtg = np.nanmean([x[2] for x in v]); hf = np.mean([x[3] for x in v])
    rows[name] = (rel, imp, rtg, hf, len(v))
    print(f"{name:26s} {rel:7.2f} {imp:6d}/{len(v)} {rtg:11.4f} {hf:8.2f}  {len(v)}")

if "h153 MES-FROZEN (open)" in rows:
    r0 = json.load(open(ARMS["h153 MES-FROZEN (open)"].format(b=BENCH, s=SEEDS[0]))).get("_h153", {})
    print(f"\nh153 SANITY: SC1 path err={r0.get('sc1_path_max_abs_err')}  "
          f"SC2 open-loop penalty={r0.get('sc2_open_loop_penalty')}  "
          f"SC3 fidelity flip frac={r0.get('sc3_ell_flip_frac')}  rollouts={r0.get('n_rollouts')}")
if "h155 UCB-LOC (closed)" in rows and "control MES (closed)" in rows:
    h, c = rows["h155 UCB-LOC (closed)"][3], rows["control MES (closed)"][3]
    print(f"\nh155 CONFOUND CHECK: HF fraction {h:.2f} vs control {c:.2f} -> "
          f"{'OK' if abs(h - c) < 0.25 else 'COLLAPSED -- ARM CONFOUNDED, NO VERDICT'}")
print("\nFORECASTS (registered before these landed):")
print("  h153 rtg_target 0.83-0.94 (85-96% of control), rel% NEAR CONTROL not 43.94")
print("  h155 see h157 C6 forecast")
