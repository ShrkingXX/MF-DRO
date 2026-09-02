"""h167 -- where does the collapse land? Centroid of the DT's own real queries
against the computable mean of each teacher's action distribution."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark

BENCH, SEEDS = "Borehole_8D", [42, 43, 44, 45, 46]
XSTAR_RAW = np.array([0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0])
ARMS = {
 "control (works)":       "experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json",
 "h155 UCB-LOC (works)":  "experiments/h155-ucb-loc/results/{b}__UCB-LOC__seed{s}.json",
 "h153 FROZEN (works)":   "experiments/h153-mes-frozen/results/{b}__MES-FROZEN__seed{s}.json",
 "h159 EXPLOIT (works)":  "experiments/h159-exploit-loc/results/{b}__EXPLOIT-LOC__seed{s}.json",
 "ORACLE (fails)":        "experiments/h145-oracle-expert-ceiling/results/{b}__ORACLE-EXPERT__seed{s}.json",
 "DIVERSE-GOOD (fails)":  "experiments/h146-why-oracle-hurts/results/{b}__DIVERSE-GOOD__seed{s}.json",
 "RANDOM-POOL (fails)":   "experiments/h149-forced-vs-teacher-quality/results/{b}__RANDOM-POOL__seed{s}.json",
}
hf = get_benchmark(f"{BENCH}_HF")
lo = np.array(hf["domain_min"], float); hi = np.array(hf["domain_max"], float)
d = len(lo)
CENTRE = np.full(d, 0.5)                       # domain centre, normalised
XSTAR = (XSTAR_RAW - lo) / (hi - lo)           # x*, normalised
MID = (CENTRE + XSTAR) / 2                     # ORACLE's predicted mean

print(f"\n{BENCH}, normalised coordinates, d={d}")
print(f"  domain centre        {np.round(CENTRE,3)}")
print(f"  x*                   {np.round(XSTAR,3)}")
print(f"  midpoint(centre,x*)  {np.round(MID,3)}   |centre - mid| = {np.linalg.norm(CENTRE-MID):.4f}\n")
print(f"  {'arm':24s} {'d(centroid, CENTRE)':>20s} {'d(centroid, MIDPOINT)':>22s} {'nearer':>10s}")
out = {}
for name, pat in ARMS.items():
    dc, dm = [], []
    for s in SEEDS:
        p = os.path.join(REPO, pat.format(b=BENCH, s=s))
        if not os.path.exists(p): continue
        X = (np.asarray(json.load(open(p))["x_t_trace"], float) - lo) / (hi - lo)
        c = X.mean(axis=0)
        dc.append(float(np.linalg.norm(c - CENTRE))); dm.append(float(np.linalg.norm(c - MID)))
    if not dc: print(f"  {name:24s} {'NO DATA':>20s}"); continue
    a, b = float(np.mean(dc)), float(np.mean(dm))
    out[name] = dict(d_centre=a, d_mid=b, d_centre_per_seed=[round(v,4) for v in dc])
    print(f"  {name:24s} {a:20.4f} {b:22.4f} {('CENTRE' if a<b else 'MIDPOINT'):>10s}")

W = [k for k in out if "works" in k]; F = [k for k in out if "fails" in k]
r = out.get("RANDOM-POOL (fails)"); o = out.get("ORACLE (fails)")
print("\n  --- P1: RANDOM-POOL nearest the centre of all arms? ---")
if r:
    others = {k: v["d_centre"] for k, v in out.items() if k != "RANDOM-POOL (fails)"}
    mn = min(others.values()); who = min(others, key=others.get)
    print(f"    RANDOM {r['d_centre']:.4f} vs nearest other {mn:.4f} ({who})"
          f"  -> P1 {'HOLDS' if r['d_centre'] < mn else 'FAILS'}")
print("  --- P2: ORACLE nearer MIDPOINT than CENTRE? ---")
if o:
    print(f"    ORACLE d(centre) {o['d_centre']:.4f}  d(midpoint) {o['d_mid']:.4f}"
          f"  -> P2 {'HOLDS' if o['d_mid'] < o['d_centre'] else 'FAILS'}")
print("  --- P3: working arms away from both? ---")
wc = np.mean([out[k]["d_centre"] for k in W]); fc = np.mean([out[k]["d_centre"] for k in F])
print(f"    working mean d(centre) {wc:.4f}   failing mean d(centre) {fc:.4f}"
      f"  -> P3 {'HOLDS' if wc > fc else 'FAILS'}")
json.dump(out, open(os.path.join(REPO, "experiments/h167-collapse-target/results/centroids.json"), "w"), indent=1)
