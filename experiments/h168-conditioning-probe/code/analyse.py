"""h168 -- does the emitted action depend on the conditioning RTG?"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

ARM = sys.argv[1] if len(sys.argv) > 1 else "RANDOM"
BENCH, SEEDS = "Hartmann_6D", [42, 43, 44, 45, 46]
PAT = f"experiments/h168-conditioning-probe/results/{BENCH}__PROBE-{ARM}__seed%d.json"

allrows, tgt = {}, []
for s in SEEDS:
    p = os.path.join(REPO, PAT % s)
    if not os.path.exists(p): continue
    P = json.load(open(p)).get("h168_probe", [])
    for e in P:
        tgt.append(e["rtg_target"])
        for pr in e["probes"]:
            allrows.setdefault(pr["rtg"], []).append(
                float(np.linalg.norm(np.array(pr["x"]) - 0.5)))
if not allrows:
    raise SystemExit(f"no probe data for {ARM}")

rt = np.array(tgt)
print(f"\n  {BENCH}  arm={ARM}   {len(rt)} probed iterations across {len(SEEDS)} seeds")
print(f"  realised rtg_target: mean {rt.mean():.4f}  range {rt.min():.3f}-{rt.max():.3f}\n")
print(f"  {'conditioning RTG':>17s} {'mean d(box centre)':>19s} {'sd':>8s} {'n':>6s}")
ks = sorted(allrows)
for k in ks:
    v = np.array(allrows[k])
    print(f"  {k:17.2f} {v.mean():19.4f} {v.std():8.4f} {len(v):6d}")

lo_k = min(ks, key=lambda k: abs(k - 0.02))      # in-support for a failing arm
hi_k = min(ks, key=lambda k: abs(k - rt.mean())) # the real target
lo, hi = np.mean(allrows[lo_k]), np.mean(allrows[hi_k])
print(f"\n  P1: at the REAL target (rtg={hi_k:.2f})     d(centre) = {hi:.4f}")
print(f"      near the box centre would be d < ~0.15  -> P1 {'HOLDS' if hi < 0.15 else 'FAILS'}")
print(f"  P2: at in-support (rtg={lo_k:.2f})           d(centre) = {lo:.4f}")
print(f"      ratio in-support / target = {lo/hi:.3f}   needs >= 2.0  -> P2 {'HOLDS' if lo/hi >= 2.0 else 'FAILS'}")
sw = np.array([np.mean(allrows[k]) for k in ks])
print(f"\n  spread across the WHOLE sweep: {sw.max()-sw.min():.4f} "
      f"({(sw.max()-sw.min())/sw.mean()*100:.1f}% of the mean)")
print(f"  -> the emitted action is {'ESSENTIALLY INDEPENDENT of' if (sw.max()-sw.min())/sw.mean() < 0.15 else 'SENSITIVE to'} the conditioning")
