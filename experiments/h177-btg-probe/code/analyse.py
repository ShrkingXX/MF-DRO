"""h177 -- does the emitted action depend on the BTG conditioning?"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
PAT = "experiments/h177-btg-probe/results/Hartmann_6D__BTG-RANDOM__seed%d.json"
byb, byr, real_btg, real_rtg = {}, {}, [], []
for s in range(42, 47):
    p = os.path.join(REPO, PAT % s)
    if not os.path.exists(p): continue
    for e in json.load(open(p)).get("h168_probe", []):
        real_btg.append(e.get("btg_now", np.nan)); real_rtg.append(e["rtg_target"])
        for pr in e.get("btg_probes", []):
            byb.setdefault(pr["btg"], []).append(float(np.linalg.norm(np.array(pr["x"]) - 0.5)))
        for pr in e.get("probes", []):
            byr.setdefault(pr["rtg"], []).append(float(np.linalg.norm(np.array(pr["x"]) - 0.5)))
rb = np.array(real_btg)
print(f"\n  Hartmann RANDOM-POOL, {len(real_btg)} probed iterations, 5 seeds")
print(f"  realised btg_now: mean {np.nanmean(rb):.3f}  range {np.nanmin(rb):.2f}-{np.nanmax(rb):.2f}\n")
print(f"  {'BTG':>7s} {'mean d(box centre)':>19s} {'sd':>8s} {'n':>6s}")
ks = sorted(byb)
for k in ks:
    v = np.array(byb[k]); print(f"  {k:7.1f} {v.mean():19.4f} {v.std():8.4f} {len(v):6d}")
m = np.array([np.mean(byb[k]) for k in ks])
print(f"\n  spread across the BTG sweep : {m.max()-m.min():.4f}  ({(m.max()-m.min())/m.mean()*100:.1f}% of mean)")
kr = sorted(byr); mr = np.array([np.mean(byr[k]) for k in kr])
print(f"  spread across the RTG sweep : {mr.max()-mr.min():.4f}  ({(mr.max()-mr.min())/mr.mean()*100:.1f}% of mean)   [h168 full sweep: 8.9%]")
print(f"\n  P1 (BTG inert, movement under ~15% of mean): "
      f"{'HOLDS' if (m.max()-m.min())/m.mean() < 0.15 else 'FAILS'}")
