"""H7 analysis: decision agreement between the live DT and the iteration-5 snapshot."""
import json, glob, os, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows = []
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    d = json.load(open(f))
    log = d.get("decision_divergence_log", [])
    if not log:
        continue
    ag = np.array([x["argmax_agree"] for x in log], dtype=float)
    di = np.array([x["dist"] for x in log], dtype=float)
    fa = np.array([x["fid_agree"] for x in log], dtype=float)
    rows.append((d["seed"], len(log), ag.mean(), di.mean(), fa.mean(), ag, di))
    print(f"seed{d['seed']}: n_decisions={len(log):4d}  argmax_agree={ag.mean():.3f}  "
          f"mean_dist={di.mean():.4f}  fid_agree={fa.mean():.3f}")
if rows:
    allag = np.concatenate([r[5] for r in rows]); alldi = np.concatenate([r[6] for r in rows])
    print(f"\nPOOLED over {len(rows)} seeds, {len(allag)} paired decisions")
    print(f"  argmax agreement = {allag.mean():.4f}")
    print(f"  mean L2 distance = {alldi.mean():.4f}")
    print(f"\nLOCKED PREDICTION 1: mean argmax_agree > 0.70  ->  "
          f"{'PASS' if allag.mean() > 0.70 else 'FAIL'} ({allag.mean():.3f})")
    # prediction 2: no progressive divergence
    per = [r[6] for r in rows]
    slopes = [np.polyfit(np.arange(len(x)), x, 1)[0] for x in per if len(x) > 3]
    print(f"LOCKED PREDICTION 2: dist does not grow with t  ->  "
          f"mean slope {np.mean(slopes):+.6f}/iter "
          f"({'PASS, flat' if abs(np.mean(slopes)) < 1e-3 else 'FAIL, drifting'})")
