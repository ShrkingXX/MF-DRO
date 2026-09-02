"""The ONE statistic h181's gate is registered against.

Both arms are measured by THIS code, so the calibrating number and the tested
number cannot silently be different quantities (the h134 failure mode).

  statistic: mean over probed iterations of |x(rtg=0) - x(rtg=1)| in the unit
             box, at the REAL inference state, pooled over seeds.
"""
import json, glob, sys
import numpy as np


def responsiveness(pattern, state="real"):
    per_seed, flips = {}, {}
    for fn in sorted(glob.glob(pattern)):
        seed = int(fn.split("seed")[1].split(".")[0])
        d, ds, fl = json.load(open(fn)), [], []
        for it in d["h168_probe"]:
            by = {}
            for pr in it["probes"]:
                by.setdefault(pr["state"], {})[pr["rtg"]] = (np.array(pr["x"]), pr["ell"])
            if state in by and 0.0 in by[state] and 1.0 in by[state]:
                a, ea = by[state][0.0]
                b, eb = by[state][1.0]
                ds.append(float(np.linalg.norm(a - b)))
                fl.append(ea != eb)
        if ds:
            per_seed[seed] = float(np.mean(ds))
            flips[seed] = float(np.mean(fl))
    return per_seed, flips


if __name__ == "__main__":
    for lab, pat in (("h179 STDCOND   ",
                      "experiments/h179-standardised-conditioning/results/Borehole*.json"),
                     ("h181 PROBECTL  ",
                      "experiments/h181-borehole-probe-control/results/Borehole*.json")):
        ps, fl = responsiveness(pat)
        if not ps:
            print(f"  {lab} -- no results yet")
            continue
        v = np.array(list(ps.values()))
        print(f"  {lab} n={len(ps)}  pooled mean {v.mean():.4f}  "
              f"per-seed {[round(x, 4) for x in v]}  fid-flip {np.mean(list(fl.values())):.2f}")
