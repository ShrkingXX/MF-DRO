import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import json, glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "tools")
from perdim import shares

CKPT = "experiments/h83-main-comparison/results/ckpt"
BENCH = ["Borehole_8D", "Hartmann_6D", "Currin_2D", "Ackley_10D"]
METH = ["MF-DRO", "MF-MES"]
SEEDS = [42, 43, 44, 45, 46]

def profile(X, robust):
    if robust:
        med = np.median(X, axis=0)
        s = np.median(np.abs(X - med), axis=0)
    else:
        s = X.std(axis=0, ddof=1)
    tot = s.sum()
    return s / tot if tot > 0 else np.full_like(s, np.nan)

out = {}
for b in BENCH:
    sh = np.asarray(shares(b), float)
    out[b] = {"shares": sh.tolist(), "d": len(sh)}
    for m in METH:
        for robust in (False, True):
            key = f"{m}|{'mad' if robust else 'sd'}"
            rows = []
            for sd_ in SEEDS:
                f = f"{CKPT}/{b}__{m}__seed{sd_}.json"
                if not os.path.exists(f):
                    rows.append({"seed": sd_, "rho": None, "n": 0, "err": "missing"}); continue
                d = json.load(open(f))
                X = np.array([q["x"] for q in d["queries"]
                              if q["fid"] == 1 and not q.get("is_init", False)], float)
                if X.shape[0] < 3:
                    rows.append({"seed": sd_, "rho": None, "n": int(X.shape[0]), "err": "too_few"}); continue
                p = profile(X, robust)
                r = spearmanr(p, sh).statistic
                rows.append({"seed": sd_, "rho": (None if not np.isfinite(r) else float(r)),
                             "n": int(X.shape[0]), "profile": p.tolist()})
            out[b][key] = rows
json.dump(out, open("experiments/h116-relevance-blindness/results/rho.json", "w"), indent=1)

# ---- report ----
DEGEN = {"Currin_2D": "d=2, Spearman in {-1,+1} only", "Ackley_10D": "S1 near-uniform (PR/d=1.000)"}
for stat in ("sd", "mad"):
    print(f"\n===== dispersion profile via {stat.upper()} =====")
    print(f"{'bench':13s} {'rho MF-DRO':>18s} {'rho MF-MES':>18s} {'paired MES-DRO':>16s} {'|mean|/sd':>10s}  note")
    for b in BENCH:
        g = {}
        for m in METH:
            g[m] = [r["rho"] for r in out[b][f"{m}|{stat}"]]
        pairs = [(a, c) for a, c in zip(g["MF-DRO"], g["MF-MES"]) if a is not None and c is not None]
        if not pairs:
            print(f"{b:13s} {'no data':>18s}"); continue
        dro = np.array([a for a, _ in pairs]); mes = np.array([c for _, c in pairs])
        diff = mes - dro
        eff = abs(diff.mean()) / diff.std(ddof=1) if len(diff) > 1 and diff.std(ddof=1) > 0 else float("nan")
        note = DEGEN.get(b, "PRIMARY" if b == "Borehole_8D" else "secondary")
        print(f"{b:13s} {np.median(dro):>8.3f} (sd{dro.std(ddof=1):.2f}) {np.median(mes):>8.3f} (sd{mes.std(ddof=1):.2f})"
              f" {diff.mean():>+10.3f}     {eff:>9.2f}  {note}  n_pairs={len(pairs)}")
