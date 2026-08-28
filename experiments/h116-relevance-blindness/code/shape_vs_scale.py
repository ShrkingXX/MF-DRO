import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import json, itertools, numpy as np
CK="experiments/h83-main-comparison/results/ckpt"; SEEDS=[42,43,44,45,46]
def prof(b,m,s):
    d=json.load(open(f"{CK}/{b}__{m}__seed{s}.json"))
    X=np.array([q["x"] for q in d["queries"] if q["fid"]==1 and not q.get("is_init",False)],float)
    sd=X.std(axis=0,ddof=1); return sd/sd.sum(), sd.sum()
print(f"{'bench':13s} {'L1 DRO-vs-MES':>14s} {'L1 within-MES':>14s} {'L1 within-DRO':>14s} {'ratio btw/within':>17s} {'scale DRO/MES':>14s}")
for b in ["Borehole_8D","Hartmann_6D","Ackley_10D"]:
    P={m:{s:prof(b,m,s) for s in SEEDS} for m in ["MF-DRO","MF-MES"]}
    btw=[np.abs(P["MF-DRO"][s][0]-P["MF-MES"][s][0]).sum() for s in SEEDS]          # seed-matched
    wM =[np.abs(P["MF-MES"][a][0]-P["MF-MES"][c][0]).sum() for a,c in itertools.combinations(SEEDS,2)]
    wD =[np.abs(P["MF-DRO"][a][0]-P["MF-DRO"][c][0]).sum() for a,c in itertools.combinations(SEEDS,2)]
    within=np.mean(wM+wD)
    scale=np.mean([P["MF-DRO"][s][1]/P["MF-MES"][s][1] for s in SEEDS])
    print(f"{b:13s} {np.mean(btw):>14.3f} {np.mean(wM):>14.3f} {np.mean(wD):>14.3f} {np.mean(btw)/within:>17.2f} {scale:>14.2f}")
