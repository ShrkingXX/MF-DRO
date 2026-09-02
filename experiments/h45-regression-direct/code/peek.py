import os,json,glob
import numpy as np
from scipy.stats import wilcoxon
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
R2=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
rows=[json.load(open(f)) for f in glob.glob(os.path.join(R,"reg__seed*.json"))]
if not rows: print("no completed jobs yet"); raise SystemExit
rows.sort(key=lambda r:r["seed"])
print(f"ARM B (regression, no pool/argmax): {len(rows)}/10 done\n")
for r in rows:
    print(f"  seed{r['seed']}  regret={r['final_regret']:.4f}  impr={r['n_improvements']}  "
          f"iters={r['n_iters']}  nHF={r['n_hf']}  spread={r['x_spread']:.4f}")
b=np.array([r["final_regret"] for r in rows]); seeds=[r["seed"] for r in rows]
def load(d,m):
    o={}
    for s in seeds:
        p=os.path.join(R2,d,"results",f"{m}__seed{s}.json")
        if os.path.exists(p):
            j=json.load(open(p)); c=j.get("hf_regret_curve") or j.get("regret_curve")
            o[s]=float(c[-1])
    return o
A=load("h17-joint-mes-frozen-eval","MF-DRO"); C=load("h31-teacher-without-dt","MF-DRO")
a=np.array([A[s] for s in seeds if s in A]); c=np.array([C[s] for s in seeds if s in C])
se=lambda x:x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else float('nan')
print(f"\n{'arm':<34}{'mean regret':>13}")
print(f"  A  candidate scoring (h17)      {a.mean():>13.4f}" if len(a) else "")
print(f"  B  REGRESSION direct (this)     {b.mean():>13.4f}")
print(f"  C  MF-MES teacher (h31)         {c.mean():>13.4f}" if len(c) else "")
if len(b)>2 and len(c)==len(b):
    d=b-c
    print(f"\nPRED 1 (B >= C, i.e. direct proposal does NOT beat the teacher):")
    print(f"   B-C = {d.mean():+.4f}   B better on {(d<0).sum()}/{len(d)}   "
          f"Wilcoxon p={wilcoxon(b,c).pvalue:.4f}")
    print(f"   -> {'HOLDS (B no better than C)' if d.mean()>=0 else 'B beats C -- hypothesis challenged'}")
if len(b)>2 and len(a)==len(b):
    d2=b-a
    print(f"PRED 2 (A beats B, machinery does the work):")
    print(f"   B-A = {d2.mean():+.4f}   A better on {(d2>0).sum()}/{len(d2)}   "
          f"Wilcoxon p={wilcoxon(b,a).pvalue:.4f}")
