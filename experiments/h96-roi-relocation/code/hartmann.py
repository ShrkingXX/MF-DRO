"""H96 Amendment 1: is the relocation account Hartmann-transferable?
H3 first (sensitivity profile MEASURED, not assumed), then weighted distance."""
import os,sys,json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Hartmann_6D"; SEEDS=(42,43,44,45,46)
hf=get_benchmark(f"{B}_HF"); obj=hf["make_objective"]()
LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float); d=len(LO)
XSTAR=np.array([0.2017,0.15,0.4769,0.2753,0.3116,0.6573])
XS=(XSTAR-LO)/(HI-LO)

def evals(Xn):
    X=LO+(HI-LO)*Xn
    import torch
    out=obj(torch.as_tensor(X,dtype=torch.float64))
    return np.asarray(out,dtype=float).reshape(-1)

rng=np.random.RandomState(0); N=3000
base=rng.rand(N,d)
v_full=evals(base).var()
print("H3 -- Hartmann sensitivity profile (midpoint-freezing, 3000 samples)\n")
shares=[]
for i in range(d):
    Xf=base.copy(); Xf[:,i]=0.5
    shares.append(max(0.0,(v_full-evals(Xf).var())/v_full))
shares=np.array(shares); W=shares/shares.sum()
for i in range(d): print(f"    dim {i}: {100*W[i]:5.1f}%   x*={XS[i]:.3f}")
print(f"\n    max share {100*W.max():.1f}%  (Borehole's dim0 was 81.6%)")
print(f"    top-2 share {100*np.sort(W)[-2:].sum():.1f}%  -> weighted and unweighted",
      "NEARLY COINCIDE" if W.max()<0.40 else "DIVERGE")

def load(p):
    q=json.load(open(p))["queries"]
    post=[e for e in q if not e.get("is_init") and e["fid"]]
    if not post: return None
    X=(np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO)
    return X
H84=os.path.join(REPO,"experiments","h84-roi-strategy","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
def ctrl(s):
    p=os.path.join(H84,f"{B}__ROI-OFF__seed{s}.json")
    return p if os.path.exists(p) else os.path.join(H83,f"{B}__MF-DRO__seed{s}.json")
print("\nH1 -- weighted distance to x*, ROI-Q10 vs no-ROI, Hartmann seeds 42-46\n")
print(f"    {'seed':>5}{'ROI w':>10}{'noROI w':>10}{'diff':>9}{'ROI u':>10}{'noROI u':>10}{'diff':>9}")
dw=[];du=[]
for s in SEEDS:
    A=load(os.path.join(H84,f"{B}__ROI-Q10__seed{s}.json")); C=load(ctrl(s))
    if A is None or C is None: print(f"    {s:>5}  MISSING"); continue
    aw=np.sqrt((((A-XS)**2)*W).sum(1)).mean(); cw=np.sqrt((((C-XS)**2)*W).sum(1)).mean()
    au=np.linalg.norm(A-XS,axis=1).mean();     cu=np.linalg.norm(C-XS,axis=1).mean()
    dw.append(aw-cw); du.append(au-cu)
    print(f"    {s:>5}{aw:>10.4f}{cw:>10.4f}{aw-cw:>+9.4f}{au:>10.4f}{cu:>10.4f}{au-cu:>+9.4f}")
dw=np.array(dw); du=np.array(du)
print(f"\n    weighted   paired mean {dw.mean():+.4f}   ROI closer {int((dw<0).sum())}/{len(dw)}")
print(f"    unweighted paired mean {du.mean():+.4f}   ROI closer {int((du<0).sum())}/{len(du)}")
reloc = int((dw<0).sum())>=4
print(f"\n  H1 (relocation ABSENT on Hartmann, i.e. NOT closer on >=4/5): "
      f"{'MET -- no relocation where the ROI failed' if not reloc else 'FAILED'}")
if reloc:
    print("\n  *** H2 APPLIES. Relocation IS present on Hartmann while the regret")
    print("      result FAILED and was withdrawn. Relocation is therefore NOT")
    print("      SUFFICIENT, and H96's account is a BOREHOLE-SPECIFIC DESCRIPTION,")
    print("      not a mechanism. Same shape as the dispersion story. ***")
