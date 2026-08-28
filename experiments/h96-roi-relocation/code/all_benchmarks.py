"""H96: does relocation track the ROI's outcome on Ackley and Currin too?
Prediction registered in findings.md (commit f88d5de) BEFORE this was run:
relocation absent on Ackley, possibly NEGATIVE on Currin."""
import os,sys,json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
SEEDS=(42,43,44,45,46)
XSTAR={"Ackley_10D":[0.5]*10,
       "Currin_2D":[0.21666666328646256,0.008707968518137932]}
ROI={"Ackley_10D":os.path.join(REPO,"experiments","h86-roi-full","results"),
     "Currin_2D": os.path.join(REPO,"experiments","h86-roi-full","results")}
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
OUTCOME={"Ackley_10D":"-0.09, 1/5  (negligible)","Currin_2D":"+0.11, 0/5  (HARMED)"}

def sens(bench,LO,HI,obj,d,N=3000):
    rng=np.random.RandomState(0); base=rng.rand(N,d)
    def ev(Xn):
        return np.asarray(obj(torch.as_tensor(LO+(HI-LO)*Xn,dtype=torch.float64)),float).reshape(-1)
    vf=ev(base).var(); sh=[]
    for i in range(d):
        X=base.copy(); X[:,i]=0.5; sh.append(max(0.0,(vf-ev(X).var())/vf))
    sh=np.array(sh); return sh/sh.sum() if sh.sum()>0 else np.ones(d)/d

def qs(p,LO,HI):
    q=json.load(open(p))["queries"]
    post=[e for e in q if not e.get("is_init") and e["fid"]]
    if not post: return None
    return (np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO)

for B in ("Ackley_10D","Currin_2D"):
    hf=get_benchmark(f"{B}_HF"); obj=hf["make_objective"]()
    LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float); d=len(LO)
    XS=(np.asarray(XSTAR[B],float)-LO)/(HI-LO)
    W=sens(B,LO,HI,obj,d)
    print(f"\n=== {B}   ROI regret outcome: {OUTCOME[B]} ===")
    print("  sensitivity: "+"  ".join(f"d{i}:{100*W[i]:.1f}%" for i in np.argsort(-W)[:4])+
          (f"   (top-4 of {d})" if d>4 else ""))
    dw=[]
    print(f"  {'seed':>5}{'ROI w':>10}{'noROI w':>10}{'diff':>9}")
    for s in SEEDS:
        a=os.path.join(ROI[B],f"{B}__ROI-Q10__seed{s}.json"); c=os.path.join(H83,f"{B}__MF-DRO__seed{s}.json")
        if not (os.path.exists(a) and os.path.exists(c)): print(f"  {s:>5}  MISSING"); continue
        A,C=qs(a,LO,HI),qs(c,LO,HI)
        if A is None or C is None: print(f"  {s:>5}  no post-init HF"); continue
        aw=np.sqrt((((A-XS)**2)*W).sum(1)).mean(); cw=np.sqrt((((C-XS)**2)*W).sum(1)).mean()
        dw.append(aw-cw); print(f"  {s:>5}{aw:>10.4f}{cw:>10.4f}{aw-cw:>+9.4f}")
    if dw:
        dw=np.array(dw)
        print(f"  paired mean {dw.mean():+.4f}   ROI closer {int((dw<0).sum())}/{len(dw)}")
        print(f"  -> relocation {'PRESENT' if int((dw<0).sum())>=4 else 'ABSENT'}")
