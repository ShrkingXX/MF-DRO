"""H80: A = Currin held-out; B = synthetic R^2 sweep. Bars in protocol.md."""
import os,sys,json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP

def vdelta_ratio(d,bnds,Xl,Yl,Xh,Yh,Xt,pin):
    out=[]
    for p in (None,pin):
        kw=dict(d=d,dkl_threshold=9999)
        if p is not None: kw["rho_fixed"]=p
        ko=KennedyOHaganGP(**kw); ko.bounds=bnds; ko.fit(Xl,Yl,Xh,Yh,bnds)
        with torch.no_grad(): vD=ko.gp_delta.posterior(Xt).variance.reshape(-1).numpy()
        out.append(float(np.mean(vD)))
    return out[0],out[1]

# ---------- A: Currin held out ----------
print("  A. HELD-OUT BENCHMARK -- Currin (R^2 = 0.99471, never used to build the account)\n")
hf=get_benchmark("Currin_2D_HF"); lf=get_benchmark("Currin_2D_LF")
lo=torch.tensor(hf["domain_min"]); hi=torch.tensor(hf["domain_max"]); d=len(lo)
bnds=torch.stack([lo,hi]); fh=hf["make_objective"](); fl=lf["make_objective"]()
rs=[]
for seed in (44,46,48):
    torch.manual_seed(seed); np.random.seed(seed)
    Xl=lo+(hi-lo)*torch.rand(15,d); Xh=lo+(hi-lo)*torch.rand(5,d)
    Yl=torch.tensor([float(fl(x.reshape(1,-1))) for x in Xl])
    Yh=torch.tensor([float(fh(x.reshape(1,-1))) for x in Xh])
    Xt=lo+(hi-lo)*torch.rand(400,d)
    a,b=vdelta_ratio(d,bnds,Xl,Yl,Xh,Yh,Xt,1.0104); rs.append(b/a)
cur=float(np.mean(rs))
print(f"  Currin var_delta ratio (RHOTRUE/fitted) = {cur:.3f}   per seed {[round(x,3) for x in rs]}")
print(f"  reference: Borehole 0.72 (R^2 1.00000) | Hartmann 1.24 (R^2 0.85574)\n")

# ---------- B: synthetic R^2 sweep ----------
print("  B. SYNTHETIC R^2 SWEEP -- f_H = s*f_L + a*g(x), a swept\n")
print(f"  {'a':>6}{'R^2':>10}{'OLS slope':>11}{'var_delta ratio':>18}{'':>4}")
d=4; lo=torch.zeros(d); hi=torch.ones(d); bnds=torch.stack([lo,hi])
fL=lambda X: torch.sin(3*X[:,0])+0.5*X[:,1]**2+0.3*X[:,2]
g =lambda X: torch.cos(7*X[:,1])*torch.sin(5*X[:,3])      # fixed nonlinear perturbation
rows=[]
for a in (0.0,0.05,0.10,0.20,0.35,0.55,0.80):
    torch.manual_seed(0); np.random.seed(0)
    Xr=torch.rand(4096,d); yl=fL(Xr); yh=1.15*yl+a*g(Xr)
    s,b0=np.polyfit(yl.numpy(),yh.numpy(),1)
    r2=1-np.var(yh.numpy()-(s*yl.numpy()+b0))/np.var(yh.numpy())
    rr=[]
    for seed in (0,1,2):
        torch.manual_seed(seed); np.random.seed(seed)
        Xl=torch.rand(30,d); Xh=torch.rand(12,d); Xt=torch.rand(400,d)
        Yl=fL(Xl); Yh=1.15*fL(Xh)+a*g(Xh)
        p,q=vdelta_ratio(d,bnds,Xl,Yl,Xh,Yh,Xt,float(s)); rr.append(q/p)
    rows.append((a,r2,float(s),float(np.mean(rr))))
    print(f"  {a:>6.2f}{r2:>10.5f}{s:>11.4f}{np.mean(rr):>18.3f}{'  <- crosses 1.0' if np.mean(rr)>1.0 and (not rows[:-1] or rows[-2][3]<=1.0) else ''}")
json.dump(dict(currin_ratio=cur,sweep=rows),
          open(os.path.join(os.path.dirname(__file__),"..","results","h80.json"),"w"))
print("\n  LOCKED PREDICTIONS")
print(f"  1 PRIMARY(A)   Currin ratio < 1.0: {cur:.3f} -> {'MET' if cur<1.0 else 'NOT MET'}")
print(f"  2 SECONDARY(A) ratio < 0.95: {cur:.3f} -> {'MET' if cur<0.95 else 'NOT MET'}")
rt=[r[3] for r in rows]
mono=all(rt[i+1]>=rt[i]-0.05 for i in range(len(rt)-1))
cross=any(r>1.0 for r in rt) and rt[0]<=1.0
print(f"  3 PRIMARY(B)   monotone as R^2 falls AND crosses 1.0: mono={mono}, cross={cross} -> "
      f"{'MET' if mono and cross else 'NOT MET'}")
print(f"  4 NULL         Currin>1.0 or no trend -> "
      f"{'FIRED -- account does not generalise' if (cur>1.0 or not mono) else 'no'}")
