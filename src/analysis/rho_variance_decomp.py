"""Decompose var_H = rho^2*var_L + var_delta under fitted vs pinned rho.
If delta absorbs the change, var_delta drops when rho rises. If it does not,
the whole rho^2 increase lands on the acquisition as extra uncertainty."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.getcwd())
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
SPEC={"Hartmann_6D":(6,45,0.9792),"Borehole_8D":(10,20,1.2566)}
print(f"  {'benchmark':<13}{'arm':<9}{'rho':>7}{'rho^2*var_L':>13}{'var_delta':>12}{'var_H':>11}")
for bench,(n_hf,n_lf,ts) in SPEC.items():
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    lo=torch.tensor(hf["domain_min"]); hi=torch.tensor(hf["domain_max"]); d=len(lo)
    bnds=torch.stack([lo,hi]); fh=hf["make_objective"](); fl=lf["make_objective"]()
    keep={}
    for arm,pin in (("fitted",None),("RHOTRUE",ts)):
        A=[];B=[];R=[]
        for seed in (44,46,48):
            torch.manual_seed(seed); np.random.seed(seed)
            Xl=lo+(hi-lo)*torch.rand(n_lf,d); Xh=lo+(hi-lo)*torch.rand(n_hf,d)
            Yl=torch.tensor([float(fl(x.reshape(1,-1))) for x in Xl])
            Yh=torch.tensor([float(fh(x.reshape(1,-1))) for x in Xh])
            kw=dict(d=d,dkl_threshold=9999)
            if pin is not None: kw["rho_fixed"]=pin
            ko=KennedyOHaganGP(**kw); ko.bounds=bnds; ko.fit(Xl,Yl,Xh,Yh,bnds)
            Xt=lo+(hi-lo)*torch.rand(400,d); r=float(ko.rho); R.append(r)
            with torch.no_grad():
                vL=ko.gp_lf.posterior(Xt).variance.reshape(-1).numpy()
                vD=ko.gp_delta.posterior(Xt).variance.reshape(-1).numpy()
            A.append(float(np.mean(r*r*vL))); B.append(float(np.mean(vD)))
        keep[arm]=(np.mean(R),np.mean(A),np.mean(B))
        print(f"  {bench:<13}{arm:<9}{np.mean(R):>7.3f}{np.mean(A):>13.2f}{np.mean(B):>12.2f}{np.mean(A)+np.mean(B):>11.2f}")
    f,t=keep["fitted"],keep["RHOTRUE"]
    print(f"  {'':13}{'ratio':<9}{'':>7}{t[1]/max(f[1],1e-12):>12.2f}x{t[2]/max(f[2],1e-12):>11.2f}x"
          f"{(t[1]+t[2])/max(f[1]+f[2],1e-12):>10.2f}x")
    print(f"  {'':13}delta {'ABSORBED the change' if t[2]<f[2]*0.9 else 'did NOT absorb it'}\n")
