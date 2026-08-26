"""Does the sigmoid ceiling bind on REAL benchmark data? h67's premise says yes
for Borehole (true slope 1.2566). Fit the actual KO model on real LF/HF samples
at h57's initial-design sizes and read the fitted rho under both links."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.getcwd())
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP

SPEC={"Borehole_8D":(10,20,1.2566),"Hartmann_6D":(6,45,0.9792),"Currin_2D":(5,15,1.0104)}
print(f"  {'benchmark':<13}{'true slope':>11}{'sigmoid rho':>13}{'softplus rho':>14}{'ceiling binds?':>16}")
for bench,(n_hf,n_lf,true_slope) in SPEC.items():
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    lo=torch.tensor(hf["domain_min"],dtype=torch.float64); hi=torch.tensor(hf["domain_max"],dtype=torch.float64)
    d=len(lo); bnds=torch.stack([lo,hi]); fh=hf["make_objective"](); fl=lf["make_objective"]()
    out={}
    for link in ("sigmoid","softplus"):
        vals=[]
        for seed in (44,46,48):
            torch.manual_seed(seed); np.random.seed(seed)
            Xl=lo+(hi-lo)*torch.rand(n_lf,d); Xh=lo+(hi-lo)*torch.rand(n_hf,d)
            Yl=torch.tensor([float(fl(x.reshape(1,-1))) for x in Xl])
            Yh=torch.tensor([float(fh(x.reshape(1,-1))) for x in Xh])
            ko=KennedyOHaganGP(d=d,dkl_threshold=9999,rho_link=link); ko.bounds=bnds
            ko.fit(Xl,Yl,Xh,Yh,bnds); vals.append(float(ko.rho))
        out[link]=np.mean(vals)
    binds = "YES" if out["sigmoid"]>0.97 else f"no ({out['sigmoid']:.3f} << 1.0)"
    print(f"  {bench:<13}{true_slope:>11.4f}{out['sigmoid']:>13.4f}{out['softplus']:>14.4f}{binds:>16}")
