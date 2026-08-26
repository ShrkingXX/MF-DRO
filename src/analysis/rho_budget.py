"""Is the binding constraint the LINK RANGE or the OPTIMIZER BUDGET?

fit() takes ONE Adam step on rho per round, 3 rounds per call, lr=0.1 from
rho_init=0.8. initial_lengthscale is None here, so log_rho is NOT re-anchored
between calls -- repeated fit() calls accumulate Adam steps on rho.

If rho is optimizer-limited, both links climb with more steps and sigmoid
saturates near 1.0 while softplus passes it. If rho is range-limited, sigmoid
should already be pinned near 1.0 -- it is not (0.84)."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.getcwd())
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from src.models.ko_gp import KennedyOHaganGP
TRUE_RHO=1.2566
d=2
print(f"  target rho = {TRUE_RHO}   (rho_init=0.8, 3 Adam steps per fit() call)\n")
print(f"  {'fit calls':>10}{'= adam steps':>14}{'sigmoid rho':>14}{'softplus rho':>15}")
for link in ("sigmoid","softplus"):
    torch.manual_seed(0); np.random.seed(0)
    X_lf=torch.rand(40,d); X_hf=torch.rand(18,d)
    f=lambda X:(torch.sin(3*X[:,0])+0.5*X[:,1]**2)
    Y_lf=f(X_lf); Y_hf=TRUE_RHO*f(X_hf)+0.02*torch.cos(5*X_hf[:,0])
    bnds=torch.stack([torch.zeros(d),torch.ones(d)])
    ko=KennedyOHaganGP(d=d,dkl_threshold=9999,rho_link=link)
    ko.bounds=bnds
    traj={}
    for c in range(1,31):
        ko.fit(X_lf,Y_lf,X_hf,Y_hf,bnds)
        traj[c]=float(ko.rho)
    globals()[link]=traj
for c in (1,2,3,5,10,20,30):
    print(f"  {c:>10}{c*3:>14}{sigmoid[c]:>14.4f}{softplus[c]:>15.4f}")
print(f"\n  sigmoid  ceiling is 1.0 -- reached {max(sigmoid.values()):.4f} after 90 Adam steps")
print(f"  softplus has no ceiling -- reached {max(softplus.values()):.4f}")
