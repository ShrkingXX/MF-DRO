"""h221 Stage 0 -- THE GATE IS THE R-SQUARED ITSELF.

The defect: adjusted R^2 of rtg[0] on the FIRST ACTION (x0, ell0), within ensemble member,
n=100 rollouts, measured at +0.007 (MES) and -0.045 (random). The arms are pointless if the
interventions do not repair that, so Stage 0 re-measures it under each setting rather than
assuming the flags help.

SC1 (GATE): CRN raises adjusted R^2 above 0.10 on >= 2 of 3 members.
SC2:        the advantage baseline leaves the WITHIN-MEMBER ORDERING of rtg[0] unchanged.
SC3:        identity gate -- run separately."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}
mod=importlib.import_module("worker_BOTH")
_oi=MF.DirectMFRegretOptimization.__init__
def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=40.0
with contextlib.redirect_stdout(io.StringIO()):
    mod.h83.run("Borehole_8D","MF-DRO",42,os.path.join(SCR,"h221_smoke.json"))
MF.DirectMFRegretOptimization.__init__=_oi
mf=CAP["mf"]; M=len(mf.ko_ensemble)
def adj_r2(g):
    X=np.stack([np.concatenate([t['actions_x'][0].numpy(),[float(t['actions_ell'][0])],[1.0]]) for t in g])
    y=np.array([float(t['rtg'][0]) for t in g])
    if y.std()<1e-12: return float('nan')
    b,*_=np.linalg.lstsq(X,y,rcond=None); yh=X@b
    r=1-((y-yh)**2).sum()/((y-y.mean())**2).sum(); n,p=len(g),X.shape[1]
    return 1-(1-r)*(n-1)/(n-p)
def batch(crn,adv,per=100):
    mf.config.fantasy_crn=crn; mf.config.rtg_advantage=adv
    mf.config.rollout_mix=[('mes',per,None)]
    with contextlib.redirect_stdout(io.StringIO()): b=mf._generate_rollout_batch()
    mf.config.rollout_mix=None
    return [b[m*per:(m+1)*per] for m in range(M)]
print("="*84)
base=batch(False,False); crn=batch(True,False); adv=batch(False,True); both=batch(True,True)
for nm,g in (("baseline (as measured)",base),("CRN",crn),("ADV",adv),("CRN+ADV",both)):
    v=[adj_r2(x) for x in g]
    print(f"  adj R^2 of rtg[0] on (x0,ell0)   {nm:22s} {[round(x,3) for x in v]}   mean {np.nanmean(v):+.3f}")
vc=[adj_r2(x) for x in crn]
ok1=sum(1 for x in vc if x>0.10)>=2
print(f"\nSC1 (GATE) CRN raises adj R^2 above 0.10 on >=2 of 3 members -> {'PASS' if ok1 else 'FAIL'}")
# SC2: advantage is a constant shift => ordering preserved
ok2=True
for gb,ga in zip(base,adv):
    rb=np.array([float(t['rtg'][0]) for t in gb]); ra=np.array([float(t['rtg'][0]) for t in ga])
    if len(rb)==len(ra):
        from scipy.stats import spearmanr
        rho=spearmanr(rb,ra).correlation
        ok2 &= (rho>0.99 if rho==rho else False)
print(f"SC2 advantage preserves within-member ordering (Spearman > 0.99) -> {'PASS' if ok2 else 'FAIL/NA'}")
print(f"SC3 identity gate: run separately via tools/identity_gate.py")
ok=ok1
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*84); sys.exit(0 if ok else 1)
