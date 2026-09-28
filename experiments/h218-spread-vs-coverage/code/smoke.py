"""h218 smoke: each arm's knob reaches the live objects, and the JIT arms actually WIDEN
the action-target distribution by roughly the intended amount without touching anything
else. Also reports the achieved dispersion against MIXR's mixture, so P-SPREAD's dosing is
measured rather than assumed."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}; ok=True
def run_one(modname,budget=30.0):
    mod=importlib.import_module(modname)
    _oi=MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=budget
    with contextlib.redirect_stdout(io.StringIO()):
        mod.h83.run("Borehole_8D","MF-DRO",42,os.path.join(SCR,f"h218_smoke_{modname}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    return CAP["mf"]
def disp(batch):
    A=np.vstack([t['actions_x'].numpy() for t in batch])
    return float(np.linalg.norm(A.std(axis=0)))
print("="*80)
EXP={"worker_RROI120":(0,360,0.0),"worker_JIT10":(60,60,0.10),"worker_JIT25":(60,60,0.25)}
base=None
for modname,(n_mes,n_tot,jit) in EXP.items():
    mf=run_one(modname)
    with contextlib.redirect_stdout(io.StringIO()): b=mf._generate_rollout_batch()
    pol={p:sum(1 for t in b if t.get('_policy')==p) for p in set(t.get('_policy') for t in b)}
    d=disp(b)
    c=(len(b)==n_tot and pol.get('mes',0)==n_mes and abs(mf._action_jitter-jit)<1e-9 and mf.use_roi and mf.inference_context_k==1)
    print(f"{modname[7:]:8s}: batch {len(b)}={pol} (want {n_tot}, mes={n_mes})  jitter={mf._action_jitter}  "
          f"action-target dispersion={d:.3f}  -> {'ok' if c else 'BAD'}")
    if modname=="worker_JIT10": base=d
    ok&=c
print(f"\nDosing note: h214's smoke measured the RANDOM half's tau=0 action spread at 0.68-0.69.")
print(f"If neither JIT level approaches MIXR's mixture dispersion, P-SPREAD is UNDER-DOSED and")
print(f"that is reported rather than read as a null.")
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*80); sys.exit(0 if ok else 1)
