"""h214 smoke: each arm's rollout_mix reaches the batch, use_roi is ON (h149's defect was
that it was OFF), and the random half genuinely draws from the ROI pool -- checked by
comparing the random actions' spread against a uniform draw over the box."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}; ok=True
def run_one(modname,bench,budget=30.0):
    mod=importlib.import_module(modname)
    _oi=MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=budget
    with contextlib.redirect_stdout(io.StringIO()):
        mod.h83.run(bench,"MF-DRO",42,os.path.join(SCR,f"h214_smoke_{modname}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    return CAP["mf"]
print("="*78)
# totals are PER BATCH = per-member spec x M(=3) members. v1 had 180/360 here,
# which was my arithmetic error, not a bad arm: D60 is 20 MES + 60 random per
# member = 240 total, D120 is 20 + 120 = 420 total. The batches were correct.
EXP={"worker_RROI":(0,60),"worker_D60":(60,240),"worker_D120":(60,420)}
for modname,(n_mes,n_tot) in EXP.items():
    mf=run_one(modname,"Borehole_8D")
    with contextlib.redirect_stdout(io.StringIO()): b=mf._generate_rollout_batch()
    pol={p:sum(1 for t in b if t.get('_policy')==p) for p in set(t.get('_policy') for t in b)}
    rnd=[t for t in b if t.get('_policy')=='random']
    lo,hi=mf.bounds[0].numpy(),mf.bounds[1].numpy()
    A=np.vstack([t['actions_x'][:1].numpy() for t in rnd])          # tau=0 actions, already unit-scaled
    spread=float(np.linalg.norm(A.std(axis=0))); ctr=float(np.linalg.norm(A.mean(axis=0)-0.5))
    c=(mf.use_roi and pol.get('mes',0)==n_mes and len(b)==n_tot)
    print(f"{modname[7:]:5s}: use_roi={mf.use_roi} K={mf.inference_context_k} batch {len(b)}={pol}  (want {n_tot}, mes={n_mes})  -> {'ok' if c else 'BAD'}")
    print(f"       random tau=0 actions: spread={spread:.3f}  |mean-0.5|={ctr:.3f}   "
          f"(uniform-over-box would give spread~0.29*sqrt(d)=0.82, |mean-0.5|~0)")
    ok&=c
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*78); sys.exit(0 if ok else 1)
