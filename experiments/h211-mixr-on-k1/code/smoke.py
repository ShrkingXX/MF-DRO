"""h211 Stage 0 smoke: each arm builds and runs (BUDGET=30) and the flags that DEFINE it
are set on the constructed objects. A silent default is the failure mode."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp")
import src.policy.mf_dro as MF
CAP={}; ok=True
def run_one(modname, bench):
    mod = importlib.import_module(modname)
    _oi = MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=30.0
    with contextlib.redirect_stdout(io.StringIO()):
        r = mod.h83.run(bench,"MF-DRO",42,os.path.join(SCR,f"h211_smoke_{modname}_{bench}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    return r, CAP["mf"]
print("="*72)
r, mf = run_one("worker_MIXRK1","Borehole_8D")
with contextlib.redirect_stdout(io.StringIO()): b = mf._generate_rollout_batch()
pol = {p: sum(1 for t in b if t.get('_policy')==p) for p in set(t.get('_policy') for t in b)}
c1 = (mf.inference_context_k==1 and not mf.dt.disable_position_embedding and mf.rollout_reward=="mes_entropy"
      and mf.use_roi and pol.get('mes')==60 and pol.get('random')==60)
print(f"MIXRK1-B : K={mf.inference_context_k} posemb={not mf.dt.disable_position_embedding} reward={mf.rollout_reward} roi={mf.use_roi} batch {len(b)}={pol}  -> {'ok' if c1 else 'BAD'}"); ok&=c1
r, mf = run_one("worker_MIXRK1","Hartmann_6D")
c2 = (mf.inference_context_k==1 and mf.rollout_reward=="mes_entropy")
print(f"MIXRK1-H : K={mf.inference_context_k} reward={mf.rollout_reward}  -> {'ok' if c2 else 'BAD'}"); ok&=c2
r, mf = run_one("worker_K1TI","Hartmann_6D")
with contextlib.redirect_stdout(io.StringIO()): b = mf._generate_rollout_batch()
c3 = (mf.inference_context_k==1 and mf.rollout_reward=="terminal_improvement" and len(b)==60
      and not mf.dt.disable_position_embedding)
print(f"K1TI-H   : K={mf.inference_context_k} reward={mf.rollout_reward} posemb={not mf.dt.disable_position_embedding} batch {len(b)}  -> {'ok' if c3 else 'BAD'}"); ok&=c3
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*72); sys.exit(0 if ok else 1)
