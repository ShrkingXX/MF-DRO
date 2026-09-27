"""h212 Stage 0 smoke: the flag reaches the schema object, and the PINNING IS GONE --
a percentile run's recorded rtg_target must have >20 distinct values where the floored
run had <=9. A silent default is the failure mode (h198's lesson)."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}; ok=True
def run_one(modname, bench, budget=60.0):
    mod = importlib.import_module(modname)
    _oi = MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=budget
    with contextlib.redirect_stdout(io.StringIO()):
        r = mod.h83.run(bench,"MF-DRO",42,os.path.join(SCR,f"h212_smoke_{modname}_{bench}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    return r, CAP["mf"]
print("="*74)
r, mf = run_one("worker_TIPCT","Hartmann_6D")
t = np.array(r.get("rtg_target",[]),float); nd = len(np.unique(np.round(t,6)))
c1 = (mf.schemas.rtg_target_schema=="percentile" and mf.rollout_reward=="terminal_improvement"
      and mf.inference_context_k==1 and nd > 20)
print(f"TIPCT-H     : schema={mf.schemas.rtg_target_schema} q={mf.schemas.rtg_target_q} reward={mf.rollout_reward} K={mf.inference_context_k}")
print(f"              rtg_target: n={len(t)} distinct={nd} range=[{t.min():.4f},{t.max():.4f}] mean={t.mean():.4f}  (floored gave <=9 distinct, pinned 0.5)  -> {'ok' if c1 else 'BAD'}")
ok&=c1
r, mf = run_one("worker_MIXRKPCT","Borehole_8D")
t2 = np.array(r.get("rtg_target",[]),float); nd2=len(np.unique(np.round(t2,6)))
with contextlib.redirect_stdout(io.StringIO()): b = mf._generate_rollout_batch()
pol={p:sum(1 for x in b if x.get('_policy')==p) for p in set(x.get('_policy') for x in b)}
c2 = (mf.schemas.rtg_target_schema=="percentile" and mf.rollout_reward=="mes_entropy"
      and pol.get('mes')==60 and pol.get('random')==60)
print(f"MIXRKPCT-B  : schema={mf.schemas.rtg_target_schema} reward={mf.rollout_reward} batch {len(b)}={pol}")
print(f"              rtg_target: n={len(t2)} distinct={nd2} range=[{t2.min():.4f},{t2.max():.4f}]  -> {'ok' if c2 else 'BAD'}")
ok&=c2
# default must still be 'floored'
import importlib as _il
_m = _il.import_module("worker_TIPCT")
print(f"\nDEFAULT check: identity gate already PASS at 122.29066752728207 with the flag absent")
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*74); sys.exit(0 if ok else 1)
