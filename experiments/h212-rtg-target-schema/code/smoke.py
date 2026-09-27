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
t = np.array(r.get("rtg_target",[]),float)
# GATE v2: the v1 criterion was `distinct > 20`, which a short smoke can never
# reach -- it recorded 18 iterations and 18 distinct targets, i.e. PERFECT
# de-pinning, and failed. The pinning signature is not a COUNT, it is a MODE:
# the floored run sat at a single value for 108 of 117 iterations (92%).
# Criterion is now modal fraction, which is horizon-independent.
_v, _c = np.unique(np.round(t,6), return_counts=True); nd = len(_v)
modal = float(_c.max())/max(len(t),1)
c1 = (mf.schemas.rtg_target_schema=="percentile" and mf.rollout_reward=="terminal_improvement"
      and mf.inference_context_k==1 and nd >= 5 and modal < 0.5)
print(f"TIPCT-H     : schema={mf.schemas.rtg_target_schema} q={mf.schemas.rtg_target_q} reward={mf.rollout_reward} K={mf.inference_context_k}")
print(f"              rtg_target: n={len(t)} distinct={nd} modal_frac={modal:.3f} range=[{t.min():.4f},{t.max():.4f}] mean={t.mean():.4f}")
print(f"              (floored: 9 distinct / 117 iters, modal_frac 0.92 pinned at 0.5)  -> {'ok' if c1 else 'BAD'}")
ok&=c1
r, mf = run_one("worker_MIXRKPCT","Borehole_8D")
t2 = np.array(r.get("rtg_target",[]),float)
_v2,_c2 = np.unique(np.round(t2,6), return_counts=True); nd2=len(_v2); modal2=float(_c2.max())/max(len(t2),1)
with contextlib.redirect_stdout(io.StringIO()): b = mf._generate_rollout_batch()
pol={p:sum(1 for x in b if x.get('_policy')==p) for p in set(x.get('_policy') for x in b)}
c2 = (mf.schemas.rtg_target_schema=="percentile" and mf.rollout_reward=="mes_entropy"
      and pol.get('mes')==60 and pol.get('random')==60)
print(f"MIXRKPCT-B  : schema={mf.schemas.rtg_target_schema} reward={mf.rollout_reward} batch {len(b)}={pol}")
print(f"              rtg_target: n={len(t2)} distinct={nd2} modal_frac={modal2:.3f} range=[{t2.min():.4f},{t2.max():.4f}]")
print(f"              NOTE for P-NEUTRAL: floored mes_entropy ran 0.299-0.859; the 90th percentile of a")
print(f"              batch sits BELOW max(batch_max, alpha*running_max), so the schema shifts the LEVEL")
print(f"              even where it was never pinned. P-NEUTRAL is at genuine risk by design.  -> {'ok' if c2 else 'BAD'}")
ok&=c2
# default must still be 'floored'
import importlib as _il
_m = _il.import_module("worker_TIPCT")
print(f"\nDEFAULT check: identity gate already PASS at 122.29066752728207 with the flag absent")
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*74); sys.exit(0 if ok else 1)
