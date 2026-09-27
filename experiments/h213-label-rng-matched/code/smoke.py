"""h213 Stage 0 smoke v2 -- THE GATE IS ROLLOUT-DATA PARITY, not query identity.

v1 required the parity run to reproduce the control's FIRST REAL QUERY. That gate was
wrong: with RNG matched, both labels generate the SAME rollouts, but they attach DIFFERENT
RTG numbers, and RTG is an input to DT training -- so the trained model, and hence the
first query, SHOULD differ. That difference is the label effect, which is the hypothesis
under test, not a precondition for it.

The correct one-factor check: from an identical state and seed, the two labels must produce
BIT-IDENTICAL rollout data (states, actions, fidelities) and DIFFERENT rtg vectors.
Verified separately: 963 Thompson calls and an identical post-batch RNG fingerprint under
both labels."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}
mod=importlib.import_module("worker_TIPAR")
_oi=MF.DirectMFRegretOptimization.__init__
def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=20.0
with contextlib.redirect_stdout(io.StringIO()):
    mod.h83.run("Hartmann_6D","MF-DRO",42,os.path.join(SCR,"h213_smoke.json"))
MF.DirectMFRegretOptimization.__init__=_oi
mf=CAP["mf"]
print("="*80)
print(f"config: reward={mf.rollout_reward} parity={getattr(mf.config,'rtg_rng_parity',None)} K={mf.inference_context_k}")
def batch_under(reward):
    mf.rollout_reward=reward
    torch.manual_seed(999)
    with contextlib.redirect_stdout(io.StringIO()): return mf._generate_rollout_batch()
b_ti = batch_under("terminal_improvement")
b_me = batch_under("mes_entropy")
mf.rollout_reward="terminal_improvement"
n=min(len(b_ti),len(b_me))
ds = max(float((b_ti[i]['states']-b_me[i]['states']).abs().max()) for i in range(n))
da = max(float((b_ti[i]['actions_x']-b_me[i]['actions_x']).abs().max()) for i in range(n))
dl = max(float((b_ti[i]['actions_ell']-b_me[i]['actions_ell']).abs().max()) for i in range(n))
dr = max(float((b_ti[i]['rtg']-b_me[i]['rtg']).abs().max()) for i in range(n))
print(f"\nSame seed, same state, {n} rollouts each:")
print(f"  max |delta states|      = {ds:.3e}   (want 0: same rollouts)")
print(f"  max |delta actions_x|   = {da:.3e}   (want 0)")
print(f"  max |delta actions_ell| = {dl:.3e}   (want 0)")
print(f"  max |delta rtg|         = {dr:.3e}   (want > 0: the label IS the difference)")
ok = (ds==0.0 and da==0.0 and dl==0.0 and dr>0.0
      and mf.rollout_reward=="terminal_improvement" and getattr(mf.config,'rtg_rng_parity',False))
print(f"\nGATE -- rollout data identical, rtg different: {'PASS' if ok else 'FAIL'}")
print(f"  (v1's gate demanded the first real QUERY match; that was wrong -- see docstring)")
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*80); sys.exit(0 if ok else 1)
