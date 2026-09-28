"""h215 smoke: BOTH flags reach the live objects, and each still does its job --
(a) rtg_rng_parity: rollout data bit-identical to mes_entropy, rtg different (h213's gate v2)
(b) percentile schema: the inference RTG target is not pinned (modal_frac < 0.5)."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp")
import src.policy.mf_dro as MF
CAP={}
mod=importlib.import_module("worker_TIBOTH")
_oi=MF.DirectMFRegretOptimization.__init__
def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=60.0
with contextlib.redirect_stdout(io.StringIO()):
    r=mod.h83.run("Hartmann_6D","MF-DRO",42,os.path.join(SCR,"h215_smoke.json"))
MF.DirectMFRegretOptimization.__init__=_oi
mf=CAP["mf"]
print("="*80)
print(f"config: reward={mf.rollout_reward} parity={getattr(mf.config,'rtg_rng_parity',None)} "
      f"schema={mf.schemas.rtg_target_schema} q={mf.schemas.rtg_target_q} K={mf.inference_context_k}")
# (b) target not pinned
t=np.array(r.get("rtg_target",[]),float); _v,_c=np.unique(np.round(t,6),return_counts=True)
modal=float(_c.max())/max(len(t),1)
print(f"\n(b) target: n={len(t)} distinct={len(_v)} modal_frac={modal:.3f} range=[{t.min():.4f},{t.max():.4f}]")
print(f"    (floored gave modal_frac 0.92 pinned at 0.5)  -> {'ok' if (len(_v)>=5 and modal<0.5) else 'BAD'}")
okb = len(_v)>=5 and modal<0.5
# (a) rollout parity
def batch_under(reward):
    mf.rollout_reward=reward; torch.manual_seed(999)
    with contextlib.redirect_stdout(io.StringIO()): return mf._generate_rollout_batch()
b_ti=batch_under("terminal_improvement"); b_me=batch_under("mes_entropy"); mf.rollout_reward="terminal_improvement"
n=min(len(b_ti),len(b_me))
ds=max(float((b_ti[i]['states']-b_me[i]['states']).abs().max()) for i in range(n))
da=max(float((b_ti[i]['actions_x']-b_me[i]['actions_x']).abs().max()) for i in range(n))
dr=max(float((b_ti[i]['rtg']-b_me[i]['rtg']).abs().max()) for i in range(n))
print(f"\n(a) parity over {n} rollouts: max|d states|={ds:.3e} max|d actions|={da:.3e} (want 0)   max|d rtg|={dr:.3e} (want >0)")
oka = (ds==0.0 and da==0.0 and dr>0.0)
print(f"    -> {'ok' if oka else 'BAD'}")
ok = oka and okb and mf.rollout_reward=="terminal_improvement" and getattr(mf.config,'rtg_rng_parity',False) and mf.schemas.rtg_target_schema=="percentile"
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*80); sys.exit(0 if ok else 1)
