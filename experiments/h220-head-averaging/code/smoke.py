"""h220 smoke. Two jobs:
(a) the defining flags reach the live objects (loc_loss, mixture counts, oracle actions);
(b) REPORT what the RTG channel is being asked to do under mes_entropy. Under
    terminal_improvement the oracle half was labelled ~+50 better than MES. Under
    mes_entropy the label is INFORMATION GAIN, and an oracle that queries x* repeatedly
    may be information-POOR -- possibly labelled WORSE than MES. That does not invalidate
    the primary readout (|x - x*| measures whether the head averages or picks a mode, which
    h207 showed is RTG-independent under MSE), but it changes what the secondary RTG story
    would mean, so it is measured here rather than assumed."""
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
    if hasattr(mod,"_ORACLE"):
        mod._ORACLE["x_star"]=torch.tensor(mod.XSTAR["Borehole_8D"],dtype=torch.float64)
        mod._ORACLE["rng"]=torch.Generator().manual_seed(42*7919+207)
        MF.simulate_mf_trajectory=mod._mixed_sim
    _oi=MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=budget
    with contextlib.redirect_stdout(io.StringIO()):
        mod.h83.run("Borehole_8D","MF-DRO",42,os.path.join(SCR,f"h220_smoke_{modname}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    return mod, CAP["mf"]
print("="*84)
for modname,exp_loss,exp in (("worker_MIXOMSE","mse",(60,30)),("worker_MIXOL1","l1",(60,30)),("worker_MIXRL1","l1",(60,60))):
    mod,mf=run_one(modname)
    with contextlib.redirect_stdout(io.StringIO()): b=mf._generate_rollout_batch()
    pol={p:sum(1 for t in b if t.get('_policy')==p) for p in set(t.get('_policy') for t in b)}
    n_mes,n_other=exp
    c=(mf.dt.loc_loss==exp_loss and mf.inference_context_k==1 and mf.rollout_reward=="mes_entropy"
       and pol.get('mes')==n_mes and sum(v for k,v in pol.items() if k!='mes')==n_other)
    print(f"{modname[7:]:9s}: loss={mf.dt.loc_loss} K={mf.inference_context_k} reward={mf.rollout_reward} batch={pol} (want mes={n_mes}, other={n_other})  -> {'ok' if c else 'BAD'}")
    ok&=c
    # what is RTG asked to do?
    r0={}
    for p in pol:
        v=[float(t['rtg'][0]) for t in b if t.get('_policy')==p]
        r0[p]=(float(np.mean(v)),float(np.std(v)))
    print(f"           rtg[0] by policy: " + "   ".join(f"{k}: {m:+.3f} (sd {s:.3f})" for k,(m,s) in sorted(r0.items())))
    if 'oracle' in r0:
        better = "HIGHER (selectable as good)" if r0['oracle'][0]>r0['mes'][0] else "LOWER -- the oracle half is labelled WORSE under an info-gain reward"
        print(f"           -> oracle half is {better}")
        xs=mod._ORACLE["x_star"]; lo,hi=mf.bounds[0],mf.bounds[1]; xsu=((xs-lo)/(hi-lo)).numpy()
        d=np.mean([np.abs(t['actions_x'][0].numpy()-xsu).max() for t in b if t.get('_policy')=='oracle'])
        print(f"           oracle tau=0 actions |a0 - x*|inf = {d:.3f} (want < 0.1)")
        ok &= (d<0.1)
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*84); sys.exit(0 if ok else 1)
