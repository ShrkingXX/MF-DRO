"""h213 Stage 0 smoke -- THE GATE IS THE PARITY CHECK ITSELF.

A terminal_improvement run with rtg_rng_parity=True must produce the SAME first real
query as the mes_entropy control on the same seed. If it does not, the streams still
differ and the label comparison is still not a one-factor test."""
import os, sys, io, contextlib, importlib, json
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR=os.environ.get("SCRATCH","/tmp"); BUDGET=30.0
import src.policy.mf_dro as MF
CAP={}
def run(modname, bench, tag):
    mod = importlib.import_module(modname)
    _oi = MF.DirectMFRegretOptimization.__init__
    def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
    MF.DirectMFRegretOptimization.__init__=_spy; mod.h83.BUDGET=BUDGET
    with contextlib.redirect_stdout(io.StringIO()):
        r = mod.h83.run(bench,"MF-DRO",42,os.path.join(SCR,f"h213_smoke_{tag}.json"))
    MF.DirectMFRegretOptimization.__init__=_oi
    q=[e for e in r["queries"] if not e.get("is_init")]
    return CAP["mf"], [(round(float(e["cost_cum"]),4), int(e["fid"]), round(float(e["y"]),9)) for e in q[:3]]
print("="*76)
mf_p, q_par = run("worker_TIPAR","Hartmann_6D","tipar")
print(f"TI-PAR-H (terminal_improvement, parity ON): reward={mf_p.rollout_reward} "
      f"parity={getattr(mf_p.config,'rtg_rng_parity',None)} K={mf_p.inference_context_k}")
print(f"  first 3 real queries: {q_par}")
sys.path.insert(0, os.path.join(REPO,"experiments/h211-mixr-on-k1/code"))
import importlib.util as _iu
_s=_iu.spec_from_file_location("ctrlmod", os.path.join(REPO,"experiments/h210-mean-vs-median/code/worker_CTRLK1.py"))
_m=_iu.module_from_spec(_s); sys.modules["ctrlmod"]=_m; _s.loader.exec_module(_m)
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self,*a,**k): _oi(self,*a,**k); CAP["mf"]=self
MF.DirectMFRegretOptimization.__init__=_spy; _m.h83.BUDGET=BUDGET
with contextlib.redirect_stdout(io.StringIO()):
    r = _m.h83.run("Hartmann_6D","MF-DRO",42,os.path.join(SCR,"h213_smoke_ctrl.json"))
MF.DirectMFRegretOptimization.__init__=_oi
mf_c=CAP["mf"]; qc=[e for e in r["queries"] if not e.get("is_init")]
q_ctl=[(round(float(e["cost_cum"]),4), int(e["fid"]), round(float(e["y"]),9)) for e in qc[:3]]
print(f"CTRL-K1-H (mes_entropy):                   reward={mf_c.rollout_reward} K={mf_c.inference_context_k}")
print(f"  first 3 real queries: {q_ctl}")
same1 = q_par[:1]==q_ctl[:1]; same3 = q_par==q_ctl
print(f"\nGATE -- parity: first real query identical? {same1}    first THREE identical? {same3}")
print(f"  (h211's unmatched pair differed on the FIRST query on 3/3 seeds)")
ok = bool(same1) and mf_p.rollout_reward=="terminal_improvement" and getattr(mf_p.config,'rtg_rng_parity',False)
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("="*76); sys.exit(0 if ok else 1)
