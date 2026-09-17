"""h209 Stage 0 smoke -- core unchanged since h207's gates; this checks the NEW pieces:
(a) MIXR and NIR run on Hartmann_6D under this config (BUDGET=30), (b) NIR120's batch is
exactly 120 MES trajectories, (c) MIXR-H's batch is 60 MES + 60 random."""
import os, sys, io, contextlib, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp")
import src.policy.mf_dro as MF
CAP = {}
ok = True
def run_one(modname, bench, budget=30.0):
    mod = importlib.import_module(modname)
    _oi = MF.DirectMFRegretOptimization.__init__
    def _spy(self, *a, **k): _oi(self, *a, **k); CAP["mf"] = self
    MF.DirectMFRegretOptimization.__init__ = _spy
    mod.h83.BUDGET = budget
    with contextlib.redirect_stdout(io.StringIO()):
        r = mod.h83.run(bench, "MF-DRO", 42, os.path.join(SCR, f"h209_smoke_{modname}_{bench}.json"))
    MF.DirectMFRegretOptimization.__init__ = _oi
    mf = CAP["mf"]
    with contextlib.redirect_stdout(io.StringIO()):
        batch = mf._generate_rollout_batch()
    pol = {p: sum(1 for t in batch if t.get('_policy') == p) for p in set(t.get('_policy') for t in batch)}
    return r, mf, batch, pol
print("=" * 70)
r, mf, b, pol = run_one("worker_MIXR", "Hartmann_6D")
print(f"MIXR on Hartmann_6D: {len(r['queries'])} queries, batch {len(b)} = {pol}, reward={mf.rollout_reward}, K={mf.inference_context_k}, nopos={mf.dt.disable_position_embedding}")
ok &= (len(b) == 120 and pol.get('mes') == 60 and pol.get('random') == 60 and mf.rollout_reward == 'terminal_improvement')
r, mf, b, pol = run_one("worker_NIR", "Hartmann_6D")
print(f"NIR  on Hartmann_6D: {len(r['queries'])} queries, batch {len(b)} = {pol}, reward={mf.rollout_reward}")
ok &= (len(b) == 60 and mf.rollout_reward == 'terminal_improvement')
r, mf, b, pol = run_one("worker_NIR120", "Borehole_8D")
print(f"NIR120 on Borehole:  {len(r['queries'])} queries, batch {len(b)} = {pol}, reward={mf.rollout_reward}")
ok &= (len(b) == 120 and pol.get('mes') == 120 and mf.rollout_reward == 'terminal_improvement')
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("=" * 70)
sys.exit(0 if ok else 1)
