"""h210 Stage 0 smoke: each arm builds and runs (BUDGET=30); the flags that define each arm
are actually set on the constructed objects (a silent default is the failure mode)."""
import os, sys, io, contextlib, importlib
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp")
import src.policy.mf_dro as MF
CAP = {}; ok = True
def run_one(modname, bench):
    mod = importlib.import_module(modname)
    _oi = MF.DirectMFRegretOptimization.__init__
    def _spy(self, *a, **k): _oi(self, *a, **k); CAP["mf"] = self
    MF.DirectMFRegretOptimization.__init__ = _spy
    mod.h83.BUDGET = 30.0
    with contextlib.redirect_stdout(io.StringIO()):
        r = mod.h83.run(bench, "MF-DRO", 42, os.path.join(SCR, f"h210_smoke_{modname}_{bench}.json"))
    MF.DirectMFRegretOptimization.__init__ = _oi
    return r, CAP["mf"]
print("=" * 70)
r, mf = run_one("worker_CTRLK1", "Hartmann_6D")
c1 = (mf.inference_context_k == 1 and mf.rollout_reward == "mes_entropy" and not mf.dt.disable_position_embedding and mf.dt.loc_loss == "mse" and mf.use_roi)
print(f"CTRLK1-H : K={mf.inference_context_k} reward={mf.rollout_reward} nopos={mf.dt.disable_position_embedding} loss={mf.dt.loc_loss} roi={mf.use_roi}  -> {'ok' if c1 else 'BAD'}"); ok &= c1
r, mf = run_one("worker_L1NIR", "Borehole_8D")
c2 = (mf.dt.loc_loss == "l1" and mf.inference_context_k == 8 and mf.dt.disable_position_embedding and mf.rollout_reward == "terminal_improvement")
print(f"L1NIR-B  : K={mf.inference_context_k} reward={mf.rollout_reward} nopos={mf.dt.disable_position_embedding} loss={mf.dt.loc_loss}  -> {'ok' if c2 else 'BAD'}"); ok &= c2
r, mf = run_one("worker_MIXRP72", "Borehole_8D")
with contextlib.redirect_stdout(io.StringIO()):
    b = mf._generate_rollout_batch()
rnd = [t for t in b if t.get('_policy') == 'random']; mes = [t for t in b if t.get('_policy') == 'mes']
hf_rnd = float(np.mean([1 - t['lf_fraction'] for t in rnd])); hf_mes = float(np.mean([1 - t['lf_fraction'] for t in mes]))
c3 = (mf.dt.loc_loss == "mse" and getattr(mf.config, 'random_p_hf', None) == 0.72 and len(rnd) == 60 and abs(hf_rnd - hf_mes) < 0.08)
print(f"MIXRP72-B: loss={mf.dt.loc_loss} random_p_hf={getattr(mf.config,'random_p_hf',None)} batch {len(mes)} mes + {len(rnd)} random; HF frac in rollouts: random {hf_rnd:.2f} vs MES {hf_mes:.2f} (want |diff| < 0.08 = mix-matched)  -> {'ok' if c3 else 'BAD'}"); ok &= c3
r, mf = run_one("worker_L1NIR", "Hartmann_6D")
c4 = (mf.dt.loc_loss == "l1")
print(f"L1NIR-H  : loss={mf.dt.loc_loss} K={mf.inference_context_k}  -> {'ok' if c4 else 'BAD'}"); ok &= c4
print(f"\nSMOKE: {'PASS' if ok else 'FAIL'}"); print("=" * 70); sys.exit(0 if ok else 1)
