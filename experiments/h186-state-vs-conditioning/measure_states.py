"""h186 confound resolver: how far apart ARE the tau=0 states the probe sweeps?

The probe records each state's LABEL, not its vector, so state sensitivity could
not be normalised against conditioning sensitivity. This recomputes the states on a
short run and measures their pairwise distances directly. Read-only w.r.t. the
policy -- it only captures what _generate_rollout_batch already stores.
"""
import os, sys, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import src.policy.mf_dro as MF

CAP = {}
_OI = MF.DirectMFRegretOptimization.__init__
def _init(self, *a, **k):
    _OI(self, *a, **k); CAP["mf"] = self
MF.DirectMFRegretOptimization.__init__ = _init

_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
h83.BUDGET = 10.0
h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ.get("SCRATCH", "/tmp"), "h186.json"))

st = CAP["mf"]._last_batch_tau0_states
if not st:
    print("  no tau=0 states captured"); sys.exit(1)
A = torch.stack([s.reshape(-1) for s in st]).double().numpy()
print(f"\n  tau=0 states captured : {A.shape[0]}  dim {A.shape[1]}")
uniq = np.unique(np.round(A, 10), axis=0)
print(f"  distinct states       : {uniq.shape[0]}")
D = [np.linalg.norm(uniq[i] - uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
if D:
    print(f"  pairwise distance     : mean {np.mean(D):.4f}  min {np.min(D):.4f}  max {np.max(D):.4f}")
print(f"  state vector norm     : mean {np.linalg.norm(A, axis=1).mean():.4f}")
print(f"  per-coordinate sd     : {A.std(axis=0).mean():.6f}")
if D:
    print(f"\n  RELATIVE state variation = pairwise dist / state norm "
          f"= {np.mean(D)/np.linalg.norm(A,axis=1).mean():.4f}")
