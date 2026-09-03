"""h201 SC1 (GATE): does the K=8 window actually move the DT's readout to position 7?

The whole premise of h201 is that the window changes WHICH timestep reaches inference.
That is currently an inference from reading decisionTransformer.py (`ts = torch.arange(T)`
and a readout of the LAST state token). This measures it in a live run instead.

GATE: the readout index must reach 7 once >=8 real queries exist. If it never exceeds 0,
h201's premise is false and both arms are void -- reported as a GATE MISS, not a result.
"""
import os, sys, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

import src.model.decisionTransformer as DTM
REC = []
_orig = DTM.DecisionTransformer.propose_mf
def _spy(self, *a, **k):
    hist = k.get("hist")
    if hist is None and len(a) >= 1 and isinstance(a[-1], list):
        hist = a[-1]
    T = (len(hist) + 1) if hist else 1
    REC.append(T - 1)                      # readout index = last state token
    return _orig(self, *a, **k)
DTM.DecisionTransformer.propose_mf = _spy

sys.argv = ["w", "Borehole_8D", "42"]
_s = importlib.util.spec_from_file_location(
    "h201A", os.path.join(REPO, "experiments/h201-oracle-teacher-window/code/worker_A.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h201A"] = w; _s.loader.exec_module(w)
w.h83.BUDGET = 62.0                        # ~22 real queries past the 40-unit init
import torch as _t
w._EXPERT["x_star"] = _t.tensor(w.XSTAR["Borehole_8D"], dtype=_t.float64)
w._EXPERT["rng"] = _t.Generator().manual_seed(42 * 7919 + 145)
import src.policy.mf_dro as MF
MF.simulate_mf_trajectory = w._expert_sim
w.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], "h201_sc.json"))

R = np.array(REC)
print("\n" + "="*66)
print(f"SC1 (GATE): readout index per propose_mf call, {len(R)} calls")
print(f"   sequence: {R[:24].tolist()}{' ...' if len(R)>24 else ''}")
print(f"   max index reached: {R.max()}   (need 7)")
gate = R.max() >= 7
print(f"   -> {'PASS -- the window DOES move the readout to timestep 7' if gate else 'GATE MISS -- premise false, h201 is void'}")
print(f"\nSC2: teacher path checks (accumulated in-run over {w._EXPERT['n']} rollouts)")
print(f"   max |x_tau7 - x*| = {w._EXPERT['tau7_max_dev']:.3e}   (need ~0)")
t0 = torch.stack(w._EXPERT["tau0"])
print(f"   tau=0 across-rollout SD (normalised) = {float(t0.std(dim=0).mean()):.4f}   (uniform start => ~0.29)")
print("="*66)
print("STAGE 0:", "PASS" if gate and w._EXPERT['tau7_max_dev'] < 1e-6 else "FAIL")
