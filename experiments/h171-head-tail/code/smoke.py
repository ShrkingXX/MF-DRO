"""h171 SC2: is the HEAD/TAIL split actually wired? A mis-wired split would
silently produce two copies of the same arm. h169 was lost to exactly this.

Observable: head_mes takes the MES argmax at tau=0 (CONCENTRATED across the
batch) and random after (DISPERSED). tail_mes is the reverse.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np, torch
import src.policy.mf_dro as MF

CAP = {}
_ORIG = MF.simulate_mf_trajectory
def _wrap(*a, **k):
    t = _ORIG(*a, **k)
    if "actions_x" in t:
        CAP.setdefault(k.get("rollout_policy", "?"), []).append(t["actions_x"].numpy())
    return t
MF.simulate_mf_trajectory = _wrap

_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
h83.BUDGET = 6.0
_OB = h83._build_mf_dro_config

for arm, pol in (("head", "head_mes"), ("tail", "tail_mes")):
    def _b(*a, _p=pol, **k):
        c = _OB(*a, **k); c.rollout_policy = _p; return c
    h83._build_mf_dro_config = _b
    CAP.clear()
    h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(os.environ["SCRATCH"], f"h171_{arm}.json"))
    A = np.array(CAP.get(pol, []))          # [n_traj, T, d]
    if A.size == 0:
        print(f"  {arm}: NO DATA -- policy string never reached simulate_mf_trajectory"); continue
    sd0 = float(A[:, 0, :].std(axis=0).mean())          # spread of tau=0 actions
    sdR = float(A[:, 1:, :].std(axis=0).mean())         # spread of tau>0 actions
    print(f"  {arm:5s} (rollout_policy={pol}): n_traj={len(A)}  "
          f"tau=0 spread {sd0:.4f}   tau>0 spread {sdR:.4f}   ratio {sd0/max(sdR,1e-9):.3f}")

print("\n  SC2 expects: head tau=0 spread MUCH SMALLER than its tau>0 (MES argmax is concentrated),")
print("               tail tau=0 spread MUCH LARGER than its tau>0 (uniform draws).")
