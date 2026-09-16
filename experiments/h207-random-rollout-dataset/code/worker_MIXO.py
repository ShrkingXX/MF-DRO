"""h207 arm MIXO -- control's 20 MES + 20 ORACLE rollouts per member. usage: worker_MIXO.py <bench> <seed>

CEILING/DIAGNOSTIC, NOT A METHOD: x* is unavailable at run time. Oracle half =
x* + N(0, (0.02*range)^2) at EVERY step (fidelity by forced_x's own info-gain
rule), so MES and oracle rollouts of a member share an identical tau=0 state and
differ only in action and label. Installed by wrapping simulate_mf_trajectory and
acting ONLY when rollout_policy == 'oracle'; the MES half is the control's rollouts.
Base config = h206N (K=8, no positional embedding, ROI-Q10). Label via h83.ROLLOUT_REWARD.
"""
import os, sys, importlib.util
import torch
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
import src.policy.mf_dro as MF
from src.policy.mf_dro import DirectMFRegretOptimization as _DMRO
SWEEP = [-1.0, -0.5, -0.2, 0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
XSTAR = {"Hartmann_6D": [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573],
         "Borehole_8D": [0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0]}
NOISE_FRAC = 0.02
_ORIG_SIM = MF.simulate_mf_trajectory
_ORACLE = {"x_star": None, "rng": None, "n": 0, "max_dev_frac": 0.0}

def _oracle_path(bounds, T):
    lo, hi = bounds[0], bounds[1]
    xs = _ORACLE["x_star"].to(dtype=bounds.dtype)
    eps = torch.randn(T, xs.numel(), generator=_ORACLE["rng"], dtype=bounds.dtype)
    path = xs.unsqueeze(0) + NOISE_FRAC * (hi - lo).unsqueeze(0) * eps
    path = torch.max(torch.min(path, hi.unsqueeze(0)), lo.unsqueeze(0))
    _ORACLE["max_dev_frac"] = max(_ORACLE["max_dev_frac"],
                                  float(((path - xs) / (hi - lo)).abs().max()))
    return path

def _mixed_sim(*args, **kw):
    if kw.get("rollout_policy") == "oracle":
        bounds = kw.get("bounds", args[6] if len(args) > 6 else None)
        T = int(kw.get("rollout_length", args[3] if len(args) > 3 else 8))
        kw["forced_x"] = _oracle_path(bounds, T)
        kw["rollout_policy"] = "mes"          # valid branch; x is overridden by forced_x
        _ORACLE["n"] += 1
    return _ORIG_SIM(*args, **kw)

_OB = h83._build_mf_dro_config
def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8; c.max_seq_length = 256
    c.absolute_timesteps = False; c.real_prefix_training = False
    c.disable_position_embedding = True    # h206
    c.rollout_mix = [('mes', 20, None), ('oracle', 20, None)]
    return c
h83._build_mf_dro_config = _build
h83.ROLLOUT_REWARD = "terminal_improvement"
_OI = _DMRO.__init__
def _init(self, *a, **k):
    _OI(self, *a, **k); self._h168_probe = SWEEP
_DMRO.__init__ = _init

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    _ORACLE["x_star"] = torch.tensor(XSTAR[bench], dtype=torch.float64)
    _ORACLE["rng"] = torch.Generator().manual_seed(seed * 7919 + 207)
    MF.simulate_mf_trajectory = _mixed_sim
    tag = f"{bench}__H207MIXO__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h207"] = dict(arm="MIXO", rollout_mix="[('mes',20,None),('oracle',20,None)]",
                      rollout_reward="terminal_improvement", oracle="x*+N(0,(0.02 range)^2) every step",
                      oracle_rollouts=_ORACLE["n"], oracle_max_dev_frac=_ORACLE["max_dev_frac"],
                      inference_context_k=8, disable_position_embedding=True, roi="Q10", h168_sweep=SWEEP)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"oracle_rollouts={_ORACLE['n']} wall={r['_wall_s']/60:.1f}m", flush=True)
