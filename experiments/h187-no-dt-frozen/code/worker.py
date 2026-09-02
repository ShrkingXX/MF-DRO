"""h187 -- the no-DT control on the FROZEN metric. usage: worker.py <bench> <seed>

h31's mechanism verbatim: replace ONLY dt.propose_mf, so compute_joint_mf_mes picks
(x, ell) directly from the same candidate pool the DT would have scored. Initial
design, cost accounting and regret curve are the identical code path.

_dt_snapshot is None by default and does not drive the run, so patching the live
dt instance is sufficient.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
import torch
from src.policy.mf_dro import (DirectMFRegretOptimization as _DMRO,
                               compute_joint_mf_mes as _mes)

_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES

_OI = _DMRO.__init__


def _init(self, *a, **k):
    _OI(self, *a, **k)

    def _teacher_propose(state, rtg_target, btg_target, timestep=0,
                         use_candidate_scoring=False, candidate_features=None,
                         fidelity_sampling=True, hist=None):
        lo, hi = self.bounds[0], self.bounds[1]
        # DEVIATION FROM h31, forced by the current code: the call site passes
        # candidate_features=None unless use_candidate_scoring is on, and that
        # flag must NOT be enabled (pool+argmax is not an acceptable fix for
        # MF-DRO). The teacher is an acquisition rule and inherently needs a pool
        # to argmax over, so this arm draws its OWN, with the same size and
        # distribution the candidate path would have used:
        #   n_infer_candidates uniform draws over bounds.
        # simulate_mf_trajectory's roi_candidates is likewise uniform over
        # bounds, so training and this control score identical input pools.
        X_raw = lo + (hi - lo) * torch.rand(
            int(self.n_infer_candidates), int(self.d), dtype=torch.float64)
        x_raw, ell, _ = _mes(self.ko_ensemble[0], X_raw, self.c_H, self.c_L)
        x_norm = ((x_raw - lo) / (hi - lo)).clamp(0.0, 1.0).float()
        # The run loop logs dt.last_p_pred after every proposal. The teacher has no
        # fidelity probability -- it chooses ell deterministically -- so record the
        # choice itself. This is also SC1's observable: p_pred is then exactly
        # 0.0/1.0 every iteration, which a live DT never produces.
        self.dt.last_p_pred = float(ell)
        return x_norm, int(ell)

    self.dt.propose_mf = _teacher_propose


_DMRO.__init__ = _init

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__NODT__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
