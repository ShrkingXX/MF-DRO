"""h208 -- rollout-length dose on the ORIGINAL SF-DRO (papers/Old_dro.py).

usage: worker.py <arm> <seed>      arm in ARMS below

papers/Old_dro.py is imported UNMODIFIED (its header forbids edits). Everything
this worker changes is done by (a) the config it hands the class and (b) a
subclass that wraps two methods for diagnostics and, in the L8-TRUNC arm,
truncates trajectories before training. See ../protocol.md.
"""
import os, sys, json, time, importlib.util
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from omegaconf import OmegaConf

_s = importlib.util.spec_from_file_location("old_dro", os.path.join(REPO, "papers/Old_dro.py"))
old = importlib.util.module_from_spec(_s); sys.modules["old_dro"] = old; _s.loader.exec_module(old)
from src.objectives import Ackley

RES = os.path.abspath(os.path.join(H, "..", "results"))

# arm -> (max_rollout_length, early_stop, truncate_to_first_step)
ARMS = {
    "ESON-L4":  (4, True,  False),   # the literal original config
    "L1":       (1, False, False),
    "L2":       (2, False, False),
    "L4":       (4, False, False),
    "L8":       (8, False, False),
    "L8-TRUNC": (8, False, True),    # 8-step rollouts, rtg[0] over 8 steps, train on position 0 only
}

# Original setting (config/test_function/Ackley.yaml + paper Sec 5.1 + main.py's
# objective construction): 10D, [-32.768, 32.768]^10, input shift 10, noise 0.01.
D = 10
LO, HI = -32.768, 32.768
SHIFT = 10.0
NOISE = 0.01
N_INIT = 5
N_ITER = 500


def _code_state():
    import subprocess
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    dirty = g("status", "--porcelain", "src", "papers/Old_dro.py", "config/method/dro.yaml")
    return dict(commit=g("rev-parse", "HEAD"), dirty=bool(dirty), dirty_files=dirty.splitlines()[:20])


def _atomic(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), default=float)
    os.replace(tmp, path)


class OldDRO(old.DirectRegretOptimization):
    """Old_dro.DirectRegretOptimization + per-iteration diagnostics + optional
    truncation of trajectories to their first step before DT training."""

    def __init__(self, cfg, obj, truncate):
        super().__init__(cfg, obj)
        self._truncate = truncate
        self._iter_lens = []      # realised rollout lengths, this iteration
        self._diag = []           # one dict per real iteration
        self._t_iter = None

    def _simulate_trajectory(self, gp_idx, initial_state, max_length):
        t = super()._simulate_trajectory(gp_idx, initial_state, max_length)
        self._iter_lens.append(int(t["actions"].shape[0]))
        return t

    def _train_decision_transformer(self, trajectories):
        # rtg[0] as Old_dro will compute it: sum of ALL rewards in the rollout.
        rtg0 = [float(t["rewards"].sum()) for t in trajectories]
        if self._truncate:
            # Keep rtg[0] EXACTLY (one reward = the full-rollout sum), drop
            # positions >= 1. Old_dro's own loop then builds a length-1 sequence
            # with rtg[0] = that sum and timestep 0 -- the inference layout.
            trajectories = [{
                "states": t["states"][:2],
                "actions": t["actions"][:1],
                "rewards": t["rewards"].sum().reshape(1),
                "final_regret": t["final_regret"],
            } for t in trajectories]
            assert all(int(t["actions"].shape[0]) == 1 for t in trajectories)
        super()._train_decision_transformer(trajectories)
        self._diag.append(dict(
            n_traj=len(trajectories),
            lens=list(self._iter_lens),
            rtg0_mean=float(np.mean(rtg0)), rtg0_max=float(np.max(rtg0)),
            rtg0_zero_frac=float(np.mean([r <= 1e-12 for r in rtg0])),
        ))
        self._iter_lens = []


def run(arm, seed):
    L, es, trunc = ARMS[arm]
    cfg = OmegaConf.load(os.path.join(REPO, "config/method/dro.yaml"))
    cfg.verbose = False
    cfg.save_dir = None
    cfg.seed = seed
    cfg.simulation.max_rollout_length = L
    cfg.simulation.early_stop = es
    cfg.bo.input_dim = D
    cfg.bo.domain_min = [LO] * D
    cfg.bo.domain_max = [HI] * D
    cfg.bo.initial_points = N_INIT
    cfg.bo.max_iterations = N_ITER
    cfg.bo.objective = "maximize"

    shift = torch.tensor([SHIFT] * D)
    obj = Ackley(dim=D, bounds=[(LO, HI)] * D, negate=True, noise_std=NOISE, shift=shift)
    f_true = Ackley(dim=D, bounds=[(LO, HI)] * D, negate=False, noise_std=None, shift=shift)

    t0 = time.time()
    dro = OldDRO(cfg, obj, trunc)
    res = dro.run_optimization()
    wall = time.time() - t0

    X = np.asarray(res["all_x"], dtype=float)
    Y = np.asarray(res["all_y"], dtype=float).reshape(-1)
    with torch.no_grad():
        ftrue = f_true(torch.tensor(X)).numpy().reshape(-1)     # Ackley >= 0, min 0 at x = shift
    # incumbent = argmax of OBSERVED y among the first t points; regret = f_true there
    inc = np.array([int(np.argmax(Y[:t + 1])) for t in range(len(Y))])
    regret_true = ftrue[inc]
    regret_obs = 0.0 - np.maximum.accumulate(Y)

    r = dict(
        arm=arm, seed=seed, rollout_length=L, early_stop=es, truncate=trunc,
        n_eval=int(len(Y)), n_expected=N_INIT + N_ITER,
        final_regret=float(regret_true[-1]), final_regret_obs=float(regret_obs[-1]),
        regret_curve=regret_true.tolist(), regret_obs_curve=regret_obs.tolist(),
        all_x=X.tolist(), all_y=Y.tolist(),
        best_x=np.asarray(res["best_x"], dtype=float).tolist(), best_y=float(res["best_y"]),
        diag=dro._diag,
        realised_len_mean=float(np.mean([l for d in dro._diag for l in d["lens"]])) if dro._diag else float("nan"),
        realised_len_min=int(min(l for d in dro._diag for l in d["lens"])) if dro._diag else -1,
        realised_len_max=int(max(l for d in dro._diag for l in d["lens"])) if dro._diag else -1,
        _wall_s=round(wall, 1), _code=_code_state(),
        _cfg=OmegaConf.to_container(cfg, resolve=True),
        _setting=dict(bench="Ackley_10D", lo=LO, hi=HI, shift=SHIFT, noise=NOISE, n_init=N_INIT, n_iter=N_ITER),
    )
    return r


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    if len(sys.argv) > 3:            # smoke: override iteration count
        N_ITER = int(sys.argv[3])
    tag = f"Ackley_10D__{arm}__seed{seed}"
    r = run(arm, seed)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, tag + ".json") if len(sys.argv) <= 3 else os.path.join(RES, "smoke_" + tag + ".json")
    _atomic(out, r)
    print(f"[done] {tag} regret_true={r['final_regret']:.4f} regret_obs={r['final_regret_obs']:.4f} "
          f"n_eval={r['n_eval']}/{r['n_expected']} len(mean/min/max)={r['realised_len_mean']:.2f}/"
          f"{r['realised_len_min']}/{r['realised_len_max']} wall={r['_wall_s']/60:.1f}m", flush=True)
