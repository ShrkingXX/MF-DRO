"""h178 trained-embedding probe (h177 worker with the emb field active) (h168 worker with the BTG axis active). Runs an existing arm unchanged, with the read-only
RNG-neutral probe in mf_dro.py switched on: at every real iteration the DT is
re-queried at the SAME state across a sweep of RTG values and each emitted x is
recorded. The arm's own query is untouched.

usage: worker.py <bench> <seed> <arm>       arm in {random, control}
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
from src.policy.mf_dro import DirectMFRegretOptimization as _DMRO

RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_ORIG_BUILD = h83._build_mf_dro_config
ARM = sys.argv[3] if len(sys.argv) > 3 else "random"

# RTG sweep: the real target sits near 0.30 for a failing arm and ~0.98 for the
# control. The in-support end (~0.0-0.05) is where the failing arms' trajectories
# actually live (h156: mean rtg[0] +0.008 to +0.020).
SWEEP = [0.0, 0.30, 1.00]          # RTG: h168 already showed this axis is inert
BTG_SWEEP = [20.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0]   # spans the observed 26.55-30.17


def _build(*a, **k):
    cfg = _ORIG_BUILD(*a, **k)
    if ARM == "random":
        cfg.rollout_policy = "random"
    return cfg


h83._build_mf_dro_config = _build
_ORIG_INIT = _DMRO.__init__


def _init(self, *a, **k):
    _ORIG_INIT(self, *a, **k)
    self._h168_probe = SWEEP
    self._h177_btg_probe = BTG_SWEEP


_DMRO.__init__ = _init

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__EMB-{ARM.upper()}__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h178"] = dict(arm=ARM, rtg_sweep=SWEEP, btg_sweep=BTG_SWEEP,
                      n_probe_iters=len(r.get("h168_probe", [])))
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} "
          f"probe_iters={r['_h168']['n_probe_iters']} wall={r['_wall_s']/60:.1f}m", flush=True)
