"""h194 Stage 0 -- does the sliding window change the DT's decision on CURRENT code?

h27 (2026-08-24) found the K=1 and K=8 proposals bit-identical. 33 commits have touched
the DT/policy since, three changing behaviour. And h27's null contradicts h185: a
per-timestep constant predictor with 13-25% between-tau variance, read out at position
T-1 instead of 0, should emit a DIFFERENT action.

Two short runs at the same seed differing ONLY in inference_context_k. Captures every
proposal and compares.
"""
import os, sys, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import src.policy.mf_dro as MF

CAP = {}


def run_k(K, budget=16.0):
    CAP[K] = {"x": [], "ctx": []}
    _orig = MF.DirectMFRegretOptimization.__init__

    def _init(self, *a, **kw):
        _orig(self, *a, **kw)
        self.inference_context_k = K
    MF.DirectMFRegretOptimization.__init__ = _init

    spec = importlib.util.spec_from_file_location(
        "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
    h83 = importlib.util.module_from_spec(spec)
    sys.modules["h83w"] = h83
    spec.loader.exec_module(h83)
    h83.BUDGET = budget
    _ob = h83._build_mf_dro_config

    def _b(*a, **kw):
        c = _ob(*a, **kw)
        c.use_roi = True
        c.roi_beta_mode = 'quantile'
        c.roi_target_accept = 0.10        # ROI-Q10, as Stage 1 will use
        c.inference_context_k = K
        return c
    h83._build_mf_dro_config = _b
    r = h83.run("Borehole_8D", "MF-DRO", 42,
                os.path.join(os.environ.get("SCRATCH", "/tmp"), f"h194_s0_K{K}.json"))
    MF.DirectMFRegretOptimization.__init__ = _orig
    return np.array(r["x_t_trace"], dtype=float)


if __name__ == "__main__":
    a = run_k(1)
    b = run_k(8)
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n]).max(axis=1)
    print(f"\n  iterations compared: {n}")
    print(f"  max |dx| per iteration: {np.round(d, 8)[:12]}")
    print(f"  iterations differing (>1e-6): {int((d > 1e-6).sum())}/{n}")
    print(f"  overall max |dx| = {d.max():.3e}")
    gpass = (d > 1e-6).sum() >= max(1, n // 2)
    verdict = ("G-PASS -- window changes the decision, h27 is STALE, Stage 1 is live"
               if gpass else
               "G-FAIL -- bit-identical, h27 REPLICATES on current code")
    print("\n  GATE: " + verdict)
