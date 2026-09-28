"""h215 arm TIBOTH -- CTRL-K1 + terminal_improvement + percentile target schema + RNG parity: both deployment defects removed at once. usage: worker_K1TI.py <bench> <seed>

The K=1 base: inference_context_k=1, positional embedding ON, mes_entropy label -- i.e.
h84's ROI-Q10 exactly, which h210 showed reproduces the last full run bit-identically.
The ONLY difference from that control is the random half, so any delta is the random
rollouts and nothing else. Label is deliberately NOT terminal_improvement here.
"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
_OB = h83._build_mf_dro_config

def _build(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10   # h84 ROI-Q10
    c.rtg_rng_parity = True                # h213: draw+discard b_tau so the RNG stream matches mes_entropy
    c.rtg_target_schema = 'percentile'     # h212: un-pin the inference RTG target
    c.rtg_target_q = 90.0


    return c
h83._build_mf_dro_config = _build
h83.ROLLOUT_REWARD = "terminal_improvement"   # the ONLY change vs CTRL-K1

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H215TIBOTH__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h215"] = dict(arm="TIBOTH", rtg_rng_parity=True, rtg_target_schema="percentile", base="h84 ROI-Q10 (K=1, pos-emb ON, mes_entropy)",
                      rollout_mix="None (default: 20 MES per member)",
                      rollout_reward="terminal_improvement", inference_context_k=1)
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
