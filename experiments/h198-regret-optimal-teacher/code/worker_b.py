"""h198b -- regret-lookahead teacher, RTG labelled 'improvement'. usage: worker_b.py <bench> <seed>"""
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
    c.use_roi = True                      # ROI-Q10, matching every control
    c.roi_beta_mode = 'quantile'
    c.roi_target_accept = 0.10
    c.rollout_policy = 'regret_lookahead' # h198: optimise the TASK
    c.rollout_reward = 'improvement'              # the arm's label
    c.teacher_lookahead_nc = 4            # Stage 0 shipping config
    c.teacher_lookahead_M = 4
    c.teacher_lookahead_base_pool = 150
    return c

h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H198B-LOOK-IMP__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h198"] = dict(arm="b", rollout_policy="regret_lookahead",
                      rollout_reward="improvement", nc=4, M=4, base_pool=150, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} lf_frac={r.get('lf_fraction')} "
          f"wall={r['_wall_s']/60:.1f}m", flush=True)
