"""h199 -- ORACLE ceiling of the lookahead schema. NOT A METHOD. usage: worker.py <bench> <seed>"""
import os, sys, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location(
    "h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
h83 = importlib.util.module_from_spec(_s); sys.modules["h83w"] = h83; _s.loader.exec_module(h83)
RES = os.path.abspath(os.path.join(H, "..", "results")); h83.RES = RES
from benchmarks import get_benchmark
_OB = h83._build_mf_dro_config

def _build(exp, base, variant, seed, **k):
    c = _OB(exp, base, variant, seed, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.rollout_policy = 'regret_lookahead'
    c.rollout_reward = 'mes_entropy'
    c.teacher_lookahead_nc = 4
    # Oracle futures are DETERMINISTIC, so M>1 would be M identical copies.
    c.teacher_lookahead_M = 1
    c.teacher_lookahead_base_pool = 150
    c.teacher_lookahead_oracle = {"H": get_benchmark(base + "_HF")["make_objective"](),
                                  "L": get_benchmark(base + "_LF")["make_objective"]()}
    return c

h83._build_mf_dro_config = _build

if __name__ == "__main__":
    bench, seed = sys.argv[1], int(sys.argv[2])
    tag = f"{bench}__H199-ORACLE-LOOK__seed{seed}"
    r = h83.run(bench, "MF-DRO", seed, os.path.join(RES, "ckpt", tag + ".json"))
    r["_h199"] = dict(ceiling=True, not_a_method=True, nc=4, M=1, base_pool=150, roi="Q10")
    h83._atomic(os.path.join(RES, tag + ".json"), r)
    print(f"[done] {tag} regret={r['final_regret']:.4f} wall={r['_wall_s']/60:.1f}m", flush=True)
